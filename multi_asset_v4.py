"""monthly dual momentum. SPY, TLT, GLD, QQQ, EEM. 2015-2024."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
np.random.seed(42)

# --- parameters ---
tickers = ["SPY", "TLT", "GLD", "QQQ", "EEM"]
benchmark = "SPY"
start_date = "2015-01-01"
end_date = "2024-12-31"
risk_free_rate = 0.02

REL_MOM_WINDOW = 252  # was 126. 12m ranked a bit better on the full sample

DEFENSIVE_BASKET = ["TLT", "GLD"]
CASH_TICKER = "CASH"

MOM_BLEND_WINDOWS = [63, 126, 252]  # 3m / 6m / 12m, averaged for the gate

# Transaction costs
COST_BPS = 5.0
PORTFOLIO_NOTIONAL = 10_000_000

# Liquidity
ADV_WINDOW = 20
PARTICIPATION_WARN = 0.05

MC_SIMULATIONS = 5_000
MC_BLOCK_SIZE = 20             # sharpe significance test in the validation script
STRATEGY_BLOCK = 126           # ~6 months. shorter blocks chop up the trends this rule trades
MC_BATCH_SIZE = 250
MC_FAN_CHART_SAMPLE = 1_000
FORECAST_YEARS = 5

DB_PATH = Path(__file__).with_name("backtest_v4.db")


# --- pull data (prices + volume) ---
def download_data():
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError(f"Expected 'Close' in downloaded data. Got: {data.columns}")
        prices = data["Close"].copy()
        volume = data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else None
    else:
        prices = data.copy()
        volume = None

    prices = prices.dropna(how="all").ffill().dropna()
    available = [c for c in tickers if c in prices.columns]
    prices = prices[available]

    if volume is not None:
        volume = volume[available].reindex(prices.index).ffill()

    if benchmark not in prices.columns:
        raise ValueError(f"{benchmark} not found in price data.")
    if prices.empty:
        raise ValueError("Price data is empty after cleaning.")

    print("Using assets:", list(prices.columns))
    return prices, volume


# --- signals ---
def compute_signals(prices):
    returns = prices.pct_change().dropna()

    rel_mom = prices.pct_change(REL_MOM_WINDOW)
    vol20 = returns.rolling(20).std()

    blend_parts = [prices.pct_change(w) for w in MOM_BLEND_WINDOWS]
    abs_mom_blend = sum(blend_parts) / len(blend_parts)

    return returns, rel_mom, vol20, abs_mom_blend


# last trading day of each month. v3 grabbed calendar month-ends and missed some.
def month_end_trading_dates(prices):
    s = prices.index.to_series()
    month_ends = s.groupby(s.index.to_period("M")).last()
    return pd.DatetimeIndex(month_ends.values)


def get_valid_dates(prices, abs_mom_blend, rel_mom):
    month_ends = month_end_trading_dates(prices)
    valid_dates = month_ends.intersection(abs_mom_blend.dropna().index)
    valid_dates = valid_dates.intersection(rel_mom.dropna().index)
    if len(valid_dates) == 0:
        raise ValueError("No valid rebalance dates found.")
    if len(valid_dates) < 24:
        print(f"WARNING: only {len(valid_dates)} rebalance periods found (<24).")
    else:
        print(f"Rebalance periods: {len(valid_dates)} monthly periods.")
    return valid_dates


# --- build monthly weights ---
def build_monthly_weights(prices, abs_mom_blend, rel_mom, vol20, valid_dates):
    all_cols = list(prices.columns) + [CASH_TICKER]
    monthly_weights = pd.DataFrame(0.0, index=valid_dates, columns=all_cols)

    defensive_candidates = [t for t in DEFENSIVE_BASKET if t in prices.columns]

    for date in valid_dates:
        abs_signal = abs_mom_blend.loc[date].dropna()
        rel_signal = rel_mom.loc[date].dropna()

        positive_assets = abs_signal[abs_signal > 0].index.tolist()
        w = pd.Series(0.0, index=all_cols)

        if len(positive_assets) == 0:
            scores = {}
            for cand in defensive_candidates:
                if cand in abs_signal.index:
                    scores[cand] = abs_signal[cand]
            scores[CASH_TICKER] = risk_free_rate  # flat 2%, not a trailing return

            best = max(scores, key=scores.get)
            w[best] = 1.0
        else:
            rel_subset = rel_signal.loc[positive_assets].sort_values(ascending=False)
            top_assets = rel_subset.head(2).index.tolist()

            current_vol = vol20.loc[date, top_assets].replace(0, np.nan).dropna()

            if len(current_vol) == len(top_assets):
                inv_vol = 1.0 / current_vol
                w[top_assets] = inv_vol / inv_vol.sum()
            else:
                w[top_assets] = 1.0 / len(top_assets)

        monthly_weights.loc[date] = w

    return monthly_weights


# --- apply weights daily, 1 day lag ---
def apply_weights(monthly_weights, returns_ext):
    weights = monthly_weights.reindex(returns_ext.index).ffill().fillna(0.0)
    weights = weights.shift(1).fillna(0.0)
    return weights


def equal_weight_returns(prices, valid_dates, returns):
    """Same five ETFs, equal weight, same month-ends, lag, and cost."""
    cols = list(prices.columns)
    monthly = pd.DataFrame(0.0, index=valid_dates, columns=cols)
    monthly[cols] = 1.0 / len(cols)
    daily = apply_weights(monthly, returns)
    gross = (daily * returns).sum(axis=1)
    costs = transaction_costs(monthly)
    return apply_costs_to_returns(gross, costs, returns.index)


def build_extended_returns(returns):
    """cash earns rf/252 per day."""
    returns_ext = returns.copy()
    returns_ext[CASH_TICKER] = risk_free_rate / 252
    return returns_ext


# --- transaction costs (same as v3) ---
def transaction_costs(monthly_weights, cost_bps=COST_BPS):
    turnover = monthly_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = monthly_weights.iloc[0].abs().sum()
    cost_drag = turnover * (cost_bps / 10_000.0)
    cost_table = pd.DataFrame({
        "turnover": turnover,
        "cost_bps_applied": cost_bps,
        "cost_drag": cost_drag,
        "cost_dollar": cost_drag * PORTFOLIO_NOTIONAL,
    })
    return cost_table


def apply_costs_to_returns(portfolio_returns, cost_table, returns_index):
    net_returns = portfolio_returns.copy()
    for date, row in cost_table.iterrows():
        pos = returns_index.searchsorted(date) + 1
        if pos < len(returns_index):
            target_day = returns_index[pos]
            net_returns.loc[target_day] -= row["cost_drag"]
    return net_returns


# --- liquidity check (cash has no volume, skip it) ---
def liquidity_evaluation(prices, volume, monthly_weights):
    if volume is None:
        print("No volume data available; skipping liquidity evaluation.")
        return None

    dollar_volume = prices * volume
    adv = dollar_volume.rolling(ADV_WINDOW).mean()

    rows = []
    for date in monthly_weights.index:
        if date not in adv.index:
            continue
        adv_row = adv.loc[date]
        for ticker in monthly_weights.columns:
            if ticker == CASH_TICKER:
                continue
            w = monthly_weights.loc[date, ticker]
            if w == 0 or pd.isna(adv_row.get(ticker, np.nan)) or adv_row[ticker] == 0:
                continue
            position_dollar = w * PORTFOLIO_NOTIONAL
            participation = position_dollar / adv_row[ticker]
            rows.append({
                "date": date,
                "ticker": ticker,
                "weight": w,
                "position_dollar": position_dollar,
                "adv_20d": adv_row[ticker],
                "participation_rate": participation,
                "liquidity_flag": participation > PARTICIPATION_WARN,
            })

    liquidity_df = pd.DataFrame(rows)
    n_flags = liquidity_df["liquidity_flag"].sum() if not liquidity_df.empty else 0
    print(f"Liquidity check: {n_flags} position/date combinations exceed "
          f"{PARTICIPATION_WARN:.0%} of 20D ADV (assumed notional ${PORTFOLIO_NOTIONAL:,.0f}).")
    return liquidity_df


# --- performance metrics ---
def sharpe_ratio(r, rf=0.02):
    r = pd.Series(r).dropna()
    if r.std() == 0 or len(r) < 2:
        return np.nan
    excess = r - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(cum_series):
    running_max = cum_series.cummax()
    drawdown = cum_series / running_max - 1
    return drawdown.min()


def rolling_24m_stats(portfolio_returns, window_days=252 * 2):
    # older version did prod(2+r). roll the returns themselves.
    roll_return = portfolio_returns.rolling(window_days).apply(
        lambda x: np.prod(1 + x) ** (252 / window_days) - 1, raw=True
    )
    roll_sharpe = portfolio_returns.rolling(window_days).apply(
        lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
    )
    return roll_return, roll_sharpe


def _rebalance_positions(prices):
    month_ends = month_end_trading_dates(prices)
    loc = prices.index.get_indexer(month_ends)
    need = max(REL_MOM_WINDOW, max(MOM_BLEND_WINDOWS), 20)
    return loc[loc >= need]


def apply_rule(prices, asset_returns, cost_bps=COST_BPS):
    """Replay the monthly rule on a panel of asset returns.

    asset_returns is (n_paths, n_days, n_assets) in the same column order as
    prices, and n_days is len(prices) - 1. Decision on day t earns day t+1.
    """
    names = list(prices.columns)
    r = np.asarray(asset_returns, dtype=float)
    n_paths, n_ret, n_assets = r.shape
    if n_assets != len(names) or n_ret != len(prices) - 1:
        raise ValueError("return panel doesn't line up with prices")

    growth = np.cumprod(1.0 + r, axis=1)

    def mom(t, window):
        num = growth[:, t - 1, :]
        den = 1.0 if t == window else growth[:, t - window - 1, :]
        return num / den - 1.0

    def_idx = [names.index(t) for t in DEFENSIVE_BASKET if t in names]
    acts = []
    held = []
    for t in _rebalance_positions(prices):
        act = int(t)
        if act >= n_ret:
            continue
        blend = sum(mom(t, w) for w in MOM_BLEND_WINDOWS) / len(MOM_BLEND_WINDOWS)
        rel = mom(t, REL_MOM_WINDOW)
        vol = r[:, t - 20:t, :].std(axis=1, ddof=1)

        w = np.zeros((n_paths, n_assets + 1))
        positive = np.isfinite(blend) & np.isfinite(rel) & (blend > 0)
        n_pos = positive.sum(axis=1)

        scores = np.empty((n_paths, len(def_idx) + 1))
        for j, di in enumerate(def_idx):
            scores[:, j] = blend[:, di]
        scores[:, -1] = risk_free_rate
        best = scores.argmax(axis=1)
        for j, di in enumerate(def_idx):
            w[best == j, di] = 1.0
        w[best == len(def_idx), n_assets] = 1.0

        rel_masked = np.where(positive, rel, -np.inf)
        part = np.argpartition(rel_masked, -2, axis=1)[:, -2:]
        part_rel = np.take_along_axis(rel_masked, part, axis=1)
        order = np.argsort(part_rel, axis=1)
        top2 = np.take_along_axis(part, order, axis=1)
        vol_top = np.take_along_axis(vol, top2, axis=1)
        rows = np.arange(n_paths)

        w[n_pos >= 1] = 0.0
        one = n_pos == 1
        w[rows[one], top2[one, 1]] = 1.0

        two = n_pos >= 2
        good_vol = np.isfinite(vol_top) & (vol_top > 0)
        use_inv = two & good_vol.all(axis=1)
        if use_inv.any():
            iv = 1.0 / vol_top[use_inv]
            iv /= iv.sum(axis=1, keepdims=True)
            w[rows[use_inv], top2[use_inv, 0]] = iv[:, 0]
            w[rows[use_inv], top2[use_inv, 1]] = iv[:, 1]
        equal = two & ~use_inv
        w[rows[equal], top2[equal, 0]] = 0.5
        w[rows[equal], top2[equal, 1]] = 0.5

        acts.append(act)
        held.append(w)

    daily_w = np.zeros((n_paths, n_ret, n_assets + 1))
    for i, act in enumerate(acts):
        end = acts[i + 1] if i + 1 < len(acts) else n_ret
        daily_w[:, act:end, :] = held[i][:, None, :]

    port = (daily_w[:, :, :n_assets] * r).sum(axis=2)
    port += daily_w[:, :, n_assets] * (risk_free_rate / 252.0)

    prev = np.zeros((n_paths, n_assets + 1))
    unit_cost = cost_bps / 10_000.0
    for i, act in enumerate(acts):
        turnover = np.abs(held[i] - prev).sum(axis=1)
        port[:, act] -= turnover * unit_cost
        prev = held[i]

    first_on = acts[0] if acts else 0
    return port, first_on


def _path_stats(port, rf=risk_free_rate):
    cum = np.cumprod(1.0 + port, axis=1)
    terminal = cum[:, -1] - 1.0
    excess = port - rf / 252.0
    sd = excess.std(axis=1, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    sharpe = np.sqrt(252) * excess.mean(axis=1) / sd
    running_max = np.maximum.accumulate(cum, axis=1)
    drawdown = (cum / running_max - 1.0).min(axis=1)
    return terminal, sharpe, drawdown, cum


def monte_carlo_simulate(prices, n_sims=MC_SIMULATIONS, block_size=STRATEGY_BLOCK,
                          batch_size=MC_BATCH_SIZE, fan_sample=MC_FAN_CHART_SAMPLE):
    """Block-bootstrap the asset returns together, then rerun the rule.

    Same block dates for every ETF, so the daily correlation stays. Reshuffling
    the finished strategy P&L doesn't do that: the real history just lands in
    the middle of a shuffle of itself.
    """
    raw = prices.pct_change()
    if raw.iloc[1:].isna().any().any():
        raise ValueError("missing returns after the first day")
    realized = raw.iloc[1:].to_numpy(dtype=float)
    n, n_assets = realized.shape
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)

    terminal_returns = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    drawdowns = np.empty(n_sims)
    year_returns = []
    fan_paths = []
    fan_collected = 0
    done = 0

    while done < n_sims:
        cur = min(batch_size, n_sims - done)
        starts = np.random.randint(0, n - block_size + 1, size=(cur, n_blocks))
        idx = starts[:, :, None] + offsets
        picked = idx.reshape(cur, n_blocks * block_size)[:, :n]
        batch = realized[picked]

        port, first_on = apply_rule(prices, batch)
        terminal, sharpe, drawdown, cum = _path_stats(port)
        terminal_returns[done:done + cur] = terminal
        sharpes[done:done + cur] = sharpe
        drawdowns[done:done + cur] = drawdown

        usable = port[:, first_on:]
        n_years = usable.shape[1] // 252
        if n_years:
            years = usable[:, :n_years * 252].reshape(cur, n_years, 252)
            year_returns.append(np.prod(1.0 + years, axis=2) - 1.0)

        if fan_collected < fan_sample:
            take = min(fan_sample - fan_collected, cur)
            fan_paths.append(cum[:take])
            fan_collected += take
        done += cur

    summary = {
        "terminal_return": terminal_returns,
        "sharpe": sharpes,
        "max_drawdown": drawdowns,
    }
    return summary, np.vstack(fan_paths), np.concatenate(year_returns, axis=0)


def print_mc_report(summary, year_returns, realized_terminal_return, realized_sharpe, realized_mdd):
    n = len(summary["terminal_return"])
    print("\nmonte carlo: rule rerun on reshuffled asset returns")
    print(f"{n:,} paths, {STRATEGY_BLOCK}-day blocks, assets moved together")
    for label, key, realized in [
        ("Terminal Return", "terminal_return", realized_terminal_return),
        ("Sharpe Ratio", "sharpe", realized_sharpe),
        ("Max Drawdown", "max_drawdown", realized_mdd),
    ]:
        arr = summary[key]
        p5, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
        better = (arr < realized).mean() * 100
        print(f"  {label:<16} P5={p5:8.3f}  P50={p50:8.3f}  P95={p95:8.3f}  "
              f"| history={realized:8.3f}  (better than {better:4.1f}% of paths)")

    flat = year_returns.ravel()
    p5 = np.percentile(flat, 5)
    tail = flat[flat <= p5]
    print(f"\n  1y return, inside the reshuffled histories")
    print(f"    5th percentile      {p5:8.2%}")
    print(f"    mean of the worst 5% {tail.mean():8.2%}")


def forecast_paths(prices, n_sims=MC_SIMULATIONS, horizon=FORECAST_YEARS * 252,
                    block_size=STRATEGY_BLOCK, batch_size=MC_BATCH_SIZE,
                    fan_sample=MC_FAN_CHART_SAMPLE):
    """Paths from the last real close. Blocks are drawn from history, but the
    lookback in front of the forecast is the actual recent prices, and the
    rule is rerun as the path unfolds."""
    names = list(prices.columns)
    hist = prices.pct_change().iloc[1:]
    if hist.isna().any().any():
        raise ValueError("missing returns after the first day")
    hist = hist.to_numpy(dtype=float)
    n_hist, n_assets = hist.shape

    tail_n = max(REL_MOM_WINDOW, max(MOM_BLEND_WINDOWS)) + 21
    tail = prices.iloc[-tail_n:]
    future_idx = pd.bdate_range(tail.index[-1] + pd.Timedelta(days=1), periods=horizon)
    cal = tail.index.append(future_idx)
    stub = pd.DataFrame(1.0, index=cal, columns=names)
    real_ret = tail.pct_change().iloc[1:].to_numpy(dtype=float)
    forecast_start = len(real_ret)

    n_blocks = int(np.ceil(horizon / block_size))
    offsets = np.arange(block_size)
    terminal_returns = np.empty(n_sims)
    cagrs = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    drawdowns = np.empty(n_sims)
    return_1y = np.empty(n_sims)
    fan_paths = []
    fan_collected = 0
    done = 0

    while done < n_sims:
        cur = min(batch_size, n_sims - done)
        starts = np.random.randint(0, n_hist - block_size + 1, size=(cur, n_blocks))
        idx = starts[:, :, None] + offsets
        future = hist[idx.reshape(cur, n_blocks * block_size)[:, :horizon]]
        real = np.repeat(real_ret[None, :, :], cur, axis=0)
        panel = np.concatenate([real, future], axis=1)

        port, _ = apply_rule(stub, panel)
        fwd = port[:, forecast_start:]
        terminal, sharpe, drawdown, cum = _path_stats(fwd)
        terminal_returns[done:done + cur] = terminal
        cagrs[done:done + cur] = (1.0 + terminal) ** (1.0 / FORECAST_YEARS) - 1.0
        sharpes[done:done + cur] = sharpe
        drawdowns[done:done + cur] = drawdown
        return_1y[done:done + cur] = np.prod(1.0 + fwd[:, :252], axis=1) - 1.0

        if fan_collected < fan_sample:
            take = min(fan_sample - fan_collected, cur)
            fan_paths.append(cum[:take])
            fan_collected += take
        done += cur

    summary = {
        "terminal_return": terminal_returns,
        "cagr": cagrs,
        "sharpe": sharpes,
        "max_drawdown": drawdowns,
        "return_1y": return_1y,
    }
    return summary, np.vstack(fan_paths), future_idx


def print_forecast(summary):
    print(f"\nforecast from the last close, next {FORECAST_YEARS} years")
    print(f"{len(summary['terminal_return']):,} paths, {STRATEGY_BLOCK}-day blocks, rule rerun")
    for label, key in [
        ("1y return", "return_1y"),
        ("5y growth", "terminal_return"),
        ("5y CAGR", "cagr"),
        ("5y Sharpe", "sharpe"),
        ("5y max drawdown", "max_drawdown"),
    ]:
        arr = summary[key]
        p5, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
        print(f"  {label:<16} P5={p5:8.3f}  P50={p50:8.3f}  P95={p95:8.3f}")
    y1 = summary["return_1y"]
    term = summary["terminal_return"]
    worst = y1[y1 <= np.percentile(y1, 5)]
    print(f"  share of paths with a 1y loss: {np.mean(y1 < 0):.1%}")
    print(f"  share of paths with a 5y loss: {np.mean(term < 0):.1%}")
    print(f"  mean of the worst 5% of 1y returns: {worst.mean():.2%}")


def _hist(arr, bins):
    counts, edges = np.histogram(arr, bins=bins)
    labels = [f"{edges[i]:.0%}" for i in range(len(counts))]
    return labels, counts.tolist(), edges


def save_forecast_plots(fan, summary, future_idx, realized_1y):
    import json
    pct = np.percentile(fan, [5, 25, 50, 75, 95], axis=0)
    pct = np.concatenate([np.ones((5, 1)), pct], axis=1)
    years = np.arange(pct.shape[1]) / 252.0

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.fill_between(years, pct[0], pct[4], color="steelblue", alpha=0.18, label="5th-95th")
    ax.fill_between(years, pct[1], pct[3], color="steelblue", alpha=0.32, label="25th-75th")
    ax.plot(years, pct[2], color="black", linewidth=2.0, label="Median")
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":")
    ax.set_xlim(0, years[-1] + 0.55)
    ax.set_title(f"Next {FORECAST_YEARS} years, {MC_SIMULATIONS:,} reruns of the rule")
    ax.set_xlabel("Years from the last close")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y")
    for band, name in ((0, "5th"), (2, "median"), (4, "95th")):
        ax.annotate(f"{name}  {pct[band, -1]:.2f}",
                    xy=(years[-1], pct[band, -1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("forecast_fan_v4.png", dpi=140)
    plt.close(fig)

    y1 = summary["return_1y"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.hist(y1, bins=40, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.4)
    marks = [
        (np.percentile(y1, 5), "5th", "right"),
        (np.percentile(y1, 95), "95th", "left"),
    ]
    ax.axvline(np.percentile(y1, 50), color="black", linewidth=1.2, label="forecast median")
    for val, name, ha in marks:
        ax.axvline(val, color="black", linewidth=1.2)
        ax.text(val, ax.get_ylim()[1] * 0.97, f"{name} {val:.0%}", fontsize=9, va="top", ha=ha)
    if realized_1y is not None and len(realized_1y):
        hist_med = float(np.median(realized_1y))
        ax.axvline(hist_med, color="#a35f1b", linewidth=1.6, linestyle="--",
                   label=f"historical 1y median ({hist_med:.0%})")
        ax.legend(frameon=False)
    ax.set_title("Distribution of the next 1-year return")
    ax.set_xlabel("Return")
    ax.set_ylabel("Paths")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig("forecast_1y_v4.png", dpi=140)
    plt.close(fig)

    dd = summary["max_drawdown"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.hist(dd, bins=40, color="steelblue", alpha=0.85, edgecolor="white", linewidth=0.4)
    for q, name, ha in ((5, "deeper, 5th", "right"), (50, "median", "left"), (95, "shallower, 95th", "left")):
        val = np.percentile(dd, q)
        ax.axvline(val, color="black", linewidth=1.2)
        ax.text(val, ax.get_ylim()[1] * 0.97, f"{name} {val:.0%}", fontsize=9, va="top", ha=ha)
    ax.set_title(f"Worst drawdown over the next {FORECAST_YEARS} years")
    ax.set_xlabel("Max drawdown")
    ax.set_ylabel("Paths")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig("forecast_drawdown_v4.png", dpi=140)
    plt.close(fig)

    step = 63
    idx = list(range(0, pct.shape[1], step))
    if idx[-1] != pct.shape[1] - 1:
        idx.append(pct.shape[1] - 1)
    as_of = future_idx[0] - pd.tseries.offsets.BDay(1)
    labels = [as_of.strftime("%b %Y")] + [future_idx[i - 1].strftime("%b %Y") for i in idx[1:]]
    y1_labels, y1_counts, _ = _hist(y1, bins=18)
    dd_labels, dd_counts, _ = _hist(dd, bins=12)

    def pcts(arr):
        p5, p50, p95 = [float(x) for x in np.nanpercentile(arr, [5, 50, 95])]
        return {"p5": p5, "p50": p50, "p95": p95}

    payload = {
        "n_paths": int(len(y1)),
        "block_days": STRATEGY_BLOCK,
        "years": FORECAST_YEARS,
        "as_of": str(as_of.date()),
        "p_loss_1y": float(np.mean(y1 < 0)),
        "p_loss_5y": float(np.mean(summary["terminal_return"] < 0)),
        "worst5_1y": float(y1[y1 <= np.percentile(y1, 5)].mean()),
        "return_1y": pcts(y1),
        "terminal": pcts(summary["terminal_return"]),
        "cagr": pcts(summary["cagr"]),
        "sharpe": pcts(summary["sharpe"]),
        "max_drawdown": pcts(dd),
        "fan_labels": labels,
        "fan_p5": [round(float(pct[0, i]), 4) for i in idx],
        "fan_p50": [round(float(pct[2, i]), 4) for i in idx],
        "fan_p95": [round(float(pct[4, i]), 4) for i in idx],
        "y1_labels": y1_labels,
        "y1_counts": y1_counts,
        "dd_labels": dd_labels,
        "dd_counts": dd_counts,
        "historical_1y_median": None if realized_1y is None or len(realized_1y) == 0 else float(np.median(realized_1y)),
    }
    with open("forecast_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("\nSaved plots: forecast_fan_v4.png, forecast_1y_v4.png, forecast_drawdown_v4.png")


# --- save everything to sqlite ---
def save_to_sql(prices, monthly_weights, cost_table, portfolio_returns, net_returns,
                 spy_returns, liquidity_df, mc_summary, mc_fan=None, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)

    prices.reset_index().melt(id_vars=prices.index.name or "index",
                               var_name="ticker", value_name="close") \
          .rename(columns={prices.index.name or "index": "date"}) \
          .to_sql("prices", conn, if_exists="replace", index=False)

    monthly_weights.reset_index().melt(id_vars="index", var_name="ticker", value_name="weight") \
                    .rename(columns={"index": "date"}) \
                    .to_sql("weights", conn, if_exists="replace", index=False)

    cost_table.reset_index().rename(columns={"index": "date"}) \
               .to_sql("trades", conn, if_exists="replace", index=False)

    perf = pd.DataFrame({
        "portfolio_return_gross": portfolio_returns,
        "portfolio_return_net": net_returns,
        "benchmark_return": spy_returns,
    })
    perf["portfolio_cum_gross"] = (1 + perf["portfolio_return_gross"]).cumprod()
    perf["portfolio_cum_net"] = (1 + perf["portfolio_return_net"]).cumprod()
    perf["benchmark_cum"] = (1 + perf["benchmark_return"]).cumprod()
    perf.reset_index().rename(columns={"index": "date"}).to_sql(
        "performance", conn, if_exists="replace", index=False)

    if liquidity_df is not None and not liquidity_df.empty:
        liquidity_df.to_sql("liquidity", conn, if_exists="replace", index=False)

    mc_df = pd.DataFrame(mc_summary)
    mc_df.to_sql("monte_carlo_results", conn, if_exists="replace", index=False)

    if mc_fan is not None:
        bands = np.percentile(mc_fan, [5, 25, 50, 75, 95], axis=0)
        pd.DataFrame({
            "day": np.arange(bands.shape[1]),
            "p5": bands[0],
            "p25": bands[1],
            "p50": bands[2],
            "p75": bands[3],
            "p95": bands[4],
        }).to_sql("monte_carlo_fan", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"\nSaved results to SQLite DB at: {db_path}")


# --- main ---
def main():
    prices, volume = download_data()
    returns, rel_mom, vol20, abs_mom_blend = compute_signals(prices)
    valid_dates = get_valid_dates(prices, abs_mom_blend, rel_mom)
    monthly_weights = build_monthly_weights(prices, abs_mom_blend, rel_mom, vol20, valid_dates)

    returns_ext = build_extended_returns(returns)
    weights = apply_weights(monthly_weights, returns_ext)
    portfolio_returns = (weights * returns_ext).sum(axis=1)
    bench_returns = equal_weight_returns(prices, valid_dates, returns)

    cost_table = transaction_costs(monthly_weights)
    net_returns = apply_costs_to_returns(portfolio_returns, cost_table, returns_ext.index)

    liquidity_df = liquidity_evaluation(prices, volume, monthly_weights)

    portfolio_cum = (1 + portfolio_returns).cumprod()
    net_cum = (1 + net_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    index_return = portfolio_cum.iloc[-1] - 1
    net_return = net_cum.iloc[-1] - 1
    bench_return = bench_cum.iloc[-1] - 1

    index_sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
    net_sharpe = sharpe_ratio(net_returns, risk_free_rate)
    bench_sharpe = sharpe_ratio(bench_returns, risk_free_rate)

    index_mdd = max_drawdown(portfolio_cum)
    net_mdd = max_drawdown(net_cum)
    bench_mdd = max_drawdown(bench_cum)

    print("\ngross vs net vs equal weight")
    print(f"{'Metric':<20}{'Gross':>12}{'Net-of-Cost':>14}{'Equal wt':>12}")
    print(f"{'Total Return':<20}{index_return:>12.2%}{net_return:>14.2%}{bench_return:>12.2%}")
    print(f"{'Sharpe Ratio':<20}{index_sharpe:>12.2f}{net_sharpe:>14.2f}{bench_sharpe:>12.2f}")
    print(f"{'Max Drawdown':<20}{index_mdd:>12.2%}{net_mdd:>14.2%}{bench_mdd:>12.2%}")
    total_cost_drag = cost_table["cost_drag"].sum()
    print(f"\nCumulative cost drag from turnover: {total_cost_drag:.2%} "
          f"(~${(total_cost_drag * PORTFOLIO_NOTIONAL):,.0f} on ${PORTFOLIO_NOTIONAL:,.0f} notional)")

    roll_return, roll_sharpe = rolling_24m_stats(net_returns)
    roll_return_valid = roll_return.dropna()
    roll_sharpe_valid = roll_sharpe.dropna()
    if len(roll_return_valid) > 0:
        print(f"\nRolling 24M annualized return: min={roll_return_valid.min():.2%}  "
              f"median={roll_return_valid.median():.2%}  max={roll_return_valid.max():.2%}")
        print(f"Rolling 24M Sharpe:            min={roll_sharpe_valid.min():.2f}  "
              f"median={roll_sharpe_valid.median():.2f}  max={roll_sharpe_valid.max():.2f}")
        pct_negative = (roll_return_valid < 0).mean()
        print(f"Share of rolling 24M windows with a negative annualized return: {pct_negative:.1%}")

    _, roll_sharpe_1y = rolling_24m_stats(net_returns, window_days=252)
    _, roll_sharpe_1y_bench = rolling_24m_stats(bench_returns, window_days=252)

    drawdown_net = net_cum / net_cum.cummax() - 1
    drawdown_bench = bench_cum / bench_cum.cummax() - 1

    # the vectorized replay has to match the pandas backtest on the real path
    check, _ = apply_rule(prices, returns[list(prices.columns)].to_numpy()[None, ...])
    if not np.allclose(check[0], net_returns.to_numpy(), equal_nan=True):
        gap = np.nanmax(np.abs(check[0] - net_returns.to_numpy()))
        raise RuntimeError(f"replay doesn't match the backtest (max gap {gap:.3e})")

    print(f"\nRunning Monte Carlo: {MC_SIMULATIONS:,} paths "
          f"(batched {MC_BATCH_SIZE:,} at a time)...")
    mc_start = time.time()
    mc_summary, mc_cum_paths, year_returns = monte_carlo_simulate(prices)
    print(f"Monte Carlo done in {time.time() - mc_start:.1f}s.")
    print_mc_report(mc_summary, year_returns, net_return, net_sharpe, net_mdd)

    realized_1y = net_returns.rolling(252).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    ).dropna()
    print(f"\nRunning forecast: {MC_SIMULATIONS:,} paths...")
    fc_start = time.time()
    fc_summary, fc_fan, future_idx = forecast_paths(prices)
    print(f"Forecast done in {time.time() - fc_start:.1f}s.")
    print_forecast(fc_summary)
    save_forecast_plots(fc_fan, fc_summary, future_idx, realized_1y)

    save_to_sql(prices, monthly_weights, cost_table, portfolio_returns, net_returns,
                bench_returns, liquidity_df, mc_summary, mc_fan=mc_cum_paths)

    plt.figure(figsize=(12, 6))
    pct_bands = np.percentile(mc_cum_paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(mc_cum_paths.shape[1])
    plt.fill_between(x, pct_bands[0], pct_bands[4], color="steelblue", alpha=0.2, label="5th-95th pct")
    plt.fill_between(x, pct_bands[1], pct_bands[3], color="steelblue", alpha=0.35, label="25th-75th pct")
    plt.plot(x, pct_bands[2], color="steelblue", linewidth=1.5, label="Median path")
    plt.plot(x, net_cum.values, color="black", linewidth=2, label="Realized")
    plt.title(f"Rule rerun on {STRATEGY_BLOCK}-day blocks "
              f"({MC_SIMULATIONS:,} paths, {mc_cum_paths.shape[0]:,} drawn)")
    plt.xlabel("Trading day"); plt.ylabel("Growth of $1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("monte_carlo_fan_chart_v4.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cum, label="Gross", linewidth=2)
    plt.plot(net_cum, label="Net of cost", linewidth=2, linestyle="--")
    plt.plot(bench_cum, label="Equal weight", linewidth=2, linestyle=":")
    plt.title("Strategy vs equal weight")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("performance_gross_net_v4.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(roll_sharpe_1y, label="Strategy", linewidth=2)
    plt.plot(roll_sharpe_1y_bench, label="Equal weight", linewidth=2)
    plt.title("Rolling 1Y Sharpe Ratio")
    plt.xlabel("Date"); plt.ylabel("Sharpe Ratio")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("rolling_1y_sharpe_vs_ew_v4.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(drawdown_net, label="Strategy", linewidth=2)
    plt.plot(drawdown_bench, label="Equal weight", linewidth=2)
    plt.title("Drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("drawdown_comparison_v4.png", dpi=120)
    plt.close()

    if len(roll_return_valid) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(roll_sharpe, label="Rolling 24M Sharpe (net-of-cost)", linewidth=2)
        plt.axhline(0, color="gray", linewidth=1, linestyle=":")
        plt.title("Rolling 24M Sharpe")
        plt.xlabel("Date"); plt.ylabel("Sharpe Ratio")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("rolling_24m_sharpe_v4.png", dpi=120)
        plt.close()
        print("\nSaved plots: performance_gross_net_v4.png, monte_carlo_fan_chart_v4.png, "
              "rolling_1y_sharpe_vs_ew_v4.png, drawdown_comparison_v4.png, "
              "rolling_24m_sharpe_v4.png")
    else:
        print("\nSaved plots: performance_gross_net_v4.png, monte_carlo_fan_chart_v4.png, "
              "rolling_1y_sharpe_vs_ew_v4.png, drawdown_comparison_v4.png")

    print("\nAverage allocation across all rebalances:")
    avg_w = monthly_weights.mean()
    avg_w = avg_w / avg_w.sum()
    for t, w in avg_w.sort_values(ascending=False).items():
        print(f"  {t}: {w:.1%}")


if __name__ == "__main__":
    main()
