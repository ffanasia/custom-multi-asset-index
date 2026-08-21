"""
a bigger universe --
30 tickers instead of 5, now that assets.py pulled in the extra price
history. holds the top 10 instead of top 2 since there's actually enough
names here to spread across.

needs internet for yfinance, run locally.
"""

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
# 30-ticker universe: the original 5 core assets plus 25 more liquid ETFs
# for broader coverage across sectors, international, fixed income,
# commodities, real assets, small/mid cap, and a currency hedge
tickers = [
    # Core (original)
    "SPY", "TLT", "GLD", "QQQ", "EEM",
    # US equity sectors (SPDR Select Sector ETFs)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    # International equity
    "EFA", "VWO", "FXI",
    # Fixed income
    "LQD", "HYG", "IEF", "SHY", "TIP",
    # Commodities
    "SLV", "USO", "DBC",
    # Real assets
    "VNQ",
    # Small/mid cap
    "IWM", "MDY",
    # Currency hedge
    "UUP",
]
benchmark = "SPY"
start_date = "2015-01-01"
end_date = "2024-12-31"
risk_free_rate = 0.02
 
# how many top-ranked names to hold each rebalance, weighted inverse to
# 20d vol. bumped up from 2 (in the 5-asset version) to 10 now that
# there's an actual universe to diversify across
TOP_N = 10

# ranking window for relative momentum, among whatever passes the gate
# below. widened from 126 (6M) to 252 (12M) -- same reasoning as v4
REL_MOM_WINDOW = 252

# fallback basket for when nothing has positive momentum. cash is always
# an option too (earns rf rate)
DEFENSIVE_BASKET = ["TLT", "GLD"]
CASH_TICKER = "CASH"

# blending these lookback windows for the risk-on/off gate instead of just
# a raw 252-day momentum signal
MOM_BLEND_WINDOWS = [63, 126, 252]  # ~3M, 6M, 12M
 
# Transaction costs
COST_BPS = 5.0
PORTFOLIO_NOTIONAL = 10_000_000
 
# Liquidity
ADV_WINDOW = 20
PARTICIPATION_WARN = 0.05
 
# Monte Carlo
MC_SIMULATIONS = 1_000_000   # total simulated paths for the risk-metric distributions
MC_BLOCK_SIZE = 20
MC_BATCH_SIZE = 10_000       # paths processed per vectorized batch (memory control;
                             # ~190MB transient per batch for a ~2500-day history)
MC_FAN_CHART_SAMPLE = 5_000  # subsample of paths kept for the fan-chart plot (percentiles
                             # converge well before 1M paths, so plotting all of them would
                             # just burn memory/time for no visual benefit)
 
DB_PATH = Path(__file__).with_name("backtest_30.db")
 
 
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
 
    rel_mom = prices.pct_change(REL_MOM_WINDOW)  # relative momentum (ranking)
    vol20 = returns.rolling(20).std()
 
    # Blended absolute momentum for the risk-on/off gate (3M/6M/12M average).
    blend_parts = [prices.pct_change(w) for w in MOM_BLEND_WINDOWS]
    abs_mom_blend = sum(blend_parts) / len(blend_parts)
 
    return returns, rel_mom, vol20, abs_mom_blend
 
 
# --- rebalance dates (fixed a bug here in v3, fine now) ---
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
            # pick whichever of TLT/GLD/CASH is holding up best instead of
            # defaulting straight to TLT
            scores = {}
            for cand in defensive_candidates:
                if cand in abs_signal.index:
                    scores[cand] = abs_signal[cand]
            # CASH's "momentum" proxy: risk-free accrual over ~1yr blend horizon
            scores[CASH_TICKER] = risk_free_rate
 
            best = max(scores, key=scores.get)
            w[best] = 1.0
        else:
            rel_subset = rel_signal.loc[positive_assets].sort_values(ascending=False)
            top_assets = rel_subset.head(TOP_N).index.tolist()
 
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
 
 
def build_extended_returns(returns):
    """Add a synthetic CASH column earning the daily risk-free rate so the
    defensive-basket logic can allocate to cash like any other asset."""
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
    # this used to roll over (1 + returns) and then add 1 again inside the
    # lambda -- double counted the +1 every window and produced nonsense
    # numbers. rolling over raw returns directly fixes it.
    roll_return = portfolio_returns.rolling(window_days).apply(
        lambda x: np.prod(1 + x) ** (252 / window_days) - 1, raw=True
    )
    roll_sharpe = portfolio_returns.rolling(window_days).apply(
        lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
    )
    return roll_return, roll_sharpe
 
 
# --- monte carlo (block bootstrap) ---
# same as v4: batched numpy instead of a per-path python loop (which took
# hours at 1M+ sims), and only a small subsample of full paths gets kept
# for the fan chart since the percentile bands converge well before 1M.
def monte_carlo_simulate(daily_returns, n_sims=MC_SIMULATIONS, block_size=MC_BLOCK_SIZE,
                          batch_size=MC_BATCH_SIZE, fan_sample=MC_FAN_CHART_SAMPLE,
                          rf=risk_free_rate):
    r = daily_returns.dropna().values
    n = len(r)
    n_blocks = int(np.ceil(n / block_size))
    offsets = np.arange(block_size)
 
    terminal_returns = np.empty(n_sims)
    sharpes = np.empty(n_sims)
    drawdowns = np.empty(n_sims)
    fan_paths = []
    fan_collected = 0
 
    done = 0
    while done < n_sims:
        cur = min(batch_size, n_sims - done)
 
        # Draw cur * n_blocks random block-start indices at once, gather the
        # corresponding return blocks, and truncate each path back to length n.
        starts = np.random.randint(0, n - block_size + 1, size=(cur, n_blocks))
        idx = starts[:, :, None] + offsets[None, None, :]
        batch = r[idx].reshape(cur, n_blocks * block_size)[:, :n]
 
        cum = np.cumprod(1 + batch, axis=1)
        terminal_returns[done:done + cur] = cum[:, -1] - 1
 
        excess = batch - rf / 252
        sharpes[done:done + cur] = np.sqrt(252) * excess.mean(axis=1) / excess.std(axis=1)
 
        running_max = np.maximum.accumulate(cum, axis=1)
        drawdowns[done:done + cur] = (cum / running_max - 1).min(axis=1)
 
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
    fan_cum_paths = np.vstack(fan_paths)
    return summary, fan_cum_paths
 
 
def print_mc_report(summary, realized_terminal_return, realized_sharpe, realized_mdd):
    print("\n===== Monte Carlo Risk Simulation (block bootstrap) =====")
    print(f"Simulations: {len(summary['terminal_return']):,}, block size: {MC_BLOCK_SIZE} days")
    for label, key, realized in [
        ("Terminal Return", "terminal_return", realized_terminal_return),
        ("Sharpe Ratio", "sharpe", realized_sharpe),
        ("Max Drawdown", "max_drawdown", realized_mdd),
    ]:
        arr = summary[key]
        p5, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
        pct_rank = (arr < realized).mean() * 100
        print(f"  {label:<16} sim P5={p5:8.3f}  P50={p50:8.3f}  P95={p95:8.3f}  "
              f"| realized={realized:8.3f}  (percentile rank {pct_rank:5.1f}%)")
 
    var95 = -np.nanpercentile(summary["terminal_return"], 5)
    cvar95 = -summary["terminal_return"][summary["terminal_return"] <= np.nanpercentile(summary["terminal_return"], 5)].mean()
    print(f"\n  1Y-equivalent VaR(95%) on terminal return : {var95:.2%}")
    print(f"  CVaR(95%) on terminal return               : {cvar95:.2%}")
 
 
# --- save everything to sqlite ---
def save_to_sql(prices, monthly_weights, cost_table, portfolio_returns, net_returns,
                 spy_returns, liquidity_df, mc_summary, db_path=DB_PATH):
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
    spy_returns = returns[benchmark]
 
    cost_table = transaction_costs(monthly_weights)
    net_returns = apply_costs_to_returns(portfolio_returns, cost_table, returns_ext.index)
 
    liquidity_df = liquidity_evaluation(prices, volume, monthly_weights)
 
    portfolio_cum = (1 + portfolio_returns).cumprod()
    net_cum = (1 + net_returns).cumprod()
    spy_cum = (1 + spy_returns).cumprod()
 
    index_return = portfolio_cum.iloc[-1] - 1
    net_return = net_cum.iloc[-1] - 1
    spy_return = spy_cum.iloc[-1] - 1
 
    index_sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
    net_sharpe = sharpe_ratio(net_returns, risk_free_rate)
    spy_sharpe = sharpe_ratio(spy_returns, risk_free_rate)
 
    index_mdd = max_drawdown(portfolio_cum)
    net_mdd = max_drawdown(net_cum)
    spy_mdd = max_drawdown(spy_cum)
 
    print("\n===== Performance Comparison (Gross vs Net-of-Cost vs SPY), 30 asset =====")
    print(f"{'Metric':<20}{'Gross':>12}{'Net-of-Cost':>14}{'SPY':>12}")
    print(f"{'Total Return':<20}{index_return:>12.2%}{net_return:>14.2%}{spy_return:>12.2%}")
    print(f"{'Sharpe Ratio':<20}{index_sharpe:>12.2f}{net_sharpe:>14.2f}{spy_sharpe:>12.2f}")
    print(f"{'Max Drawdown':<20}{index_mdd:>12.2%}{net_mdd:>14.2%}{spy_mdd:>12.2%}")
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
 
    # rolling 1y sharpe vs spy, same as v4
    _, roll_sharpe_1y = rolling_24m_stats(net_returns, window_days=252)
    _, roll_sharpe_1y_spy = rolling_24m_stats(spy_returns, window_days=252)

    # full drawdown series, not just the min, for the same plots
    drawdown_net = net_cum / net_cum.cummax() - 1
    drawdown_spy = spy_cum / spy_cum.cummax() - 1
 
    print(f"\nRunning Monte Carlo: {MC_SIMULATIONS:,} simulations "
          f"(batched {MC_BATCH_SIZE:,} at a time)...")
    mc_start = time.time()
    mc_summary, mc_cum_paths = monte_carlo_simulate(net_returns)
    print(f"Monte Carlo done in {time.time() - mc_start:.1f}s.")
    print_mc_report(mc_summary, net_return, net_sharpe, net_mdd)
 
    save_to_sql(prices, monthly_weights, cost_table, portfolio_returns, net_returns,
                spy_returns, liquidity_df, mc_summary)
 
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cum, label="30-Asset Index (Gross)", linewidth=2)
    plt.plot(net_cum, label="30-Asset Index (Net of Cost)", linewidth=2, linestyle="--")
    plt.plot(spy_cum, label="SPY", linewidth=2, linestyle=":")
    plt.title("30-Asset Multi-Asset Index vs SPY")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("performance_gross_net_30.png", dpi=120)
    plt.close()

    plt.figure(figsize=(12, 6))
    pct_bands = np.percentile(mc_cum_paths, [5, 25, 50, 75, 95], axis=0)
    x = np.arange(mc_cum_paths.shape[1])
    plt.fill_between(x, pct_bands[0], pct_bands[4], color="steelblue", alpha=0.2, label="5th-95th pct")
    plt.fill_between(x, pct_bands[1], pct_bands[3], color="steelblue", alpha=0.35, label="25th-75th pct")
    plt.plot(x, pct_bands[2], color="steelblue", linewidth=1.5, label="Median simulated path")
    plt.plot(x, net_cum.values, color="black", linewidth=2, label="Realized (net-of-cost)")
    plt.title(f"30-Asset Monte Carlo Block-Bootstrap Fan Chart "
              f"({MC_SIMULATIONS:,} sims, {mc_cum_paths.shape[0]:,} plotted)")
    plt.xlabel("Trading Day"); plt.ylabel("Growth of $1")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("monte_carlo_fan_chart_30.png", dpi=120)
    plt.close()

    # Rolling 1Y Sharpe: Strategy vs SPY (matches original Figure 2)
    plt.figure(figsize=(12, 6))
    plt.plot(roll_sharpe_1y, label="30-Asset Index Rolling Sharpe (net-of-cost)", linewidth=2)
    plt.plot(roll_sharpe_1y_spy, label="SPY Rolling Sharpe", linewidth=2)
    plt.title("Rolling 1Y Sharpe Ratio")
    plt.xlabel("Date"); plt.ylabel("Sharpe Ratio")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("rolling_1y_sharpe_vs_spy_30.png", dpi=120)
    plt.close()

    # Drawdown Comparison: Strategy vs SPY (matches original Figure 3)
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown_net, label="30-Asset Index Drawdown (net-of-cost)", linewidth=2)
    plt.plot(drawdown_spy, label="SPY Drawdown", linewidth=2)
    plt.title("Drawdown Comparison")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("drawdown_comparison_30.png", dpi=120)
    plt.close()

    if len(roll_return_valid) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(roll_sharpe, label="Rolling 24M Sharpe (net-of-cost)", linewidth=2)
        plt.axhline(0, color="gray", linewidth=1, linestyle=":")
        plt.title("Rolling 24-Month Sharpe Ratio (strategy-only diagnostic)")
        plt.xlabel("Date"); plt.ylabel("Sharpe Ratio")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("rolling_24m_sharpe_30.png", dpi=120)
        plt.close()
        print("\nSaved plots: performance_gross_net_30.png, monte_carlo_fan_chart_30.png, "
              "rolling_1y_sharpe_vs_spy_30.png, drawdown_comparison_30.png, "
              "rolling_24m_sharpe_30.png")
    else:
        print("\nSaved plots: performance_gross_net_30.png, monte_carlo_fan_chart_30.png, "
              "rolling_1y_sharpe_vs_spy_30.png, drawdown_comparison_30.png")
 
    print("\nAverage allocation across all rebalances:")
    avg_w = monthly_weights.mean()
    avg_w = avg_w / avg_w.sum()
    for t, w in avg_w.sort_values(ascending=False).items():
        print(f"  {t}: {w:.1%}")
 
 
if __name__ == "__main__":
    main()
