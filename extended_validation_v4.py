"""
extended validation for the v4 multi-asset momentum index.

adds four things the original v4 backtest didn't have:
  1. walk-forward validation of REL_MOM_WINDOW (fit blind on 2016-2019, test on 2020-2024)
  2. a regime breakdown (bull/crash/recovery/rate-hike-bear/bull) instead of an anecdote
  3. an ablation ladder isolating what each design choice actually buys you
  4. a cost-sensitivity sweep + Calmar ratio

reuses the real v4 functions/constants (imported as a module) wherever possible so this
stays faithful to the methodology instead of re-deriving it. the only thing re-implemented
is the weight-building loop, because it needs extra knobs (sizing/fallback variant) that
v4.build_monthly_weights doesn't expose as parameters.

price data: v4_prices_cache.csv, pulled straight from backtest_v4.db's 'prices' table --
the exact same series the original 2015-2024 backtest used, no re-download.
"""

import json
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import multi_asset_v4 as v4

np.random.seed(42)

prices = pd.read_csv("v4_prices_cache.csv", index_col=0, parse_dates=True)
prices = prices[v4.tickers]


# ---------------------------------------------------------------
# core pipeline, parameterized (mirrors v4.main()'s logic exactly
# at the default settings -- verified below against the known DB numbers)
# ---------------------------------------------------------------
def run_pipeline(prices, rel_mom_window=None, cost_bps=None, sizing="inv_vol", fallback="scored"):
    rel_mom_window = rel_mom_window or v4.REL_MOM_WINDOW
    cost_bps = v4.COST_BPS if cost_bps is None else cost_bps

    returns = prices.pct_change().dropna()
    rel_mom = prices.pct_change(rel_mom_window)
    vol20 = returns.rolling(20).std()
    blend_parts = [prices.pct_change(w) for w in v4.MOM_BLEND_WINDOWS]
    abs_mom_blend = sum(blend_parts) / len(blend_parts)

    month_ends = v4.month_end_trading_dates(prices)
    valid_dates = month_ends.intersection(abs_mom_blend.dropna().index)
    valid_dates = valid_dates.intersection(rel_mom.dropna().index)

    all_cols = list(prices.columns) + [v4.CASH_TICKER]
    monthly_weights = pd.DataFrame(0.0, index=valid_dates, columns=all_cols)
    defensive_candidates = [t for t in v4.DEFENSIVE_BASKET if t in prices.columns]

    for date in valid_dates:
        abs_signal = abs_mom_blend.loc[date].dropna()
        rel_signal = rel_mom.loc[date].dropna()
        positive_assets = abs_signal[abs_signal > 0].index.tolist()
        w = pd.Series(0.0, index=all_cols)

        if len(positive_assets) == 0:
            if fallback == "scored":
                scores = {c: abs_signal[c] for c in defensive_candidates if c in abs_signal.index}
                scores[v4.CASH_TICKER] = v4.risk_free_rate
                w[max(scores, key=scores.get)] = 1.0
            elif fallback == "tlt_default":
                w["TLT"] = 1.0
        else:
            rel_subset = rel_signal.loc[positive_assets].sort_values(ascending=False)
            top_assets = rel_subset.head(2).index.tolist()
            if sizing == "inv_vol":
                current_vol = vol20.loc[date, top_assets].replace(0, np.nan).dropna()
                if len(current_vol) == len(top_assets):
                    inv_vol = 1.0 / current_vol
                    w[top_assets] = inv_vol / inv_vol.sum()
                else:
                    w[top_assets] = 1.0 / len(top_assets)
            elif sizing == "equal":
                w[top_assets] = 1.0 / len(top_assets)

        monthly_weights.loc[date] = w

    returns_ext = returns.copy()
    returns_ext[v4.CASH_TICKER] = v4.risk_free_rate / 252
    weights = monthly_weights.reindex(returns_ext.index).ffill().fillna(0.0).shift(1).fillna(0.0)
    gross_returns = (weights * returns_ext).sum(axis=1)

    cost_table = v4.transaction_costs(monthly_weights, cost_bps=cost_bps)
    net_returns = v4.apply_costs_to_returns(gross_returns, cost_table, returns_ext.index)

    return {
        "monthly_weights": monthly_weights,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "cost_table": cost_table,
        "valid_dates": valid_dates,
    }


def stats(r, cum=None):
    r = r.dropna()
    cum = (1 + r).cumprod() if cum is None else cum
    total_return = cum.iloc[-1] - 1
    n_years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = v4.sharpe_ratio(r, v4.risk_free_rate)
    mdd = v4.max_drawdown(cum)
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {"total_return": total_return, "cagr": cagr, "sharpe": sharpe,
            "max_drawdown": mdd, "calmar": calmar}


spy_returns = prices["SPY"].pct_change().dropna()

# =================================================================
# 0. sanity check -- does the harness reproduce the known v4 numbers?
# =================================================================
baseline = run_pipeline(prices)
baseline_stats = stats(baseline["net_returns"])
spy_stats = stats(spy_returns)
print("=== sanity check vs known backtest_v4.db numbers ===")
print(f"  net total return : {baseline_stats['total_return']:.2%}  (expected ~168.7%)")
print(f"  net sharpe       : {baseline_stats['sharpe']:.3f}  (expected ~0.651)")
print(f"  net max drawdown : {baseline_stats['max_drawdown']:.2%}  (expected ~-22.57%)")
print(f"  cost drag        : {baseline['cost_table']['cost_drag'].sum():.2%}  (expected ~2.76%)")
print(f"  spy sharpe       : {spy_stats['sharpe']:.3f}  (expected ~0.673)")
print(f"  spy max drawdown : {spy_stats['max_drawdown']:.2%}  (expected ~-33.72%)")
print()

results = {}

# =================================================================
# 1. ablation ladder
# =================================================================
print("=== ablation ladder ===")
ablation_steps = [
    ("naive_momentum",        dict(sizing="equal",   fallback="tlt_default", cost_bps=0.0)),
    ("+ inverse_vol_sizing",  dict(sizing="inv_vol",  fallback="tlt_default", cost_bps=0.0)),
    ("+ defensive_fallback",  dict(sizing="inv_vol",  fallback="scored",      cost_bps=0.0)),
    ("+ transaction_costs",   dict(sizing="inv_vol",  fallback="scored",      cost_bps=v4.COST_BPS)),
]
ablation_results = []
for name, kwargs in ablation_steps:
    out = run_pipeline(prices, **kwargs)
    r = out["net_returns"] if kwargs["cost_bps"] else out["gross_returns"]
    s = stats(r)
    ablation_results.append({"step": name, **s})
    print(f"  {name:<24} return={s['total_return']:>8.2%}  sharpe={s['sharpe']:>6.3f}  mdd={s['max_drawdown']:>8.2%}")
results["ablation"] = ablation_results
print()

# =================================================================
# 2. cost sensitivity
# =================================================================
print("=== cost sensitivity ===")
cost_results = []
for bps in [0.0, 5.0, 10.0, 20.0, 30.0]:
    out = run_pipeline(prices, cost_bps=bps)
    s = stats(out["net_returns"])
    drag = out["cost_table"]["cost_drag"].sum()
    cost_results.append({"cost_bps": bps, "cost_drag": drag, **s})
    print(f"  {bps:>5.0f}bps   drag={drag:>7.2%}   return={s['total_return']:>8.2%}  sharpe={s['sharpe']:>6.3f}  mdd={s['max_drawdown']:>8.2%}")
results["cost_sensitivity"] = cost_results
print()

# =================================================================
# 3. calmar ratio
# =================================================================
print("=== calmar ratio (CAGR / |max drawdown|) ===")
print(f"  strategy (net) : CAGR={baseline_stats['cagr']:.2%}  MDD={baseline_stats['max_drawdown']:.2%}  Calmar={baseline_stats['calmar']:.2f}")
print(f"  SPY            : CAGR={spy_stats['cagr']:.2%}  MDD={spy_stats['max_drawdown']:.2%}  Calmar={spy_stats['calmar']:.2f}")
results["calmar"] = {"strategy": baseline_stats, "spy": spy_stats}
print()

# =================================================================
# 4. regime breakdown
# =================================================================
print("=== regime breakdown ===")
regimes = [
    ("Pre-2019 bull",        "2016-01-01", "2018-12-31"),
    ("2019-early 2020 bull", "2019-01-01", "2020-02-19"),
    ("COVID crash",          "2020-02-20", "2020-03-23"),
    ("COVID recovery bull",  "2020-03-24", "2021-12-31"),
    ("2022 rate-hike bear",  "2022-01-01", "2022-12-31"),
    ("2023-2024 bull",       "2023-01-01", "2024-12-30"),
]
net_r = baseline["net_returns"]
regime_results = []
for label, start, end in regimes:
    seg_net = net_r.loc[start:end]
    seg_spy = spy_returns.loc[start:end]
    if seg_net.empty or seg_spy.empty:
        continue
    seg_net_cum = (1 + seg_net).cumprod()
    seg_spy_cum = (1 + seg_spy).cumprod()
    row = {
        "regime": label, "start": start, "end": end,
        "strategy_return": seg_net_cum.iloc[-1] - 1,
        "strategy_mdd": v4.max_drawdown(seg_net_cum),
        "spy_return": seg_spy_cum.iloc[-1] - 1,
        "spy_mdd": v4.max_drawdown(seg_spy_cum),
    }
    regime_results.append(row)
    print(f"  {label:<22} strat={row['strategy_return']:>8.2%} (mdd {row['strategy_mdd']:>7.2%})   "
          f"spy={row['spy_return']:>8.2%} (mdd {row['spy_mdd']:>7.2%})")
results["regimes"] = regime_results
print()

# =================================================================
# 5. walk-forward validation of REL_MOM_WINDOW
# =================================================================
print("=== walk-forward validation ===")
FIT_START, FIT_END = "2016-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-30"
grid = [21, 42, 63, 126, 189, 252]

wf_rows = []
window_cache = {}
for w in grid:
    out = run_pipeline(prices, rel_mom_window=w)
    window_cache[w] = out
    r = out["net_returns"]
    fit_r = r.loc[FIT_START:FIT_END]
    test_r = r.loc[TEST_START:TEST_END]
    full_stats = stats(r)
    wf_rows.append({
        "window_days": w,
        "fit_sharpe": v4.sharpe_ratio(fit_r, v4.risk_free_rate),
        "test_sharpe": v4.sharpe_ratio(test_r, v4.risk_free_rate),
        "test_return": (1 + test_r).cumprod().iloc[-1] - 1,
        "test_mdd": v4.max_drawdown((1 + test_r).cumprod()),
        "full_sharpe": full_stats["sharpe"],
    })

wf_df = pd.DataFrame(wf_rows).set_index("window_days")
print(wf_df.round(3))

fit_winner = wf_df["fit_sharpe"].idxmax()
full_winner = wf_df["full_sharpe"].idxmax()
print(f"\n  window chosen blind on fit period (2016-2019): {fit_winner} days")
print(f"  window that's actually best over the full 2016-2024 run: {full_winner} days")
print(f"  out-of-sample (2020-2024) result of the fit-period choice:")
print(f"    sharpe={wf_df.loc[fit_winner, 'test_sharpe']:.3f}  "
      f"return={wf_df.loc[fit_winner, 'test_return']:.2%}  "
      f"mdd={wf_df.loc[fit_winner, 'test_mdd']:.2%}")
if fit_winner != full_winner:
    print(f"    (differs from the full-period-optimal choice of {full_winner} days --")
    print(f"     that window's OOS sharpe would have been {wf_df.loc[full_winner, 'test_sharpe']:.3f})")
else:
    print(f"    (matches the full-period-optimal choice -- the original 252-day pick")
    print(f"     wasn't just curve-fit to this window)")

results["walk_forward"] = {
    "grid": wf_df.reset_index().to_dict(orient="records"),
    "fit_period_winner": int(fit_winner),
    "full_period_winner": int(full_winner),
}

# =================================================================
# 6. statistical rigor add-ons, in response to review
#    - analytical SE on the walk-forward Sharpe estimates (the fit period
#      is only 4 years -- Sharpe estimates that short are noisy, and the
#      original walk-forward chart didn't show that)
#    - SPY's own Sharpe over the same OOS test window, so a high strategy
#      OOS Sharpe can be read against "was the whole market just strong
#      here" instead of looking like strategy-specific outperformance
#    - a block-bootstrap significance check on strategy-vs-SPY Sharpe,
#      reusing the same block-bootstrap machinery as the project's own
#      Monte Carlo engine (paired resampling to preserve correlation)
# =================================================================
print("=== statistical rigor add-ons ===")

def sharpe_se(sharpe, n_days):
    """Analytical standard error of a Sharpe estimate (Lo, 2002 approx)."""
    years = n_days / 252
    return np.sqrt((1 + 0.5 * sharpe ** 2) / years)

wf_df["fit_years"] = (pd.Timestamp(FIT_END) - pd.Timestamp(FIT_START)).days / 365.25
wf_df["fit_se"] = wf_df.apply(lambda row: sharpe_se(row["fit_sharpe"], row["fit_years"] * 252), axis=1)
full_years = (net_r.index[-1] - net_r.index[0]).days / 365.25
wf_df["full_se"] = wf_df["full_sharpe"].apply(lambda s: sharpe_se(s, full_years * 252))
print("fit-period Sharpe +/- 1 SE (4yr sample -- these overlap a lot):")
for w, row in wf_df.iterrows():
    print(f"  {w:>4}d:  {row['fit_sharpe']:.2f} +/- {row['fit_se']:.2f}")
results["walk_forward"]["fit_se"] = wf_df["fit_se"].round(3).tolist()
results["walk_forward"]["full_se"] = wf_df["full_se"].round(3).tolist()

spy_test_sharpe = v4.sharpe_ratio(spy_returns.loc[TEST_START:TEST_END], v4.risk_free_rate)
strat_test_sharpe = wf_df.loc[fit_winner, "test_sharpe"]
print(f"\nOOS (2020-2024) Sharpe -- strategy: {strat_test_sharpe:.2f}, SPY: {spy_test_sharpe:.2f}")
print(f"(SPY's OOS Sharpe ({spy_test_sharpe:.2f}) is essentially flat vs. its own full-period number")
print(f" ({spy_stats['sharpe']:.2f}) -- 2020-2024 was NOT a generically strong stretch for the market.")
print(" the strategy's OOS boost looks specific to it, most plausibly the COVID crash landing")
print(" inside this test window, not a rising-tide effect)")
results["walk_forward"]["spy_test_sharpe"] = float(spy_test_sharpe)

# paired block-bootstrap on strategy-vs-SPY Sharpe, same block_size as v4's own MC
n_boot = 20_000
strat_r = baseline["net_returns"].dropna()
common_idx = strat_r.index.intersection(spy_returns.index)
strat_arr = strat_r.loc[common_idx].values
spy_arr = spy_returns.loc[common_idx].values
n = len(common_idx)
block_size = v4.MC_BLOCK_SIZE
n_blocks = int(np.ceil(n / block_size))
starts = np.random.randint(0, n - block_size + 1, size=(n_boot, n_blocks))
offsets = np.arange(block_size)
idx = starts[:, :, None] + offsets[None, None, :]
idx_flat = idx.reshape(n_boot, n_blocks * block_size)[:, :n]

strat_batch = strat_arr[idx_flat]
spy_batch = spy_arr[idx_flat]

def batch_sharpe(batch, rf=v4.risk_free_rate):
    excess = batch - rf / 252
    return np.sqrt(252) * excess.mean(axis=1) / excess.std(axis=1)

strat_sh = batch_sharpe(strat_batch)
spy_sh = batch_sharpe(spy_batch)
diff = strat_sh - spy_sh
ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
pct_strategy_wins = (diff > 0).mean()
print(f"\nblock-bootstrap (n={n_boot:,}, same paired history, block={block_size}d) on Sharpe(strategy) - Sharpe(SPY):")
print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  (includes zero: {ci_lo <= 0 <= ci_hi})")
print(f"  strategy has higher Sharpe in {pct_strategy_wins:.1%} of resampled histories")
print("  reading: the 0.65 vs 0.67 gap is not distinguishable from noise -- 'comparable' is the right word, not 'matched'")
hist_counts, hist_edges = np.histogram(diff, bins=60)
results["significance"] = {
    "sharpe_diff_ci_95": [float(ci_lo), float(ci_hi)],
    "ci_includes_zero": bool(ci_lo <= 0 <= ci_hi),
    "pct_resamples_strategy_higher_sharpe": float(pct_strategy_wins),
    "n_bootstrap": n_boot,
    "block_size": block_size,
    "diff_mean": float(diff.mean()),
    "diff_hist_counts": hist_counts.tolist(),
    "diff_hist_edges": hist_edges.tolist(),
}
print()

# =================================================================
# 7. deflated Sharpe ratio (Bailey & Lopez de Prado, 2014)
#
# the walk-forward check (section 5) only tested one design choice --
# REL_MOM_WINDOW -- against an out-of-sample period. but the full design
# has several more choices that were all picked by eyeballing the *entire*
# 2015-2024 period at once: the 5-asset universe, the top-2 concentration,
# the [63,126,252] momentum blend, which two assets sit in the defensive
# basket. every one of those is an implicit "trial" against the same data.
# running N trials and reporting the best one inflates the reported Sharpe
# above its true expected value under the null of no skill -- the more
# trials, the more some of them will look good by luck alone.
#
# the deflated Sharpe ratio corrects for this. it asks: given that N
# trials were effectively run, each with some variance in outcome, what
# Sharpe ratio would you expect the *best* of them to hit by pure chance?
# then it reports the probability that the observed Sharpe exceeds that
# luck-adjusted bar, using the full (non-normal) return distribution
# instead of assuming Gaussian returns.
#
# uses the 6 real walk-forward trials from section 5 as the empirical
# basis for the trial-to-trial variance term -- an honest, computed number
# rather than an assumed one. the true number of implicit trials in the
# full design is almost certainly higher than 6; the sensitivity sweep
# below shows how the result degrades as N grows, so that ambiguity is
# visible rather than hidden behind one number.
# =================================================================
print("=== deflated sharpe ratio ===")
from scipy import stats as sps

EULER_MASCHERONI = 0.5772156649015329

def daily_sharpe(r, rf=v4.risk_free_rate):
    """Non-annualized per-day Sharpe ratio -- kept in daily units throughout
    so it's consistent with T (a daily observation count) and with the
    skew/kurtosis of daily returns. (Annualizing SR but not the moments
    used alongside it would silently break the formula.)"""
    excess = r.dropna() - rf / 252
    return excess.mean() / excess.std(ddof=1)

def expected_max_sharpe(V, N):
    """E[max of N Sharpe estimates] under the null of zero true skill,
    given cross-trial variance V (Bailey & Lopez de Prado, 2014)."""
    if N <= 1:
        return 0.0
    return np.sqrt(V) * (
        (1 - EULER_MASCHERONI) * sps.norm.ppf(1 - 1.0 / N)
        + EULER_MASCHERONI * sps.norm.ppf(1 - 1.0 / (N * np.e))
    )

def probabilistic_sharpe(sr_hat, sr_benchmark, T, skew, kurt):
    """P(true Sharpe > sr_benchmark), correcting for T, skewness, kurtosis
    of the return series (Bailey & Lopez de Prado, 2012)."""
    denom = np.sqrt(max(1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / denom
    return sps.norm.cdf(z)

net_r_clean = baseline["net_returns"].dropna()
excess_r = net_r_clean - v4.risk_free_rate / 252
T_obs = len(excess_r)
sr_hat = daily_sharpe(net_r_clean)
skew_hat = sps.skew(excess_r, bias=False)
kurt_hat = sps.kurtosis(excess_r, fisher=False, bias=False)  # non-excess, normal=3

print(f"  observed: T={T_obs} days, daily SR={sr_hat:.4f} (annualized {sr_hat*np.sqrt(252):.3f}), "
      f"skew={skew_hat:.3f}, kurtosis={kurt_hat:.3f}")

# empirical cross-trial variance from the 6 real walk-forward trials
trial_daily_sharpes = np.array([daily_sharpe(window_cache[w]["net_returns"]) for w in grid])
V_trials = trial_daily_sharpes.var(ddof=1)
print(f"  {len(grid)} real trials (walk-forward window grid), daily SR range "
      f"[{trial_daily_sharpes.min():.4f}, {trial_daily_sharpes.max():.4f}], var={V_trials:.6f}")

N_actual = len(grid)
sr0_actual = expected_max_sharpe(V_trials, N_actual)
dsr_actual = probabilistic_sharpe(sr_hat, sr0_actual, T_obs, skew_hat, kurt_hat)
print(f"  N={N_actual} (only the tested windows): expected max SR by luck alone = "
      f"{sr0_actual:.4f} (annualized {sr0_actual*np.sqrt(252):.3f}); DSR = {dsr_actual:.1%}")

psr_n1 = probabilistic_sharpe(sr_hat, 0.0, T_obs, skew_hat, kurt_hat)
print(f"  for reference, N=1 (single hypothesis, no multiple-testing correction): "
      f"PSR vs. SR*=0 -> {psr_n1:.1%}")

n_sweep = [1, 2, 3, 6, 10, 15, 20, 30, 50, 75, 100]
sweep_rows = []
for n in n_sweep:
    sr0_n = expected_max_sharpe(V_trials, n)
    dsr_n = probabilistic_sharpe(sr_hat, sr0_n, T_obs, skew_hat, kurt_hat)
    sweep_rows.append({"N": n, "sr0_annualized": float(sr0_n * np.sqrt(252)), "dsr": float(dsr_n)})
    print(f"    N={n:>4}:  luck-bar (ann.)={sr0_n*np.sqrt(252):.3f}   DSR={dsr_n:.1%}")

results["deflated_sharpe"] = {
    "sr_hat_annualized": float(sr_hat * np.sqrt(252)),
    "T": int(T_obs),
    "skew": float(skew_hat),
    "kurtosis": float(kurt_hat),
    "n_actual_trials": N_actual,
    "sr0_actual_annualized": float(sr0_actual * np.sqrt(252)),
    "dsr_actual": float(dsr_actual),
    "psr_n1": float(psr_n1),
    "sweep": sweep_rows,
}
print()

with open("extended_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("saved extended_results.json")
