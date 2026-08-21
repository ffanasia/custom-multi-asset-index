# extended validation, v4

Follow-up to the original v4 backtest. The original result (Sharpe 0.65 vs SPY's 0.67,
max drawdown -22% vs -34%, 2015-2024) was a single historical run with a lookback window
I'd picked by eyeballing the full period. This digs into whether that holds up under more
scrutiny: is the window overfit, what's actually driving the result, is the cost assumption
fragile, and does the "defensive fallback saved 2022" story hold up numerically instead of
just anecdotally.

Reuses the real `multi_asset_v4.py` functions/constants throughout (imported as a module) --
this isn't a re-derivation, it's the same pipeline with a few extra parameters exposed.
Script: `extended_validation_v4.py`. Sanity-checked against `backtest_v4.db` before trusting
any of it -- the baseline run reproduces the known numbers (0.651 Sharpe, -22.57% MDD, 2.76%
cost drag) to 3 decimal places.

Second pass, after a self-review of the first version of this doc: added analytical standard
errors to the walk-forward Sharpe estimates, ran a paired block-bootstrap significance test on
the strategy-vs-SPY Sharpe gap, disclosed that the regime windows were chosen with hindsight,
flagged the Calmar ratio's single-episode fragility, and relabeled the ablation chart's
gross/net axes, which were ambiguous in the first draft. None of the underlying numbers
changed -- this pass adds uncertainty bounds and scope caveats around claims that were stated
too plainly the first time.

Third pass: the walk-forward check only tests one design choice against out-of-sample data,
which leaves open whether the *rest* of the design (universe, concentration, blend windows,
defensive basket) was implicitly overfit by eyeballing the full period. Added a deflated
Sharpe ratio (Bailey & Lopez de Prado, 2014) to answer that directly -- it corrects the
reported Sharpe for how many design choices were effectively tried, using the real
(non-normal) shape of the daily return distribution rather than assuming Gaussian returns.
This is a step most backtests, including the original version of this one, skip entirely.

## walk-forward validation

The 252-day (12M) ranking window was chosen by sweeping the full 2015-2024 period, which
means the original result doesn't rule out overfitting. Re-ran the sweep two ways: pick the
window using only 2016-2019 data (blind to what happens after), then check how that choice
performs on 2020-2024 it never saw.

Blind fit-period pick: 189 trading days. Full-period-optimal pick: also 189 days. They agree
-- for this one hyperparameter, the window wasn't just curve-fit to the reporting period.
Out-of-sample (2020-2024) result for the fit-blind choice: Sharpe 0.83, return +93.7%, max
drawdown -17.7%.

Two things worth being upfront about, added after a second look:

**Scope.** This only tests one design choice -- the momentum ranking window -- picked blind
on an earlier period and checked out-of-sample. It says nothing about the rest of the design
(the two-asset concentration, the defensive-fallback logic, the asset universe itself), all of
which were chosen by looking at the whole 2015-2024 run. "Not overfit" below refers only to
the ranking window, not to the strategy as a whole.

**The fit-period Sharpe estimates are noisy.** Four years of monthly-rebalanced returns is a
small sample, and Sharpe ratios estimated on samples that short carry wide standard errors. Added
analytical SEs (Lo, 2002 approximation) to each window's fit-period estimate:

| window | fit Sharpe | ± SE |
|---|---|---|
| 21d | 0.39 | 0.52 |
| 42d | 0.34 | 0.51 |
| 63d | 0.53 | 0.53 |
| 126d | 0.32 | 0.51 |
| 189d | 0.58 | 0.54 |
| 252d | 0.52 | 0.53 |

Every one of these error bars overlaps every other one. The 189-day "winner" on the fit period
is not statistically distinguishable from 252d, 63d, or even 21d given only four years of data
-- the blind-selection process picked a window that happens to also do well out-of-sample, but
the fit-period comparison itself doesn't have the power to declare a winner. The "they agree"
finding above is real and worth keeping, but it shouldn't be read as "189d is provably the
best window" -- it's "189d survived a blind test," which is a weaker and more honest claim.

**One more wrinkle:** this sweep finds 189 days edges out 252 days on Sharpe over the full
period (0.673 vs 0.651) -- see the significance section below for why that gap isn't
meaningful either. The original 252-day pick remains what's actually running and is fully
defensible.

**On the out-of-sample Sharpe (0.83).** It's tempting to read that as proof the fit-blind
window works. Checked whether 2020-2024 was just a strong market generally by computing SPY's
own Sharpe over the same window: 0.66, essentially flat versus SPY's own full-period number
(0.67). So this wasn't a rising tide lifting all boats -- the strategy's OOS Sharpe really is
higher than its own full-period number. The more likely explanation is structural, not
statistical: the COVID crash (Feb-Mar 2020) landed inside this test window, and the regime
breakdown below shows that's exactly where this design earns its keep. That's a real result,
but it's a result about one crash landing in the test period, not a general validation that
the strategy performs better going forward.

See `walk_forward_v4.png`.

## ablation study

Isolated what each design layer actually contributes, gross returns except the last step:

| step | return | Sharpe | max drawdown |
|---|---|---|---|
| naive momentum (equal-weight, default-to-TLT) | +142.6% | 0.54 | -37.3% |
| + inverse-vol sizing | +137.1% | 0.53 | -37.0% |
| + defensive fallback (scored TLT/GLD/CASH) | +176.1% | 0.67 | -22.4% |
| + transaction costs (net, 5bps) | +168.7% | 0.65 | -22.6% |

The honest finding here: inverse-vol sizing alone does almost nothing -- Sharpe and drawdown
both come out slightly *worse* than naive equal-weighting. The entire improvement, both the
Sharpe gain and the ~15-point drawdown reduction, comes from replacing the "default to TLT"
fallback with the scored TLT/GLD/CASH logic. That matches the story in the original script's
notes (TLT alone failed badly in 2022) but it's now measured instead of anecdotal, and it
means the sizing scheme was never the load-bearing piece of this design -- the fallback logic
was.

Note on the table: only the last row nets out transaction costs -- the first three isolate the
gross effect of each design layer so the comparison isn't muddied by costs changing alongside
the strategy. The chart labels this explicitly (gross/gross/gross/net) after an earlier version
left it ambiguous.

See `fig5_ablation.png`.

## regime breakdown

One caveat before the numbers: these six windows were chosen with hindsight, by looking at
where the strategy and SPY visibly diverged and drawing boundaries around it (the COVID crash
dates, the 2022 bear market, and so on are all well-known, easily-identified periods). That's
a reasonable way to organize a narrative, but it's not a blind or pre-registered partition of
the data, and hand-picked regime boundaries can flatter whichever story you're already telling.
Treat this section as a structured description of what happened in known periods, not as
independent statistical evidence.

Sliced strategy (net) vs. SPY returns and drawdown across six labeled periods:

| regime | strategy | SPY |
|---|---|---|
| Pre-2019 bull | +23% | +30% |
| 2019-early 2020 bull | +17% | +38% |
| COVID crash (Feb-Mar 2020) | -3% | -34% |
| COVID recovery bull | +52% | +119% |
| 2022 rate-hike bear | -20% | -18% |
| 2023-2024 bull | +57% | +58% |

The COVID crash is where the design earns its keep -- essentially flat while SPY dropped a
third. It also gives back a lot of upside in strong bull markets (COVID recovery, 2019), which
is the expected cost of running ~4 points lower annualized vol, not a flaw.

The one regime that complicates the "defensive fallback fixed 2022" narrative: the strategy
actually lost slightly *more* than SPY in full-year 2022 (-20% vs -18%), even though its
max drawdown within the year was shallower (-22% vs -25%). The fallback logic reduced how bad
the trough got, but it didn't make 2022 a win, and the original code comments overstate this
a little. Worth being precise about that distinction going forward: shallower drawdown, not
better full-year return.

See `regime_breakdown_v4.png`.

## significance: is 0.65 vs 0.67 even a real gap?

Ran a paired block-bootstrap on the strategy-vs-SPY Sharpe difference -- same block-resampling
technique as the project's own 1M-path Monte Carlo engine (20-day blocks), applied here to test
a hypothesis instead of estimate a risk distribution. 20,000 resamples, same paired dates for
both series each draw so the correlation structure is preserved.

**Result: 95% CI on Sharpe(strategy) − Sharpe(SPY) = [−0.65, 0.56]. Includes zero. The strategy
has the higher Sharpe in only 45.1% of resampled histories.**

Plainly: the 0.65 vs 0.67 Sharpe gap is not distinguishable from noise. "Comparable Sharpe" is
the right way to describe it; "matched Sharpe" or "equal Sharpe" overstates what a single
9-year backtest can actually establish. This doesn't undercut the drawdown result -- the
-22% vs -34% max-drawdown gap is a single realized fact, not a ratio with a standard error in
the same way -- but the Sharpe-parity framing needed this check before it could be stated as
fact rather than observation.

See `fig9_significance.png`.

## deflated Sharpe ratio: is the edge itself real, or overfit to this design?

The walk-forward check above only tested one design choice -- REL_MOM_WINDOW -- against an
out-of-sample period. But the full design has several more choices that were all picked by
looking at the *entire* 2015-2024 period at once: the 5-asset universe, the top-2
concentration, the [63,126,252] momentum blend, which two assets sit in the defensive basket.
Every one of those is an implicit "trial" against the same data, and running N trials and
keeping the best one inflates the reported Sharpe above its true expected value under the null
of no skill -- purely by chance, some of N trials will look good.

This is a different question from the significance test above. That test asked "is the
strategy's Sharpe meaningfully *higher than SPY's*?" (answer: no, not distinguishable). This
one asks "is the strategy's Sharpe meaningfully *higher than zero* -- once corrected for how
many things were implicitly tried while designing it?"

Used the deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) to answer this properly: it
computes the Sharpe ratio you'd expect the *best* of N trials to hit by pure luck (using the
actual variance across trials, not an assumption), then reports the probability the observed
Sharpe clears that bar -- using the real skewness and kurtosis of daily returns instead of
assuming a normal distribution (daily returns here have skew -0.48 and kurtosis 6.7, both far
from normal, which the standard Sharpe ratio silently ignores).

The honest complication: the *true* number of implicit trials in the full design is almost
certainly more than the 6 windows actually swept in the walk-forward check -- there's no way
to give an exact count for "how many times did I eyeball a chart before landing on 5 tickers
and a 2-asset defensive basket." Rather than pick one flattering number, computed it across a
range and let the reader see how it holds up:

| assumed N | luck bar (annualized Sharpe) | DSR (P(edge is real)) |
|---|---|---|
| 1 (no correction) | 0.00 | 97.9% |
| 6 (windows actually tested) | 0.10 | 95.7% |
| 20 | 0.15 | 94.1% |
| 100 | 0.20 | 92.1% |

Even under a generous assumption of 100 implicit trials, the "luck bar" only rises to a Sharpe
of ~0.20 -- nowhere near the observed 0.65 -- because the variance in outcomes across the
windows actually tested was small (they ranged Sharpe 0.47 to 0.67, not wildly). The deflated
Sharpe ratio stays above 92% across the entire sweep. Reading: this isn't proof the specific
252-day window is the best possible choice (the walk-forward section already showed several
windows are statistically indistinguishable from each other), but it is good evidence that the
overall approach -- multi-asset momentum with a defensive fallback -- has a real edge over
holding cash, rather than being the lucky survivor of an unreported search.

See `fig10_deflated_sharpe.png`.

## cost sensitivity + Calmar ratio

Reran net returns at 0/5/10/20/30bps per rebalance. Sharpe degrades roughly linearly from
0.67 (0bps) to 0.55 (30bps) -- a 6x increase over the assumed 5bps still leaves a Sharpe
comparable to SPY's, and max drawdown barely moves (-22.4% to -24.1%). The edge isn't fragile
to the cost assumption.

See `cost_sensitivity_v4.png`.

Calmar ratio (CAGR / |max drawdown|), which tells the risk-reduction story better than Sharpe
does:

- strategy (net): CAGR 10.4%, MDD -22.6%, **Calmar 0.46**
- SPY: CAGR 13.1%, MDD -33.7%, **Calmar 0.39**

The strategy actually wins on Calmar despite a near-tied Sharpe -- more return per unit of
max drawdown taken, which is arguably the more relevant number given the entire point of the
defensive-fallback design.

Worth flagging: Calmar is a fragile statistic here -- both numbers are driven by a single
worst-drawdown episode per series (the strategy's -22.6% happened once, SPY's -33.7% happened
once), not an average or distribution. A different draw of market history could put either
number somewhere else entirely. Unlike the Sharpe comparison above, there wasn't a clean way
to bootstrap a confidence interval on this one (the max-drawdown statistic doesn't decompose
into independent blocks the way a mean return does), so treat the 0.46 vs 0.39 gap as
directional and real for this specific 2015-2024 history, not as a precisely estimated
population parameter.
