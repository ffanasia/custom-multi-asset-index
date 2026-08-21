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

## walk-forward validation

The 252-day (12M) ranking window was chosen by sweeping the full 2015-2024 period, which
means the original result doesn't rule out overfitting. Re-ran the sweep two ways: pick the
window using only 2016-2019 data (blind to what happens after), then check how that choice
performs on 2020-2024 it never saw.

Blind fit-period pick: 189 trading days. Full-period-optimal pick: also 189 days. They agree
-- the window wasn't just curve-fit to the reporting period. Out-of-sample (2020-2024) result
for the fit-blind choice: Sharpe 0.83, return +93.7%, max drawdown -17.7%.

One wrinkle worth being upfront about: this sweep finds 189 days edges out 252 days on Sharpe
over the full period (0.673 vs 0.651) -- a modest, not dramatic, difference. The original
252-day pick remains defensible (it's what's actually running, and the gap is small enough
that it could be noise), but 189 days is arguably a fractionally better choice and worth
testing further before switching.

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

See `ablation_ladder_v4.png`.

## regime breakdown

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
