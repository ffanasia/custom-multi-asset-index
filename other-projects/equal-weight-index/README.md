# equal-weight index (not the published project)

This is a separate, ongoing side project — not the dual-momentum index that's
on the site/resume. Different universe (SPY/TLT/GLD/VNQ/DBC, no QQQ or EEM),
different logic (equal-weight across whichever 3-of-5 assets are held each
month, not momentum-ranked or inverse-vol sized), and it's a live-updating
tracker from 2012 to whenever it was last run, not a fixed backtest window.

It has no cost model, no liquidity check, no Monte Carlo, no walk-forward
validation, and no deflated Sharpe correction — none of the stress-testing
that `multi_asset_v4.py` goes through. Its numbers (Sharpe ~0.98 vs. SPY's
~0.93) are NOT comparable to the published project's numbers and shouldn't be
cited alongside them without running the same validation first.

- `custom_multi_asset_index_notebook.ipynb` — the notebook that produces this
- `index_levels.csv` — daily index level vs. SPY, 2012–present
- `rebalance_history.csv` — monthly holdings (equal weight, 3 of 5 assets)
- `performance_summary.csv` — total return / CAGR / vol / Sharpe / max DD,
  reconciles exactly against `index_levels.csv`
