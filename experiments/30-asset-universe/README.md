# 30-asset universe experiment (unpublished)

Same dual-momentum strategy and code as `multi_asset_v4.py`, run against a
much wider universe (sector SPDRs, international equity, fixed income,
commodities, real assets, small/mid cap — see `assets.py` for the full list)
instead of the published 5-ticker set (SPY/TLT/GLD/QQQ/EEM).

Kept as a "was this worth switching to" test, not published anywhere.
`backtest_v4.db` (the 5-asset version, at the project root) is the one behind
the site and resume.

- `assets.py` — pulls the wider price universe
- `multi_asset_30.py` — same backtest logic, run against it
- `backtest_30.db` — SQLite output (same 6-table schema as `backtest_v4.db`)
- `universe_prices.csv` — the raw pulled prices
- `*_30.png` — charts from this run
