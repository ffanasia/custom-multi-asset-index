"""
pulls extra ETF price history to widen the backtest universe past the
original 5 tickers. needs internet access for yfinance, so run locally --
output feeds straight into multi_asset_30.py as universe_prices.csv.
"""

import yfinance as yf
import pandas as pd

# sector SPDRs + a handful of other liquid ETFs, picked for broad
# asset-class coverage on top of the original SPY/TLT/GLD/QQQ/EEM
NEW_TICKERS = [
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

START = "2015-01-01"
END = "2024-12-31"

print(f"Fetching {len(NEW_TICKERS)} tickers from {START} to {END}...")
data = yf.download(NEW_TICKERS, start=START, end=END, progress=True)["Close"]

# Reshape to long format: date, ticker, close (matches your existing prices table schema)
long_df = data.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
long_df = long_df.rename(columns={"Date": "date"}).dropna(subset=["close"])
long_df = long_df.sort_values(["ticker", "date"])

long_df.to_csv("universe_prices.csv", index=False)
print(f"Saved {len(long_df)} rows to universe_prices.csv")
print(long_df.head())
print(long_df["ticker"].unique())
