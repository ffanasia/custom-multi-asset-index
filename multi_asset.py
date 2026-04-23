import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# =========================================
# 1. Parameters
# =========================================
tickers = ["SPY", "TLT", "GLD", "QQQ", "EEM"]
benchmark = "SPY"
start_date = "2015-01-01"
end_date = "2024-12-31"
risk_free_rate = 0.02

# =========================================
# 2. Download data
# =========================================
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=True
)

# Robust column handling
if isinstance(data.columns, pd.MultiIndex):
    if "Close" in data.columns.get_level_values(0):
        prices = data["Close"].copy()
    else:
        raise ValueError(f"Expected 'Close' in downloaded data. Got: {data.columns}")
else:
    prices = data.copy()

prices = prices.dropna(how="all").ffill().dropna()

# Keep only requested tickers that actually downloaded
available = [c for c in tickers if c in prices.columns]
prices = prices[available]

if benchmark not in prices.columns:
    raise ValueError(f"{benchmark} not found in price data.")

if prices.empty:
    raise ValueError("Price data is empty after cleaning.")

print("Using assets:", list(prices.columns))

# =========================================
# 3. Returns and signals
# =========================================
returns = prices.pct_change().dropna()

# Dual momentum signals
abs_mom = prices.pct_change(252)   # 12M absolute momentum
rel_mom = prices.pct_change(126)   # 6M relative momentum
vol20 = returns.rolling(20).std()

# =========================================
# 4. Monthly rebalancing dates
# Use business month-end dates that exist in prices
# =========================================
month_ends = prices.resample("BME").last().index

# Keep only dates where all signals exist
valid_dates = month_ends.intersection(abs_mom.dropna().index)
valid_dates = valid_dates.intersection(rel_mom.dropna().index)

if len(valid_dates) == 0:
    raise ValueError("No valid rebalance dates found.")

# =========================================
# 5. Build monthly target weights
# =========================================
monthly_weights = pd.DataFrame(0.0, index=valid_dates, columns=prices.columns)

for date in valid_dates:
    abs_signal = abs_mom.loc[date].dropna()
    rel_signal = rel_mom.loc[date].dropna()

    # assets with positive absolute momentum
    positive_assets = abs_signal[abs_signal > 0].index.tolist()

    w = pd.Series(0.0, index=prices.columns)

    if len(positive_assets) == 0:
        # defensive fallback
        if "TLT" in prices.columns:
            w["TLT"] = 1.0
        else:
            w[benchmark] = 1.0
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

# =========================================
# 6. Apply weights daily with 1-day lag
# Important: rebalance at month-end, hold next trading day onward
# =========================================
weights = monthly_weights.reindex(returns.index).ffill().fillna(0.0)
weights = weights.shift(1).fillna(0.0)

# =========================================
# 7. Portfolio returns
# =========================================
portfolio_returns = (weights * returns).sum(axis=1)
spy_returns = returns[benchmark]

portfolio_cum = (1 + portfolio_returns).cumprod()
spy_cum = (1 + spy_returns).cumprod()

# =========================================
# 8. Metrics
# =========================================
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

index_return = portfolio_cum.iloc[-1] - 1
spy_return = spy_cum.iloc[-1] - 1

index_sharpe = sharpe_ratio(portfolio_returns, risk_free_rate)
spy_sharpe = sharpe_ratio(spy_returns, risk_free_rate)

percent_outperformance = ((index_return - spy_return) / abs(spy_return)) * 100
sharpe_improvement = ((index_sharpe - spy_sharpe) / abs(spy_sharpe)) * 100

index_mdd = max_drawdown(portfolio_cum)
spy_mdd = max_drawdown(spy_cum)

# =========================================
# 9. Results
# =========================================
print("\n===== Performance Comparison =====")
print(f"Custom Index Return: {index_return:.2%}")
print(f"SPY Return: {spy_return:.2%}")
print(f"Outperformance: {percent_outperformance:.2f}%")

print("\nSharpe Ratios:")
print(f"Custom Index Sharpe: {index_sharpe:.2f}")
print(f"SPY Sharpe: {spy_sharpe:.2f}")
print(f"Sharpe Improvement: {sharpe_improvement:.2f}%")

print("\nMax Drawdown:")
print(f"Custom Index Max Drawdown: {index_mdd:.2%}")
print(f"SPY Max Drawdown: {spy_mdd:.2%}")

print("\nSample monthly weights:")
print(monthly_weights.tail(10).round(3))

# =========================================
# 10. Plots
# =========================================
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum, label="Custom Dual Momentum Index", linewidth=2)
plt.plot(spy_cum, label="SPY", linewidth=2)
plt.title("Custom Multi-Asset Index vs SPY")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()

rolling_sharpe_index = portfolio_returns.rolling(252).apply(
    lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
)
rolling_sharpe_spy = spy_returns.rolling(252).apply(
    lambda x: sharpe_ratio(pd.Series(x), risk_free_rate), raw=False
)

plt.figure(figsize=(12, 6))
plt.plot(rolling_sharpe_index, label="Custom Index Rolling Sharpe", linewidth=2)
plt.plot(rolling_sharpe_spy, label="SPY Rolling Sharpe", linewidth=2)
plt.title("Rolling 1Y Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
drawdown_index = portfolio_cum / portfolio_cum.cummax() - 1
drawdown_spy = spy_cum / spy_cum.cummax() - 1
plt.plot(drawdown_index, label="Custom Index Drawdown", linewidth=2)
plt.plot(drawdown_spy, label="SPY Drawdown", linewidth=2)
plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)
plt.show()