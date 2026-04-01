import numpy as np
import pandas as pd

# ==========================================================
# LOAD YOUR BACKTEST RESULT
# ==========================================================
df = pd.read_csv("a2c_test_portfolio_values.csv")

# Ensure sorted
df = df.sort_values("date")

# ==========================================================
# COMPUTE DAILY RETURNS
# ==========================================================
df["daily_return"] = df["portfolio_value"].pct_change()
df = df.dropna()

returns = df["daily_return"].values

# Risk-free rate (paper uses 0)
risk_free_rate = 0.0


# ==========================================================
# 1. SHARPE RATIO
# ==========================================================
def sharpe_ratio(returns, rf=0.0):
    excess_returns = returns - rf
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)


# ==========================================================
# 2. SORTINO RATIO
# ==========================================================
def sortino_ratio(returns, rf=0.0):
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    return (np.mean(returns) - rf) / (downside_std + 1e-8)


# ==========================================================
# 3. CALMAR RATIO
# ==========================================================
def max_drawdown(portfolio_values):
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    return np.min(drawdown)


def calmar_ratio(returns, portfolio_values):
    annual_return = np.mean(returns) * 252
    mdd = abs(max_drawdown(portfolio_values))
    return annual_return / (mdd + 1e-8)


# ==========================================================
# 4. OMEGA RATIO
# ==========================================================
def omega_ratio(returns, threshold=0.0):
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    return gains.sum() / (losses.sum() + 1e-8)


# ==========================================================
# 5. TAIL RATIO
# ==========================================================
def tail_ratio(returns):
    upper = np.percentile(returns, 95)
    lower = np.percentile(returns, 5)
    return upper / (abs(lower) + 1e-8)


# ==========================================================
# CALCULATE ALL METRICS
# ==========================================================
portfolio_values = df["portfolio_value"].values

results = {
    "Sharpe Ratio": sharpe_ratio(returns),
    "Sortino Ratio": sortino_ratio(returns),
    "Calmar Ratio": calmar_ratio(returns, portfolio_values),
    "Omega Ratio": omega_ratio(returns),
    "Tail Ratio": tail_ratio(returns),
}

# ==========================================================
# PRINT RESULTS
# ==========================================================
print("\n📊 Evaluation Metrics (A2C - Your Model)\n")

for k, v in results.items():
    print(f"{k}: {v:.6f}")