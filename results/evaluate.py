import numpy as np
import pandas as pd
from scipy.stats import linregress

FILE_PATH = "a2c_test_portfolio_values.csv"
MODEL_NAME = "DDPG Trading Model"
RISK_FREE_RATE = 0.0
TRADING_DAYS = 252

df = pd.read_csv(FILE_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["daily_return"] = df["portfolio_value"].pct_change()
df = df.dropna().reset_index(drop=True)

returns = df["daily_return"].to_numpy(dtype=np.float64)
portfolio_values = df["portfolio_value"].to_numpy(dtype=np.float64)

def cumulative_return(portfolio_values):
    return float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0])

def annual_return(returns):
    compounded = np.prod(1 + returns)
    n = len(returns)
    return float(compounded ** (TRADING_DAYS / n) - 1)

def annual_volatility(returns):
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))

def sharpe_ratio(returns, rf=0.0):
    excess = returns - rf
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float((np.mean(excess) / std) * np.sqrt(TRADING_DAYS))

def sortino_ratio(returns, rf=0.0):
    downside = returns[returns < rf]
    if len(downside) < 2:
        return 0.0
    downside_std = np.std(downside - rf, ddof=1)
    if downside_std < 1e-12:
        return 0.0
    return float(((np.mean(returns) - rf) / downside_std) * np.sqrt(TRADING_DAYS))

def max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return float(np.min(drawdown))

def calmar_ratio(returns, portfolio_values):
    ann_ret = annual_return(returns)
    mdd = abs(max_drawdown(portfolio_values))
    if mdd < 1e-12:
        return 0.0
    return float(ann_ret / mdd)

def omega_ratio(returns, threshold=0.0):
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    loss_sum = losses.sum()
    if loss_sum < 1e-12:
        return float("inf")  # fixed
    return float(gains.sum() / loss_sum)

def tail_ratio(returns):
    upper = abs(np.percentile(returns, 95))  # fixed
    lower = abs(np.percentile(returns, 5))
    if lower < 1e-12:
        return 0.0
    return float(upper / lower)

def stability(portfolio_values):
    log_values = np.log(portfolio_values / portfolio_values[0] + 1e-9)
    x = np.arange(len(log_values))
    _, _, r, _, _ = linregress(x, log_values)
    return float(r ** 2)

results = {
    "Cumulative Return":  cumulative_return(portfolio_values),
    "Annual Return":      annual_return(returns),
    "Annual Volatility":  annual_volatility(returns),
    "Stability":          stability(portfolio_values),
    "Max Drawdown":       max_drawdown(portfolio_values),
    "Sharpe Ratio":       sharpe_ratio(returns, rf=RISK_FREE_RATE),
    "Calmar Ratio":       calmar_ratio(returns, portfolio_values),
    "Omega Ratio":        omega_ratio(returns),
    "Tail Ratio":         tail_ratio(returns),
    "Sortino Ratio":      sortino_ratio(returns, rf=RISK_FREE_RATE),
}

print(f"\n📊 Evaluation Metrics ({MODEL_NAME})\n")
for k, v in results.items():
    print(f"{k:<22}: {v:.6f}")