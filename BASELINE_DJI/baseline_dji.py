import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

TRADING_DAYS = 252

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
        return float("inf")
    return float(gains.sum() / loss_sum)

def tail_ratio(returns):
    upper = abs(np.percentile(returns, 95))
    lower = abs(np.percentile(returns, 5))
    if lower < 1e-12:
        return 0.0
    return float(upper / lower)

def stability(portfolio_values):
    log_values = np.log(portfolio_values / portfolio_values[0] + 1e-9)
    x = np.arange(len(log_values))
    _, _, r, _, _ = linregress(x, log_values)
    return float(r ** 2)


def compute_baseline_metrics(
    csv_path: str = "data/test_data_paper21.csv",
    initial_amount: float = 1_000_000.0,
    price_col: str = "close_raw",
    date_col: str = "date",
    ticker_col: str = "ticker",
):
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])

    required_cols = [date_col, ticker_col, price_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.sort_values([date_col, ticker_col]).reset_index(drop=True)

    price_df = df.pivot(
        index=date_col, columns=ticker_col, values=price_col
    ).sort_index()
    price_df = price_df.ffill().bfill()

    # Drop first row — pct_change gives NaN/0 there
    portfolio_returns = price_df.pct_change().mean(axis=1).iloc[1:]
    portfolio_value = (1.0 + portfolio_returns).cumprod() * initial_amount

    ret_arr = portfolio_returns.to_numpy(dtype=np.float64)
    pv_arr  = portfolio_value.to_numpy(dtype=np.float64)

    results = {
        "Cumulative Return": cumulative_return(pv_arr),
        "Annual Return":     annual_return(ret_arr),
        "Annual Volatility": annual_volatility(ret_arr),
        "Stability":         stability(pv_arr),
        "Max Drawdown":      max_drawdown(pv_arr),
        "Sharpe Ratio":      sharpe_ratio(ret_arr),
        "Calmar Ratio":      calmar_ratio(ret_arr, pv_arr),
        "Omega Ratio":       omega_ratio(ret_arr),
        "Tail Ratio":        tail_ratio(ret_arr),
        "Sortino Ratio":     sortino_ratio(ret_arr),
    }

    print("\n📊 BASELINE (^DJI Proxy / Equal-Weight Dow-30)")
    print(f"{'Initial Value':<22}: {initial_amount:,.2f}")
    print(f"{'Final Value':<22}: {pv_arr[-1]:,.2f}")
    for k, v in results.items():
        print(f"{k:<22}: {v:.6f}")

    baseline_df = pd.DataFrame({
        "date":            portfolio_value.index,
        "portfolio_value": pv_arr,
        "daily_return":    ret_arr,
    })
    baseline_df.to_csv("results/baseline_dji_portfolio_values.csv", index=False)
    print("\n✅ Saved: baseline_dji_portfolio_values.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(baseline_df["date"], baseline_df["portfolio_value"])
    plt.title("Baseline (^DJI Proxy) Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compute_baseline_metrics()