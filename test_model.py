# ==========================================================
# FINAL TEST / BACKTEST CODE (FIXED)
# ==========================================================

import os
import numpy as np
import pandas as pd

from stable_baselines3 import A2C
from env.trading_env import PaperTradingEnv

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
MODEL_PATH = "models/a2c_trading_model.zip"

INITIAL_AMOUNT = 1_000_000
HMAX = 100
REWARD_ALPHA = 0.9
SHARPE_WINDOW = 20
RISK_FREE_RATE = 0.0

# ----------------------------------------------------------
# 2. LOAD TEST DATA
# ----------------------------------------------------------
test_df = pd.read_csv("data/test_data_paper21.csv")
test_df["date"] = pd.to_datetime(test_df["date"])

# ----------------------------------------------------------
# 3. FEATURE LIST
# ----------------------------------------------------------
state_features = [
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "adx", "turbulence",
    "current_ratio", "acid_test_ratio", "operating_cash_flow_ratio",
    "debt_ratio", "debt_to_equity", "interest_coverage_ratio",
    "asset_turnover", "inventory_turnover_ratio",
    "day_sales_in_inventory_ratio", "roa", "roe"
]

# ----------------------------------------------------------
# 4. VALIDATE DATA
# ----------------------------------------------------------
required_cols = ["date", "ticker", "close_raw"] + state_features
missing_cols = [col for col in required_cols if col not in test_df.columns]

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# ----------------------------------------------------------
# 5. ALIGN DATA
# ----------------------------------------------------------
expected_stocks = test_df["ticker"].nunique()

date_counts = test_df.groupby("date")["ticker"].nunique()
valid_dates = date_counts[date_counts == expected_stocks].index

test_df = test_df[test_df["date"].isin(valid_dates)].copy()
test_df = test_df.sort_values(["date", "ticker"]).reset_index(drop=True)

print("Expected stocks per date:", expected_stocks)
print("Remaining test dates:", test_df["date"].nunique())
print("Test shape:", test_df.shape)

# ----------------------------------------------------------
# 6. BUILD ENV
# ----------------------------------------------------------
test_env = PaperTradingEnv(
    df=test_df,
    stock_dim=test_df["ticker"].nunique(),
    state_feature_cols=state_features,
    initial_amount=INITIAL_AMOUNT,
    hmax=HMAX,
    reward_alpha=REWARD_ALPHA,
    risk_free_rate=RISK_FREE_RATE,
    sharpe_window=SHARPE_WINDOW,
    close_col="close_raw",
    date_col="date",
    ticker_col="ticker",
    print_verbosity=0,
)

# ----------------------------------------------------------
# 7. LOAD MODEL
# ----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = A2C.load(MODEL_PATH)
print("✅ Model loaded")

# ----------------------------------------------------------
# 8. RESET ENV (FIXED)
# ----------------------------------------------------------
obs = test_env.reset()

if isinstance(obs, tuple):
    obs = obs[0]

done = False

# ----------------------------------------------------------
# 9. BACKTEST LOOP (FIXED)
# ----------------------------------------------------------
while not done:
    action, _ = model.predict(obs, deterministic=True)

    action = np.array(action).flatten()

    result = test_env.step(action)

    if len(result) == 4:
        obs, reward, done, info = result
    else:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated

    if isinstance(obs, tuple):
        obs = obs[0]

print("✅ Backtesting complete")

# ----------------------------------------------------------
# 10. SAVE RESULTS
# ----------------------------------------------------------
portfolio_df = test_env.save_asset_memory()
reward_df = test_env.save_reward_memory()
action_df = test_env.save_action_memory()

portfolio_df.to_csv("a2c_test_portfolio_values.csv", index=False)
reward_df.to_csv("a2c_test_rewards.csv", index=False)
action_df.to_csv("a2c_test_actions.csv", index=False)

print("✅ Results saved")

# ----------------------------------------------------------
# 11. EVALUATION
# ----------------------------------------------------------
initial = portfolio_df["portfolio_value"].iloc[0]
final = portfolio_df["portfolio_value"].iloc[-1]

returns = portfolio_df["portfolio_value"].pct_change().dropna()

cumulative_return = (final - initial) / initial
sharpe_ratio = returns.mean() / (returns.std() + 1e-8)

print("\n📊 PERFORMANCE")
print("Initial Value:", initial)
print("Final Value:", final)
print("Cumulative Return:", cumulative_return)
print("Sharpe Ratio:", sharpe_ratio)