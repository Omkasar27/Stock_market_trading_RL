# ==========================================================
# FINAL TEST / BACKTEST CODE (FIXED)
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd

from stable_baselines3 import A2C
from rl_env.trading_env import PaperTradingEnv

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
MODEL_PATH = "models/a2c_trading_model.zip"
SCALER_PATH = "models/feature_scaler.pkl"

INITIAL_AMOUNT = 1_000_000
HMAX = 100
REWARD_ALPHA = 0.9
SHARPE_WINDOW = 30   # match training
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


print("Remaining test dates:", test_df["date"].nunique())
print("Test shape:", test_df.shape)

# ----------------------------------------------------------
# 6. APPLY TRAIN SCALER TO TEST FEATURES
# ----------------------------------------------------------
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")



# ----------------------------------------------------------
# 7. BUILD ENV
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
    portfolio_reward_scale=1.0,
)

# ----------------------------------------------------------
# 8. LOAD MODEL
# ----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = A2C.load(MODEL_PATH)
print("✅ Model loaded")

# ----------------------------------------------------------
# 9. RESET ENV
# ----------------------------------------------------------
obs, _ = test_env.reset()
done = False

# ----------------------------------------------------------
# 10. BACKTEST LOOP
# ----------------------------------------------------------
while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = np.array(action).flatten()

    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

print("✅ Backtesting complete")

# ----------------------------------------------------------
# 11. SAVE RESULTS
# ----------------------------------------------------------
portfolio_df = test_env.save_asset_memory()
reward_df = test_env.save_reward_memory()
action_df = test_env.save_action_memory()

portfolio_df.to_csv("results/a2c_test_portfolio_values.csv", index=False)
reward_df.to_csv("results/a2c_test_rewards.csv", index=False)
action_df.to_csv("results/a2c_test_actions.csv", index=False)

print("✅ Results saved")

# ----------------------------------------------------------
# 12. EVALUATION
# ----------------------------------------------------------
initial = portfolio_df["portfolio_value"].iloc[0]
final = portfolio_df["portfolio_value"].iloc[-1]

cumulative_return = (final - initial) / initial
sharpe_ratio = test_env.get_sharpe_ratio()
max_drawdown = test_env.get_max_drawdown()

print("\n📊 PERFORMANCE")
print("Initial Value:", initial)
print("Final Value:", final)
print("Cumulative Return:", cumulative_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)