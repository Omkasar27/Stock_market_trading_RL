# ==========================================================
# FINAL TEST / BACKTEST CODE FOR DDPG
# ==========================================================

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DDPG
from rl_env.trading_env import PaperTradingEnv

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
MODEL_PATH = "models/ddpg_trading_model.zip"

INITIAL_AMOUNT = 1_000_000
HMAX = 100
REWARD_ALPHA = 0.9
SHARPE_WINDOW = 20
RISK_FREE_RATE = 0.0
PORTFOLIO_REWARD_SCALE = 1.0 / 1000.0

# ----------------------------------------------------------
# 2. LOAD TEST DATA
# ----------------------------------------------------------
test_df = pd.read_csv("data/test_data_paper21.csv")
test_df["date"] = pd.to_datetime(test_df["date"])

state_features = [
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "adx", "turbulence",
    "current_ratio", "acid_test_ratio", "operating_cash_flow_ratio",
    "debt_ratio", "debt_to_equity", "interest_coverage_ratio",
    "asset_turnover", "inventory_turnover_ratio",
    "day_sales_in_inventory_ratio", "roa", "roe"
]

required_cols = ["date", "ticker", "close_raw"] + state_features
missing_cols = [c for c in required_cols if c not in test_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

test_df = test_df.sort_values(["date", "ticker"]).reset_index(drop=True)

print("Test dates:", test_df["date"].nunique())
print("Test tickers:", test_df["ticker"].nunique())
print("Test shape:", test_df.shape)

# ----------------------------------------------------------
# 3. BUILD ENV
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
# 4. LOAD MODEL
# ----------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = DDPG.load(MODEL_PATH)
print("✅ DDPG model loaded")

# ----------------------------------------------------------
# 5. BACKTEST LOOP
# ----------------------------------------------------------
obs, _ = test_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = np.array(action).flatten()

    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

print("✅ Backtesting complete")

# ----------------------------------------------------------
# 6. SAVE RESULTS
# ----------------------------------------------------------
portfolio_df = test_env.save_asset_memory()
reward_df = test_env.save_reward_memory()
action_df = test_env.save_action_memory()

portfolio_df.to_csv("results/ddpg_test_portfolio_values.csv", index=False)
reward_df.to_csv("results/ddpg_test_rewards.csv", index=False)
action_df.to_csv("results/ddpg_test_actions.csv", index=False)

print("✅ Results saved")

# ----------------------------------------------------------
# 7. EVALUATION
# ----------------------------------------------------------
initial = portfolio_df["portfolio_value"].iloc[0]
final = portfolio_df["portfolio_value"].iloc[-1]

cumulative_return = (final - initial) / initial
sharpe_ratio = test_env.get_sharpe_ratio()
max_drawdown = test_env.get_max_drawdown()




print("\n📊 PERFORMANCE (DDPG)")
print("Initial Value:", initial)
print("Final Value:", final)
print("Cumulative Return:", cumulative_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)