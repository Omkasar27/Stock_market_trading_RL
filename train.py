# ==========================================================
# TRAIN A2C MODEL ONLY (NO TRANSACTION COST)
# ==========================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.preprocessing import StandardScaler

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env.trading_env import PaperTradingEnv

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INITIAL_AMOUNT = 1_000_000
HMAX = 100
REWARD_ALPHA = 0.9
SHARPE_WINDOW = 30
RISK_FREE_RATE = 0.0

TOTAL_TIMESTEPS = 50_000

# ----------------------------------------------------------
# 2. LOAD TRAIN DATA
# ----------------------------------------------------------
train_df = pd.read_csv("data/train_data_paper21.csv")
train_df["date"] = pd.to_datetime(train_df["date"])
print(train_df.columns.tolist())
print(train_df[["date", "ticker", "close", "close_raw"]].head(10))
print(train_df.groupby("date")["ticker"].nunique().value_counts().sort_index())

# ----------------------------------------------------------
# 3. PAPER FEATURE LIST (DTF)
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
# 4. VALIDATE REQUIRED COLUMNS
# ----------------------------------------------------------
required_cols = ["date", "ticker", "close_raw"] + state_features
missing_cols = [col for col in required_cols if col not in train_df.columns]

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# ----------------------------------------------------------
# 5. ALIGN DATA
# ----------------------------------------------------------


print("Remaining train dates:", train_df["date"].nunique())
print("Train shape:", train_df.shape)

# ----------------------------------------------------------
# 6. SCALE FEATURES (IMPORTANT)
# ----------------------------------------------------------

# ----------------------------------------------------------
# 7. BUILD ENV
# ----------------------------------------------------------
train_env = DummyVecEnv([
    lambda: PaperTradingEnv(
        df=train_df,
        stock_dim=train_df["ticker"].nunique(),
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
])

# ----------------------------------------------------------
# 8. TRAIN A2C
# ----------------------------------------------------------
model = A2C(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    n_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=SEED,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

# ----------------------------------------------------------
# 9. SAVE MODEL
# ----------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "a2c_trading_model")
model.save(model_path)

print("✅ Training complete")
print(f"✅ Model saved at: {model_path}")
