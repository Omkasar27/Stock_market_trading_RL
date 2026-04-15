# ==========================================================
# TRAIN DDPG MODEL ONLY
# ==========================================================

import os
import random
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env.trading_env import PaperTradingEnv

# ----------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

INITIAL_AMOUNT = 1_000_000
HMAX = 100
REWARD_ALPHA = 0.9
SHARPE_WINDOW = 20
RISK_FREE_RATE = 0.0
PORTFOLIO_REWARD_SCALE = 1.0 / 1000.0

TOTAL_TIMESTEPS = 50_000

# ----------------------------------------------------------
# 2. LOAD TRAIN DATA
# ----------------------------------------------------------
train_df = pd.read_csv("data/train_data_paper21.csv")
train_df["date"] = pd.to_datetime(train_df["date"])

state_features = [
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "cci", "adx", "turbulence",
    "current_ratio", "acid_test_ratio", "operating_cash_flow_ratio",
    "debt_ratio", "debt_to_equity", "interest_coverage_ratio",
    "asset_turnover", "inventory_turnover_ratio",
    "day_sales_in_inventory_ratio", "roa", "roe"
]

required_cols = ["date", "ticker", "close_raw"] + state_features
missing_cols = [c for c in required_cols if c not in train_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

train_df = train_df.sort_values(["date", "ticker"]).reset_index(drop=True)

print("Train dates:", train_df["date"].nunique())
print("Train tickers:", train_df["ticker"].nunique())
print("Train shape:", train_df.shape)

# ----------------------------------------------------------
# 3. BUILD ENV
# ----------------------------------------------------------
def make_env():
    return PaperTradingEnv(
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
        portfolio_reward_scale=PORTFOLIO_REWARD_SCALE,
    )

train_env = DummyVecEnv([make_env])

# ----------------------------------------------------------
# 4. ACTION NOISE
# ----------------------------------------------------------
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)
)

# ----------------------------------------------------------
# 5. TRAIN DDPG
# ----------------------------------------------------------
model = DDPG(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=1_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
    device="cpu",
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10)

# ----------------------------------------------------------
# 6. SAVE MODEL
# ----------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "ddpg_trading_model")
model.save(model_path)

print("✅ DDPG training complete")
print(f"✅ Model saved at: {model_path}")