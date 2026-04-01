# ==========================================================
# FINAL PREPROCESSING PIPELINE (21 FEATURES - PAPER ALIGNED)
# NO TRAIN-TEST LEAKAGE
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv("data/merged_dataset.csv")

# ----------------------------------------------------------
# 2. BASIC CLEANING
# ----------------------------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.drop_duplicates()
df = df.loc[:, ~df.columns.duplicated()]
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Keep raw close for environment/reward calculations
df["close_raw"] = df["close"]

# ----------------------------------------------------------
# 3. SAFE DIVISION
# ----------------------------------------------------------
def safe_divide(a, b):
    b = b.replace(0, np.nan)
    return a / b

# ----------------------------------------------------------
# 4. REQUIRED TECHNICAL FEATURES
# Paper uses MACD, RSI, CCI, DMI/ADX family, Turbulence
# ----------------------------------------------------------
if "adx" not in df.columns:
    if "dmi" in df.columns:
        df["adx"] = df["dmi"]
    elif "dx" in df.columns:
        df["adx"] = df["dx"]
    else:
        raise ValueError("Need one of: adx, dmi, or dx")

if "turbulence" not in df.columns:
    raise ValueError("Turbulence feature is required to match the paper")

# ----------------------------------------------------------
# 5. FUNDAMENTAL RATIOS (11)
# ----------------------------------------------------------
if "current_ratio" not in df.columns:
    df["current_ratio"] = safe_divide(df["totalCurrentAssets"], df["totalCurrentLiabilities"])

if "acid_test_ratio" not in df.columns:
    df["acid_test_ratio"] = safe_divide(
        df["cashAndCashEquivalentsAtCarryingValue"] + df["currentNetReceivables"],
        df["totalCurrentLiabilities"]
    )

if "operating_cash_flow_ratio" not in df.columns:
    if "operatingCashflow" in df.columns:
        df["operating_cash_flow_ratio"] = safe_divide(df["operatingCashflow"], df["totalCurrentLiabilities"])
    elif "operatingCashFlow" in df.columns:
        df["operating_cash_flow_ratio"] = safe_divide(df["operatingCashFlow"], df["totalCurrentLiabilities"])
    else:
        raise ValueError("Need operatingCashflow or operatingCashFlow")

if "debt_ratio" not in df.columns:
    df["debt_ratio"] = safe_divide(df["totalLiabilities"], df["totalAssets"])

if "debt_to_equity" not in df.columns:
    df["debt_to_equity"] = safe_divide(df["totalLiabilities"], df["totalShareholderEquity"])

if "interest_coverage_ratio" not in df.columns:
    df["interest_coverage_ratio"] = safe_divide(df["ebit"], df["interestExpense"].abs())

if "asset_turnover" not in df.columns:
    df["asset_turnover"] = safe_divide(df["totalRevenue"], df["totalAssets"])

if "inventory_turnover_ratio" not in df.columns:
    if "costOfRevenue" in df.columns:
        df["inventory_turnover_ratio"] = safe_divide(df["costOfRevenue"], df["inventory"])
    elif "costofGoodsAndServicesSold" in df.columns:
        df["inventory_turnover_ratio"] = safe_divide(df["costofGoodsAndServicesSold"], df["inventory"])
    else:
        raise ValueError("Need costOfRevenue or costofGoodsAndServicesSold")

if "day_sales_in_inventory_ratio" not in df.columns:
    df["day_sales_in_inventory_ratio"] = safe_divide(
        pd.Series(365, index=df.index), df["inventory_turnover_ratio"]
    )

if "roa" not in df.columns:
    df["roa"] = safe_divide(df["netIncome"], df["totalAssets"])

if "roe" not in df.columns:
    df["roe"] = safe_divide(df["netIncome"], df["totalShareholderEquity"])

# Optional for reward analysis
if "daily_return" not in df.columns:
    df["daily_return"] = df.groupby("ticker")["close_raw"].pct_change().fillna(0)

# ----------------------------------------------------------
# 6. FINAL PAPER FEATURE SET = 21
# 5 market + 5 technical + 11 fundamental
# ----------------------------------------------------------
market_features = ["open", "high", "low", "close", "volume"]
technical_features = ["macd", "rsi", "cci", "adx", "turbulence"]
fundamental_features = [
    "current_ratio",
    "acid_test_ratio",
    "operating_cash_flow_ratio",
    "debt_ratio",
    "debt_to_equity",
    "interest_coverage_ratio",
    "asset_turnover",
    "inventory_turnover_ratio",
    "day_sales_in_inventory_ratio",
    "roa",
    "roe",
]

state_features = market_features + technical_features + fundamental_features

missing_state = [c for c in state_features if c not in df.columns]
if missing_state:
    raise ValueError(f"Missing state features: {missing_state}")

# ----------------------------------------------------------
# 7. TIME-BASED SPLIT FIRST (IMPORTANT)
# Paper training/backtest windows
# ----------------------------------------------------------
split_date = pd.Timestamp("2021-10-01")

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()

# ----------------------------------------------------------
# 8. PREPROCESS EACH SPLIT SEPARATELY
# Fill missing values only within each split to avoid leakage
# ----------------------------------------------------------
def fill_by_ticker(dataframe, cols):
    dataframe = dataframe.copy()
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in cols:
        dataframe[col] = dataframe.groupby("ticker")[col].ffill()
        dataframe[col] = dataframe.groupby("ticker")[col].bfill()

    dataframe[cols] = dataframe[cols].fillna(0)
    return dataframe

train_df = fill_by_ticker(train_df, state_features)
test_df = fill_by_ticker(test_df, state_features)

# ----------------------------------------------------------
# 9. SCALE PER TICKER: FIT ON TRAIN, APPLY TO TEST
# ----------------------------------------------------------
scalers = {}
scaled_train_groups = []
scaled_test_groups = []

for ticker, train_group in train_df.groupby("ticker", sort=False):
    train_group = train_group.copy()

    scaler = StandardScaler()
    train_group[state_features] = scaler.fit_transform(train_group[state_features])
    train_group[state_features] = (
        train_group[state_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    scalers[ticker] = scaler
    scaled_train_groups.append(train_group)

    test_group = test_df[test_df["ticker"] == ticker].copy()
    if not test_group.empty:
        test_group[state_features] = scaler.transform(test_group[state_features])
        test_group[state_features] = (
            test_group[state_features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        scaled_test_groups.append(test_group)

train_df = pd.concat(scaled_train_groups, axis=0).sort_values(["ticker", "date"]).reset_index(drop=True)
test_df = pd.concat(scaled_test_groups, axis=0).sort_values(["ticker", "date"]).reset_index(drop=True)

# Combined processed dataframe
processed_df = pd.concat([train_df, test_df], axis=0).sort_values(["ticker", "date"]).reset_index(drop=True)

# ----------------------------------------------------------
# 10. FINAL VALIDATION
# ----------------------------------------------------------
assert train_df[state_features].isna().sum().sum() == 0, "NaNs in train features"
assert test_df[state_features].isna().sum().sum() == 0, "NaNs in test features"
assert np.isfinite(train_df[state_features].to_numpy()).all(), "Inf in train features"
assert np.isfinite(test_df[state_features].to_numpy()).all(), "Inf in test features"

# ----------------------------------------------------------
# 11. RL INPUT MATRICES
# Note: environment can also append balance and shares held
# as paper state variables during runtime.
# ----------------------------------------------------------
train_state_data = train_df[state_features].values
test_state_data = test_df[state_features].values

print("✅ Preprocessing Complete")
print("Total Features:", len(state_features))
print("Train State Shape:", train_state_data.shape)
print("Test State Shape:", test_state_data.shape)
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

# ----------------------------------------------------------
# 12. SAVE
# ----------------------------------------------------------
processed_df.to_csv("processed_data_paper21.csv", index=False)
train_df.to_csv("train_data_paper21.csv", index=False)
test_df.to_csv("test_data_paper21.csv", index=False)

print("Saved:")
print("- processed_data_paper21.csv")
print("- train_data_paper21.csv")
print("- test_data_paper21.csv")