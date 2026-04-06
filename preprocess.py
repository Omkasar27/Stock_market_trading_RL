# ==========================================================
# FINAL PREPROCESSING PIPELINE (21 FEATURES - PAPER ALIGNED)
# FIXED: COMPLETE DATE-TICKER PANEL + NO DOUBLE SCALING LATER
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

df = df.dropna(subset=["date", "ticker"]).copy()
df = df.drop_duplicates().copy()
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

# Keep raw close for environment/reward calculations
df["close_raw"] = df["close"]

# ----------------------------------------------------------
# 3. SAFE DIVISION
# ----------------------------------------------------------
def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return a / b

# ----------------------------------------------------------
# 4. REQUIRED TECHNICAL FEATURES
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
    if "ebit" not in df.columns or "interestExpense" not in df.columns:
        raise ValueError("Need ebit and interestExpense for interest_coverage_ratio")
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
        pd.Series(365.0, index=df.index), df["inventory_turnover_ratio"]
    )

# Resolve net income source robustly
if "netIncome" in df.columns:
    net_income_col = "netIncome"
elif "netIncome_income" in df.columns:
    net_income_col = "netIncome_income"
elif "netIncome_cashflow" in df.columns:
    net_income_col = "netIncome_cashflow"
else:
    raise ValueError("Need one of: netIncome, netIncome_income, netIncome_cashflow")

if "roa" not in df.columns:
    df["roa"] = safe_divide(df[net_income_col], df["totalAssets"])

if "roe" not in df.columns:
    df["roe"] = safe_divide(df[net_income_col], df["totalShareholderEquity"])

# Optional for analysis only
if "daily_return" not in df.columns:
    df["daily_return"] = df.groupby("ticker")["close_raw"].pct_change()

# ----------------------------------------------------------
# 6. FINAL PAPER FEATURE SET = 21
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
# 7. BUILD COMPLETE DATE × TICKER PANEL  <-- MAIN FIX
# ----------------------------------------------------------
all_dates = np.sort(df["date"].dropna().unique())
all_tickers = np.sort(df["ticker"].dropna().unique())

full_index = pd.MultiIndex.from_product(
    [all_dates, all_tickers],
    names=["date", "ticker"]
)

df = (
    df.set_index(["date", "ticker"])
      .reindex(full_index)
      .reset_index()
)

# ----------------------------------------------------------
# 8. RESTORE / FILL NON-FEATURE COLUMNS
# ----------------------------------------------------------
# Forward/backward fill every column within ticker.
# This creates complete rows for missing ticker-days.
cols_to_fill = [c for c in df.columns if c not in ["date", "ticker"]]

for col in cols_to_fill:
    df[col] = df.groupby("ticker")[col].ffill()
    df[col] = df.groupby("ticker")[col].bfill()

# Replace remaining NaN/inf
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df[cols_to_fill] = df[cols_to_fill].fillna(0)

# Recompute daily_return after panel completion
df["daily_return"] = df.groupby("ticker")["close_raw"].pct_change().fillna(0)

# ----------------------------------------------------------
# 9. TIME-BASED SPLIT FIRST
# ----------------------------------------------------------
split_date = pd.Timestamp("2021-10-01")

train_df = df[df["date"] < split_date].copy()
test_df  = df[df["date"] >= split_date].copy()

# ----------------------------------------------------------
# 10. FINAL CLEANING WITHIN EACH SPLIT
# ----------------------------------------------------------
def clean_split(dataframe, cols):
    dataframe = dataframe.copy()
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in cols + ["close_raw"]:
        dataframe[col] = dataframe.groupby("ticker")[col].ffill()
        dataframe[col] = dataframe.groupby("ticker")[col].bfill()

    dataframe[cols + ["close_raw"]] = dataframe[cols + ["close_raw"]].fillna(0)
    return dataframe

train_df = clean_split(train_df, state_features)
test_df  = clean_split(test_df, state_features)

# ----------------------------------------------------------
# 11. SCALE PER TICKER: FIT ON TRAIN, APPLY TO TEST
# ----------------------------------------------------------
scaled_train_groups = []
scaled_test_groups = []

for ticker in all_tickers:
    train_group = train_df[train_df["ticker"] == ticker].copy()
    test_group  = test_df[test_df["ticker"] == ticker].copy()

    if train_group.empty:
        continue

    scaler = StandardScaler()
    train_group[state_features] = scaler.fit_transform(train_group[state_features])

    if not test_group.empty:
        test_group[state_features] = scaler.transform(test_group[state_features])

    train_group[state_features] = (
        train_group[state_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    test_group[state_features] = (
        test_group[state_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    scaled_train_groups.append(train_group)
    if not test_group.empty:
        scaled_test_groups.append(test_group)

train_df = pd.concat(scaled_train_groups, axis=0).sort_values(["date", "ticker"]).reset_index(drop=True)
test_df  = pd.concat(scaled_test_groups, axis=0).sort_values(["date", "ticker"]).reset_index(drop=True)

processed_df = pd.concat([train_df, test_df], axis=0).sort_values(["date", "ticker"]).reset_index(drop=True)

# ----------------------------------------------------------
# 12. FINAL VALIDATION
# ----------------------------------------------------------
assert train_df[state_features].isna().sum().sum() == 0, "NaNs in train features"
assert test_df[state_features].isna().sum().sum() == 0, "NaNs in test features"
assert np.isfinite(train_df[state_features].to_numpy()).all(), "Inf in train features"
assert np.isfinite(test_df[state_features].to_numpy()).all(), "Inf in test features"

train_counts = train_df.groupby("date")["ticker"].nunique()
test_counts = test_df.groupby("date")["ticker"].nunique()

assert train_counts.min() == len(all_tickers), f"Train dates incomplete: min tickers/date={train_counts.min()}"
assert test_counts.min() == len(all_tickers), f"Test dates incomplete: min tickers/date={test_counts.min()}"

# ----------------------------------------------------------
# 13. SAVE
# ----------------------------------------------------------
processed_df.to_csv("data/processed_data_paper21.csv", index=False)
train_df.to_csv("data/train_data_paper21.csv", index=False)
test_df.to_csv("data/test_data_paper21.csv", index=False)

print("✅ Preprocessing Complete")
print("Total Features:", len(state_features))
print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
print("Test date range:", test_df["date"].min(), "to", test_df["date"].max())
print("Unique tickers:", len(all_tickers))
print("Min train tickers/date:", train_counts.min())
print("Min test tickers/date:", test_counts.min())
print("Saved:")
print("- data/processed_data_paper21.csv")
print("- data/train_data_paper21.csv")
print("- data/test_data_paper21.csv")
