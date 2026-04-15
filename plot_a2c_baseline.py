import pandas as pd
import matplotlib.pyplot as plt

# Load baseline
baseline_df = pd.read_csv("baseline_dji_portfolio_values.csv")
baseline_df["date"] = pd.to_datetime(baseline_df["date"])

# Load A2C results
a2c_df = pd.read_csv("a2c_test_portfolio_values.csv")
a2c_df["date"] = pd.to_datetime(a2c_df["date"])

# Merge on date to ensure alignment
plot_df = pd.merge(
    baseline_df[["date", "portfolio_value"]],
    a2c_df[["date", "portfolio_value"]],
    on="date",
    how="inner",
    suffixes=("_baseline", "_a2c")
)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(plot_df["date"], plot_df["portfolio_value_baseline"], label="Baseline (^DJI Proxy)")
plt.plot(plot_df["date"], plot_df["portfolio_value_a2c"], label="A2C")

plt.title("A2C vs Baseline Portfolio Value")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()