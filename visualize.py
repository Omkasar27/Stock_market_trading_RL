"""
============================================================
PORTFOLIO PERFORMANCE ANALYSIS
A2C vs DDPG vs Baseline (^DJI)
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# STYLE CONFIG
# ----------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#0D1117",
    "figure.facecolor": "#0D1117",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.linewidth": 0.8,
    "text.color": "#C9D1D9",
    "axes.titlecolor": "#F0F6FC",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
})

COLORS = {
    "a2c":      "#58A6FF",   # bright blue
    "ddpg":     "#3FB950",   # bright green
    "baseline": "#F78166",   # coral red
    "bg":       "#0D1117",
    "card":     "#161B22",
    "border":   "#30363D",
    "text":     "#C9D1D9",
    "muted":    "#8B949E",
    "accent":   "#F0F6FC",
}

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------
def load_portfolio(path, value_col=None):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect date column
    date_candidates = [c for c in df.columns if "date" in c]
    if date_candidates:
        df["date"] = pd.to_datetime(df[date_candidates[0]])
    else:
        df["date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="B")

    # Detect portfolio value column
    if value_col and value_col in df.columns:
        df["portfolio_value"] = df[value_col]
    else:
        val_candidates = [c for c in df.columns if any(k in c for k in ["portfolio", "value", "total", "nav", "equity"])]
        if val_candidates:
            df["portfolio_value"] = df[val_candidates[0]]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df["portfolio_value"] = df[numeric_cols[0]]
            else:
                raise ValueError(f"Cannot find portfolio value column in {path}")

    return df[["date", "portfolio_value"]].sort_values("date").reset_index(drop=True)


print("📂 Loading portfolio data...")
a2c_df      = load_portfolio("results/a2c_test_portfolio_values.csv")
ddpg_df     = load_portfolio("results/ddpg_test_portfolio_values.csv")
baseline_df = load_portfolio("results/baseline_dji_portfolio_values.csv")
print(f"   A2C rows:      {len(a2c_df)}")
print(f"   DDPG rows:     {len(ddpg_df)}")
print(f"   Baseline rows: {len(baseline_df)}")


# ----------------------------------------------------------
# 2. METRICS
# ----------------------------------------------------------
def compute_metrics(df, name, initial=None):
    pv = df["portfolio_value"].values.astype(float)
    if initial is None:
        initial = pv[0]

    total_return = (pv[-1] - initial) / initial * 100
    daily_ret    = np.diff(pv) / pv[:-1]
    ann_return   = np.mean(daily_ret) * 252 * 100
    ann_vol      = np.std(daily_ret) * np.sqrt(252) * 100
    sharpe       = (np.mean(daily_ret) * 252) / (np.std(daily_ret) * np.sqrt(252) + 1e-10)

    roll_max     = np.maximum.accumulate(pv)
    drawdown     = (pv - roll_max) / (roll_max + 1e-10) * 100
    max_dd       = drawdown.min()

    calmar       = ann_return / (abs(max_dd) + 1e-10)

    pos_days     = np.sum(daily_ret > 0)
    win_rate     = pos_days / len(daily_ret) * 100 if len(daily_ret) > 0 else 0

    return {
        "Model":           name,
        "Initial Value":   f"${initial:,.0f}",
        "Final Value":     f"${pv[-1]:,.0f}",
        "Total Return (%)":f"{total_return:.2f}%",
        "Ann. Return (%)": f"{ann_return:.2f}%",
        "Ann. Volatility":  f"{ann_vol:.2f}%",
        "Sharpe Ratio":    f"{sharpe:.4f}",
        "Max Drawdown":    f"{max_dd:.2f}%",
        "Calmar Ratio":    f"{calmar:.4f}",
        "Win Rate (%)":    f"{win_rate:.2f}%",
        # raw for plotting
        "_total_return":   total_return,
        "_sharpe":         sharpe,
        "_max_dd":         max_dd,
        "_ann_vol":        ann_vol,
        "_ann_return":     ann_return,
        "_calmar":         calmar,
        "_win_rate":       win_rate,
    }

initial = a2c_df["portfolio_value"].iloc[0]
a2c_m   = compute_metrics(a2c_df,      "A2C",      initial)
ddpg_m  = compute_metrics(ddpg_df,     "DDPG",     initial)
base_m  = compute_metrics(baseline_df, "Baseline", initial)
metrics = [a2c_m, ddpg_m, base_m]

# Normalise all to same start
def normalise(df):
    df = df.copy()
    df["portfolio_value"] = df["portfolio_value"] / df["portfolio_value"].iloc[0] * 100
    return df

a2c_n   = normalise(a2c_df)
ddpg_n  = normalise(ddpg_df)
base_n  = normalise(baseline_df)

# Drawdown series
def drawdown_series(df):
    pv = df["portfolio_value"].values.astype(float)
    roll_max = np.maximum.accumulate(pv)
    return (pv - roll_max) / (roll_max + 1e-10) * 100

# ----------------------------------------------------------
# HELPER — axis styling
# ----------------------------------------------------------
def style_ax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(COLORS["card"])
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.grid(True, axis="x", linestyle="--", alpha=0.2)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
        spine.set_linewidth(0.8)
    if title:  ax.set_title(title, color=COLORS["accent"], fontsize=12, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=COLORS["muted"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=9)
    ax.tick_params(colors=COLORS["muted"], labelsize=8)

def add_legend(ax, **kwargs):
    leg = ax.legend(facecolor=COLORS["bg"], edgecolor=COLORS["border"],
                    labelcolor=COLORS["text"], fontsize=8, **kwargs)
    return leg

def fmt_pct_y(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

# ----------------------------------------------------------
# 3. PLOT 1 — A2C vs BASELINE
# ----------------------------------------------------------
def plot_a2c_vs_baseline():
    fig, axes = plt.subplots(3, 1, figsize=(13, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("A2C  vs  Baseline (^DJI)",
                 color=COLORS["accent"], fontsize=16, fontweight="bold", y=0.98)

    # — Portfolio Value
    ax = axes[0]
    ax.plot(a2c_n["date"], a2c_n["portfolio_value"],
            color=COLORS["a2c"], lw=2, label="A2C")
    ax.plot(base_n["date"], base_n["portfolio_value"],
            color=COLORS["baseline"], lw=2, label="Baseline (^DJI)", linestyle="--")
    ax.fill_between(a2c_n["date"], a2c_n["portfolio_value"], 100,
                    alpha=0.08, color=COLORS["a2c"])
    style_ax(ax, "Normalised Portfolio Value (Base = 100)",
             ylabel="Portfolio Value (Normalised)")
    ax.axhline(100, color=COLORS["muted"], lw=0.8, linestyle=":")
    fmt_pct_y(ax)
    add_legend(ax)

    # — Drawdown
    ax2 = axes[1]
    ax2.fill_between(a2c_df["date"], drawdown_series(a2c_df),
                     alpha=0.4, color=COLORS["a2c"], label="A2C DD")
    ax2.fill_between(baseline_df["date"], drawdown_series(baseline_df),
                     alpha=0.4, color=COLORS["baseline"], label="Baseline DD")
    ax2.plot(a2c_df["date"], drawdown_series(a2c_df), color=COLORS["a2c"], lw=1)
    ax2.plot(baseline_df["date"], drawdown_series(baseline_df),
             color=COLORS["baseline"], lw=1, linestyle="--")
    style_ax(ax2, "Drawdown (%)", ylabel="Drawdown (%)")
    add_legend(ax2, loc="lower left")

    # — Rolling Sharpe (30d)
    ax3 = axes[2]
    def rolling_sharpe(df, w=30):
        r = df["portfolio_value"].pct_change().dropna()
        rs = r.rolling(w).mean() / (r.rolling(w).std() + 1e-10) * np.sqrt(252)
        return df["date"].iloc[1:], rs

    d_a2c, rs_a2c = rolling_sharpe(a2c_df)
    d_bas, rs_bas = rolling_sharpe(baseline_df)
    ax3.plot(d_a2c, rs_a2c, color=COLORS["a2c"], lw=1.5, label="A2C")
    ax3.plot(d_bas, rs_bas, color=COLORS["baseline"], lw=1.5,
             linestyle="--", label="Baseline")
    ax3.axhline(0, color=COLORS["muted"], lw=0.7, linestyle=":")
    style_ax(ax3, "Rolling 30-Day Sharpe Ratio",
             xlabel="Date", ylabel="Sharpe")
    add_legend(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{OUTPUT_DIR}/1_a2c_vs_baseline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {path}")


# ----------------------------------------------------------
# 4. PLOT 2 — DDPG vs BASELINE
# ----------------------------------------------------------
def plot_ddpg_vs_baseline():
    fig, axes = plt.subplots(3, 1, figsize=(13, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("DDPG  vs  Baseline (^DJI)",
                 color=COLORS["accent"], fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(ddpg_n["date"], ddpg_n["portfolio_value"],
            color=COLORS["ddpg"], lw=2, label="DDPG")
    ax.plot(base_n["date"], base_n["portfolio_value"],
            color=COLORS["baseline"], lw=2, linestyle="--", label="Baseline (^DJI)")
    ax.fill_between(ddpg_n["date"], ddpg_n["portfolio_value"], 100,
                    alpha=0.08, color=COLORS["ddpg"])
    style_ax(ax, "Normalised Portfolio Value (Base = 100)", ylabel="Portfolio Value (Normalised)")
    ax.axhline(100, color=COLORS["muted"], lw=0.8, linestyle=":")
    fmt_pct_y(ax)
    add_legend(ax)

    ax2 = axes[1]
    ax2.fill_between(ddpg_df["date"], drawdown_series(ddpg_df),
                     alpha=0.4, color=COLORS["ddpg"], label="DDPG DD")
    ax2.fill_between(baseline_df["date"], drawdown_series(baseline_df),
                     alpha=0.4, color=COLORS["baseline"], label="Baseline DD")
    ax2.plot(ddpg_df["date"], drawdown_series(ddpg_df), color=COLORS["ddpg"], lw=1)
    ax2.plot(baseline_df["date"], drawdown_series(baseline_df),
             color=COLORS["baseline"], lw=1, linestyle="--")
    style_ax(ax2, "Drawdown (%)", ylabel="Drawdown (%)")
    add_legend(ax2, loc="lower left")

    ax3 = axes[2]
    def rolling_sharpe(df, w=30):
        r = df["portfolio_value"].pct_change().dropna()
        rs = r.rolling(w).mean() / (r.rolling(w).std() + 1e-10) * np.sqrt(252)
        return df["date"].iloc[1:], rs

    d_ddpg, rs_ddpg = rolling_sharpe(ddpg_df)
    d_bas,  rs_bas  = rolling_sharpe(baseline_df)
    ax3.plot(d_ddpg, rs_ddpg, color=COLORS["ddpg"], lw=1.5, label="DDPG")
    ax3.plot(d_bas,  rs_bas,  color=COLORS["baseline"], lw=1.5,
             linestyle="--", label="Baseline")
    ax3.axhline(0, color=COLORS["muted"], lw=0.7, linestyle=":")
    style_ax(ax3, "Rolling 30-Day Sharpe Ratio", xlabel="Date", ylabel="Sharpe")
    add_legend(ax3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{OUTPUT_DIR}/2_ddpg_vs_baseline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {path}")


# ----------------------------------------------------------
# 5. PLOT 3 — A2C vs DDPG
# ----------------------------------------------------------
def plot_a2c_vs_ddpg():
    fig, axes = plt.subplots(3, 1, figsize=(13, 12),
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("A2C  vs  DDPG",
                 color=COLORS["accent"], fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(a2c_n["date"], a2c_n["portfolio_value"],
            color=COLORS["a2c"], lw=2, label="A2C")
    ax.plot(ddpg_n["date"], ddpg_n["portfolio_value"],
            color=COLORS["ddpg"], lw=2, label="DDPG")
    ax.fill_between(a2c_n["date"], a2c_n["portfolio_value"],
                    ddpg_n["portfolio_value"],
                    where=(a2c_n["portfolio_value"].values >= ddpg_n["portfolio_value"].values),
                    alpha=0.10, color=COLORS["a2c"], label="A2C outperforms")
    ax.fill_between(a2c_n["date"], a2c_n["portfolio_value"],
                    ddpg_n["portfolio_value"],
                    where=(a2c_n["portfolio_value"].values < ddpg_n["portfolio_value"].values),
                    alpha=0.10, color=COLORS["ddpg"], label="DDPG outperforms")
    style_ax(ax, "Normalised Portfolio Value (Base = 100)", ylabel="Portfolio Value (Normalised)")
    ax.axhline(100, color=COLORS["muted"], lw=0.8, linestyle=":")
    fmt_pct_y(ax)
    add_legend(ax)

    ax2 = axes[1]
    ax2.fill_between(a2c_df["date"], drawdown_series(a2c_df),
                     alpha=0.4, color=COLORS["a2c"], label="A2C DD")
    ax2.fill_between(ddpg_df["date"], drawdown_series(ddpg_df),
                     alpha=0.4, color=COLORS["ddpg"], label="DDPG DD")
    ax2.plot(a2c_df["date"], drawdown_series(a2c_df), color=COLORS["a2c"], lw=1)
    ax2.plot(ddpg_df["date"], drawdown_series(ddpg_df), color=COLORS["ddpg"], lw=1)
    style_ax(ax2, "Drawdown (%)", ylabel="Drawdown (%)")
    add_legend(ax2, loc="lower left")

    # Daily return scatter
    ax3 = axes[2]
    ret_a2c  = a2c_df["portfolio_value"].pct_change().dropna().values * 100
    ret_ddpg = ddpg_df["portfolio_value"].pct_change().dropna().values * 100
    min_len  = min(len(ret_a2c), len(ret_ddpg))
    diff     = ret_a2c[:min_len] - ret_ddpg[:min_len]
    dates    = a2c_df["date"].iloc[1:min_len+1]
    ax3.bar(dates, diff, color=np.where(diff >= 0, COLORS["a2c"], COLORS["ddpg"]),
            alpha=0.7, width=1)
    ax3.axhline(0, color=COLORS["muted"], lw=0.8)
    style_ax(ax3, "Daily Return Difference  (A2C − DDPG, %)",
             xlabel="Date", ylabel="Return Diff (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = f"{OUTPUT_DIR}/3_a2c_vs_ddpg.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {path}")


# ----------------------------------------------------------
# 6. PLOT 4 — ALL THREE COMBINED (Dashboard)
# ----------------------------------------------------------
def plot_all_combined():
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Portfolio Performance Dashboard  |  A2C · DDPG · Baseline (^DJI)",
                 color=COLORS["accent"], fontsize=17, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- ROW 0: Full portfolio value (spans 2 cols)
    ax_main = fig.add_subplot(gs[0, :2])
    ax_main.plot(a2c_n["date"],   a2c_n["portfolio_value"],
                 color=COLORS["a2c"],      lw=2,   label="A2C")
    ax_main.plot(ddpg_n["date"],  ddpg_n["portfolio_value"],
                 color=COLORS["ddpg"],     lw=2,   label="DDPG")
    ax_main.plot(base_n["date"],  base_n["portfolio_value"],
                 color=COLORS["baseline"], lw=2,   label="Baseline (^DJI)", linestyle="--")
    ax_main.fill_between(a2c_n["date"],  a2c_n["portfolio_value"],  100, alpha=0.05, color=COLORS["a2c"])
    ax_main.fill_between(ddpg_n["date"], ddpg_n["portfolio_value"], 100, alpha=0.05, color=COLORS["ddpg"])
    ax_main.axhline(100, color=COLORS["muted"], lw=0.8, linestyle=":")
    style_ax(ax_main, "Normalised Portfolio Value (Base = 100)", ylabel="Normalised Value")
    fmt_pct_y(ax_main)
    add_legend(ax_main, loc="upper left")

    # --- ROW 0, COL 2: Metric Bar Chart — Total Return
    ax_bar = fig.add_subplot(gs[0, 2])
    models = ["A2C", "DDPG", "Baseline"]
    rets   = [a2c_m["_total_return"], ddpg_m["_total_return"], base_m["_total_return"]]
    cols   = [COLORS["a2c"], COLORS["ddpg"], COLORS["baseline"]]
    bars   = ax_bar.bar(models, rets, color=cols, alpha=0.85, width=0.5)
    for bar, val in zip(bars, rets):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color=COLORS["text"], fontsize=9, fontweight="bold")
    ax_bar.axhline(0, color=COLORS["muted"], lw=0.8)
    style_ax(ax_bar, "Total Return (%)", ylabel="%")

    # --- ROW 1: Drawdowns
    ax_dd = fig.add_subplot(gs[1, :2])
    ax_dd.fill_between(a2c_df["date"],      drawdown_series(a2c_df),
                       alpha=0.35, color=COLORS["a2c"],      label="A2C")
    ax_dd.fill_between(ddpg_df["date"],     drawdown_series(ddpg_df),
                       alpha=0.35, color=COLORS["ddpg"],     label="DDPG")
    ax_dd.fill_between(baseline_df["date"], drawdown_series(baseline_df),
                       alpha=0.35, color=COLORS["baseline"], label="Baseline")
    ax_dd.plot(a2c_df["date"],      drawdown_series(a2c_df),      color=COLORS["a2c"],      lw=1)
    ax_dd.plot(ddpg_df["date"],     drawdown_series(ddpg_df),     color=COLORS["ddpg"],     lw=1)
    ax_dd.plot(baseline_df["date"], drawdown_series(baseline_df), color=COLORS["baseline"], lw=1, linestyle="--")
    style_ax(ax_dd, "Underwater / Drawdown Chart (%)",
             xlabel="Date", ylabel="Drawdown (%)")
    add_legend(ax_dd, loc="lower left")

    # --- ROW 1, COL 2: Sharpe Bar
    ax_sh = fig.add_subplot(gs[1, 2])
    sharpes = [a2c_m["_sharpe"], ddpg_m["_sharpe"], base_m["_sharpe"]]
    bars2   = ax_sh.bar(models, sharpes, color=cols, alpha=0.85, width=0.5)
    for bar, val in zip(bars2, sharpes):
        ax_sh.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.01 if val >= 0 else -0.08),
                   f"{val:.3f}", ha="center", va="bottom",
                   color=COLORS["text"], fontsize=9, fontweight="bold")
    ax_sh.axhline(0, color=COLORS["muted"], lw=0.8)
    style_ax(ax_sh, "Sharpe Ratio", ylabel="Sharpe")

    # --- ROW 2, COL 0: Volatility Bar
    ax_vol = fig.add_subplot(gs[2, 0])
    vols = [a2c_m["_ann_vol"], ddpg_m["_ann_vol"], base_m["_ann_vol"]]
    bars3 = ax_vol.bar(models, vols, color=cols, alpha=0.85, width=0.5)
    for bar, val in zip(bars3, vols):
        ax_vol.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"{val:.1f}%", ha="center", va="bottom",
                    color=COLORS["text"], fontsize=9, fontweight="bold")
    style_ax(ax_vol, "Ann. Volatility (%)", ylabel="%")

    # --- ROW 2, COL 1: Max Drawdown Bar
    ax_mdd = fig.add_subplot(gs[2, 1])
    mdds = [a2c_m["_max_dd"], ddpg_m["_max_dd"], base_m["_max_dd"]]
    bars4 = ax_mdd.bar(models, mdds, color=cols, alpha=0.85, width=0.5)
    for bar, val in zip(bars4, mdds):
        ax_mdd.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                    f"{val:.1f}%", ha="center", va="top",
                    color=COLORS["text"], fontsize=9, fontweight="bold")
    style_ax(ax_mdd, "Max Drawdown (%)", ylabel="%")

    # --- ROW 2, COL 2: Win Rate Bar
    ax_wr = fig.add_subplot(gs[2, 2])
    wrs = [a2c_m["_win_rate"], ddpg_m["_win_rate"], base_m["_win_rate"]]
    bars5 = ax_wr.bar(models, wrs, color=cols, alpha=0.85, width=0.5)
    for bar, val in zip(bars5, wrs):
        ax_wr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f"{val:.1f}%", ha="center", va="bottom",
                   color=COLORS["text"], fontsize=9, fontweight="bold")
    style_ax(ax_wr, "Win Rate (%)", ylabel="%")

    path = f"{OUTPUT_DIR}/4_all_combined_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {path}")


# ----------------------------------------------------------
# 7. METRICS TABLE — print + save CSV
# ----------------------------------------------------------
def save_metrics_table():
    display_keys = [
        "Model", "Initial Value", "Final Value",
        "Total Return (%)", "Ann. Return (%)", "Ann. Volatility",
        "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Win Rate (%)"
    ]
    rows = []
    for m in metrics:
        rows.append({k: m[k] for k in display_keys})
    df_m = pd.DataFrame(rows)
    csv_path = "results/performance_metrics_summary.csv"
    df_m.to_csv(csv_path, index=False)
    print("\n" + "="*70)
    print("  PERFORMANCE METRICS SUMMARY")
    print("="*70)
    print(df_m.to_string(index=False))
    print("="*70)
    print(f"\n✅ Metrics saved: {csv_path}")
    return df_m


# ----------------------------------------------------------
# 8. METRICS HEATMAP (bonus — useful in reports)
# ----------------------------------------------------------
def plot_metrics_heatmap():
    raw_keys = ["_total_return", "_ann_return", "_ann_vol",
                "_sharpe", "_max_dd", "_calmar", "_win_rate"]
    labels   = ["Total Return", "Ann. Return", "Ann. Volatility",
                 "Sharpe", "Max Drawdown", "Calmar", "Win Rate"]
    model_names = ["A2C", "DDPG", "Baseline"]
    data = np.array([[a2c_m[k], ddpg_m[k], base_m[k]] for k in raw_keys])

    # Normalise row-wise to [0,1] for colour only
    normed = np.zeros_like(data)
    for i, key in enumerate(raw_keys):
        row = data[i]
        mn, mx = row.min(), row.max()
        if mx - mn < 1e-9:
            normed[i] = 0.5
        elif key in ["_ann_vol", "_max_dd"]:   # lower is better → invert
            normed[i] = 1 - (row - mn) / (mx - mn)
        else:
            normed[i] = (row - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    cmap = plt.cm.RdYlGn
    im = ax.imshow(normed, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, color=COLORS["accent"], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=COLORS["text"], fontsize=10)
    ax.tick_params(length=0)

    for i in range(len(labels)):
        for j in range(len(model_names)):
            val = data[i, j]
            txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if normed[i, j] > 0.4 else "white",
                    fontsize=10, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Performance Metrics Heatmap  (green = better)",
                 color=COLORS["accent"], fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/5_metrics_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"✅ Saved: {path}")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Generating all plots...\n")
    plot_a2c_vs_baseline()
    plot_ddpg_vs_baseline()
    plot_a2c_vs_ddpg()
    plot_all_combined()
    save_metrics_table()
    plot_metrics_heatmap()
    print(f"\n🎉 All outputs saved to  →  {OUTPUT_DIR}/")
    print("   1_a2c_vs_baseline.png")
    print("   2_ddpg_vs_baseline.png")
    print("   3_a2c_vs_ddpg.png")
    print("   4_all_combined_dashboard.png")
    print("   5_metrics_heatmap.png")
    print("   results/performance_metrics_summary.csv")