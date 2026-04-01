# 📈 RL Stock Trading — A Multifaceted Approach

> A Deep Reinforcement Learning framework for automated multi-stock trading using **A2C** and **DDPG** agents, validated on the **Dow Jones 30** constituent stocks.

---

## 🧠 Overview

This project implements a multifaceted Deep Reinforcement Learning (DRL) strategy for automated multi-stock market trading. The core idea is to enrich the RL agent's state representation by fusing three categories of market data — **daily historical prices**, **technical indicators**, and **fundamental indicators** — enabling the agent to make more informed, risk-aware trading decisions.

The agent is trained in a multi-stock environment where it simultaneously manages a portfolio of multiple stocks, issuing buy, hold, or sell signals along with the quantity of shares for each stock per trading day.

This work is based on the research paper:

> **"A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"**  
> Ansari et al., *IEEE Access*, Vol. 12, 2024. DOI: [10.1109/ACCESS.2024.3418510](https://doi.org/10.1109/ACCESS.2024.3418510)

---

## 🏗️ Architecture

The system is formulated as a **Partially Observable Markov Decision Process (POMDP)** with the following components:

### 🗂️ State Space
The agent's observation at each time step is a fusion of:

**Daily Historical Data (OHLCV)**
- Open, High, Low, Close prices and Volume for each stock

**Technical Indicators**
| Indicator | Description |
|---|---|
| MACD | Moving Average Convergence Divergence — momentum & trend strength |
| RSI | Relative Strength Index — overbought/oversold signal |
| CCI | Commodity Channel Index — momentum oscillator |
| DMI/ADX | Directional Movement Index — trend direction & magnitude |
| Turbulence | Market volatility & discord index |

**Fundamental Indicators** (sourced from Alpha Vantage API)
| Category | Ratios |
|---|---|
| Liquidity | Current Ratio, Acid Test Ratio, Operating Cash Flow Ratio |
| Leverage | Debt Ratio, Debt-to-Equity Ratio, Interest Coverage Ratio |
| Efficiency | Asset Turnover Ratio, Inventory Turnover Ratio, Day Sales in Inventory |
| Profitability | Return on Assets, Return on Equity |

Fundamental data (quarterly) is resampled to daily frequency using rolling techniques for smooth integration.

### ⚡ Action Space
At each time step, the agent decides for each stock:
- **Buy** (+1), **Hold** (0), or **Sell** (−1)
- Plus the **quantity** of shares to trade, bounded between 1 and 100 shares (k = 100)

Action space: `{-k, ..., -1, 0, 1, ..., k}` for each stock in the portfolio.

### 🏆 PSR Reward Function
A novel **Portfolio-Sharpe-Returns (PSR)** reward function is proposed to make the agent risk-aware:

```
Reward = ΔPortfolio + Sharpe Ratio + 0.9 × Daily Returns
```

Where the Sharpe Ratio is:
```
Sharpe Ratio = E[Ra - Rb] / σa
```

This reward simultaneously encourages the agent to:
- Grow the portfolio value
- Maximize risk-adjusted returns
- Capture short-term daily profitability

---

## 🤖 RL Algorithms

### Advantage Actor-Critic (A2C)
A2C combines policy-based and value-based methods. The critic computes an **advantage function** instead of just the value function, reducing gradient variance and improving robustness. Synchronized gradient updates make it well-suited for large batch stock data.

```
∇J(θ) = E[ Σ ∇θ log πθ(at|st) · A(at|st) ]
A(at|st) = r(st, at, st+1) + γV(st+1) − V(st)
```

### Deep Deterministic Policy Gradient (DDPG)
DDPG integrates deep Q-learning with deterministic policy gradients, handling **continuous action spaces** naturally. It uses a replay buffer, target networks, and learns directly from observations — making it ideal for continuous portfolio sizing decisions.

```
yi = ri + γQ'(si+1, μ'(si+1 | θμ') | θQ')
L = (1/N) Σ [yi − Q(si, ai | θQ)]²
```

Target networks are soft-updated:
```
θQ' ← τθQ + (1 − τ)θQ'
θμ' ← τθμ + (1 − τ)θμ'
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Universe** | Dow Jones 30 constituent stocks |
| **Daily Data Source** | Yahoo Finance API |
| **Fundamental Data Source** | Alpha Vantage API |
| **Training Period** | 2010-01-01 → 2021-10-01 |
| **Backtesting Period** | 2021-10-01 → 2023-03-01 |
| **Training Steps** | 50,000 timesteps |
| **Initial Portfolio Value** | $1,000,000 |
| **Max Shares per Trade** | 100 |

Fundamental data (balance sheets, income statements, cash flow statements) is collected quarterly and converted to daily frequency via rolling window techniques.

---

## 📐 Experimental Protocols

Three experimental setups were evaluated to measure the contribution of each data source:

| Protocol | State Inputs | Best Agent |
|---|---|---|
| **DTF** | Daily + Technical + Fundamental | DDPG ✅ (best overall) |
| **DT** | Daily + Technical | DDPG |
| **TF** | Technical + Fundamental | A2C |

The **DTF protocol with DDPG** achieved the best performance across all metrics, confirming that combining all three data sources provides the richest market representation.

---

## 📈 Results

Performance compared against the baseline **^DJI (Dow Jones Industrial Average)** index:

| Metric | A2C (DTF) | DDPG (DTF) | Baseline ^DJI |
|---|---|---|---|
| Sharpe Ratio | 0.079 | **0.17** | -0.105 |
| Cumulative Return | -0.478 | **1.648%** | -0.049 |
| Annual Return | -0.358% | **1.146%** | -0.035% |
| Max Drawdown | -20.478% | **-19.492%** | -22% |
| Omega Ratio | 1.013 | **1.028** | 0.983 |
| Sortino Ratio | 0.111 | **0.229** | -0.147 |

Key findings:
- **DDPG outperforms the ^DJI Index** across Sharpe ratio, Sortino ratio, and annual returns in the DTF setup
- **PSR reward significantly improves** agent decision-making over a simple portfolio-change reward
- The proposed method also **generalizes to the S&P 500** Energy sector (23 stocks), outperforming the baseline index

---

## 📁 Project Structure

```
rl_stock_trading/
│
├── data/
│   ├── fetch_yahoo.py          # Download OHLCV data from Yahoo Finance
│   ├── fetch_alphavantage.py   # Download fundamental data from Alpha Vantage
│   └── preprocess.py           # Feature engineering & data fusion
│
├── env/
│   └── stock_trading_env.py    # Custom multi-stock Gym environment (POMDP)
│
├── agents/
│   ├── a2c_agent.py            # A2C agent implementation
│   └── ddpg_agent.py           # DDPG agent implementation
│
├── reward/
│   └── psr_reward.py           # PSR (Portfolio-Sharpe-Returns) reward function
│
├── indicators/
│   ├── technical.py            # MACD, RSI, CCI, DMI, Turbulence
│   └── fundamental.py          # Liquidity, Leverage, Efficiency, Profitability ratios
│
├── train.py                    # Training script
├── backtest.py                 # Backtesting & evaluation
├── evaluate.py                 # Compute evaluation metrics
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

Key dependencies:
- `stable-baselines3` — A2C and DDPG implementations
- `gymnasium` — RL environment interface
- `yfinance` — Yahoo Finance data fetching
- `alpha_vantage` — Fundamental data API
- `pandas`, `numpy` — Data processing
- `ta` / `pandas-ta` — Technical indicator computation
- `matplotlib`, `pyfolio` — Visualization and performance analysis

### 1. Fetch Data
```bash
python data/fetch_yahoo.py --start 2010-01-01 --end 2023-03-01
python data/fetch_alphavantage.py --api_key YOUR_API_KEY
```

### 2. Train Agents
```bash
# Train DDPG with DTF protocol (Daily + Technical + Fundamental)
python train.py --agent ddpg --protocol DTF --timesteps 50000

# Train A2C
python train.py --agent a2c --protocol DTF --timesteps 50000
```

### 3. Backtest
```bash
python backtest.py --agent ddpg --model_path models/ddpg_dtf.zip
```

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| **Cumulative Return** | Total portfolio return at end of trading session |
| **Annual Return** | Annualized profit/loss of the trading strategy |
| **Sharpe Ratio** | Risk-adjusted return (higher = better) |
| **Max Drawdown** | Largest peak-to-trough portfolio loss |
| **Annual Volatility** | Standard deviation of portfolio returns |
| **Calmar Ratio** | Annual return divided by max drawdown |
| **Omega Ratio** | Risk-return performance ratio |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Tail Ratio** | Investment downside risk measure |

---

## 🔮 Future Work

- Incorporate **macroeconomic variables** (interest rates, inflation) into state representation
- Add **news and social media sentiment** (Twitter/X, financial news)
- Experiment with alternative reward functions (e.g., Calmar Ratio-based)
- Explore **ensemble RL agents** combining A2C and DDPG signals
- Apply **parallel/distributed training** to reduce execution time
- Integrate **human feedback** (RLHF) to accelerate policy learning

---

## 📚 Reference

```bibtex
@article{ansari2024multifaceted,
  title     = {A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning},
  author    = {Ansari, Yasmeen and Gillani, Saira and Bukhari, Maryam and Lee, Byeongcheon and Maqsood, Muazzam and Rho, Seungmin},
  journal   = {IEEE Access},
  volume    = {12},
  pages     = {90041--90060},
  year      = {2024},
  doi       = {10.1109/ACCESS.2024.3418510}
}
```

---

## ⚠️ Disclaimer

This project is developed for **research and educational purposes only**. It is not financial advice. Stock trading involves substantial risk of loss and is not appropriate for all investors. Past performance of any model does not guarantee future results.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
