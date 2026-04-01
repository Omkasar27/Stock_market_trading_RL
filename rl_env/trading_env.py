import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding


class PaperTradingEnv(gym.Env):
    """
    Multi-stock trading environment — aligned to:
    "A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning"
    Ansari et al., IEEE Access 2024.

    State  (Section III-A-1):
        [cash] + [shares_held × N] + [flattened per-stock features × N]

        Per-stock features (21 total — paper Table 1/2/3 + Section III-A):
            5  daily market   : open, high, low, close, volume
            5  technical      : MACD, RSI, CCI, ADX/DX, Turbulence
            11 fundamental    : current_ratio, acid_test, ocf_ratio,
                                debt_ratio, debt_to_equity, interest_coverage,
                                asset_turnover, inventory_turnover,
                                day_sales_inventory, roa, roe

    Action  (Section III-A-2):
        Continuous [-1, 1] per stock.
        Mapped to integer shares in {-hmax, ..., 0, ..., hmax}.
        -1 → sell hmax shares, 0 → hold, +1 → buy hmax shares.
        Paper sets hmax = 100.

    Reward  (Section III-A-3)  — PSR (Portfolio-Sharpe-Returns):
        reward = change_in_portfolio + sharpe_ratio + alpha * daily_return
        where alpha = 0.9  (tuned via simulations in the paper).

        change_in_portfolio  = end_portfolio_value - begin_portfolio_value
                               (raw $ change, matching paper eq. 1)
        sharpe_ratio         = E[Ra - Rb] / σa  (paper eq. 2, Rb = 0)
                               computed over a rolling window of past returns.
        daily_return         = change_in_portfolio / begin_portfolio_value

    FIX 1 — Reward scaling:
        Original code had change_in_portfolio in raw dollars (~thousands)
        while sharpe and daily_return are tiny decimals → sharpe/daily_return
        terms were completely swamped.
        Fix: normalise change_in_portfolio by initial_amount so all three
        terms are in the same ~[-1, 1] range, consistent with the paper's
        intent that all three terms contribute meaningfully.

    FIX 2 — Sharpe formula:
        Paper eq. 2: Sharpe = E[Ra - Rb] / σa
        Original used ddof=0 std and added 1e-8 inside denominator.
        Fixed to: mean(excess_returns) / (std(excess_returns, ddof=1) + 1e-9)
        ddof=1 is the standard unbiased estimator for a sample window.

    FIX 3 — State normalisation:
        Cash and shares in the state vector were in raw scale (millions / hundreds)
        while feature values are small decimals → caused numerical issues.
        Fix: normalise cash by initial_amount and shares by hmax in the
        observation, so the full state vector is in a consistent scale.
        The internal bookkeeping (self.cash, self.shares_held) stays in
        raw units — only the returned observation is scaled.

    FIX 4 — Action scaling (paper Section III-A-2):
        Paper action space is {-k, ..., k} where k = hmax = 100.
        Continuous action a ∈ [-1, 1] is mapped to integer shares via
        round(a * hmax), not truncation (astype(int)).
        round() is more faithful: action=0.005 → 0 shares (hold),
        not a spurious 0 from truncation.

    FIX 5 — Terminal state guard:
        Original returned stale state on terminal day. Fixed to always
        advance data before returning the final observation.

    FIX 6 — POMDP note:
        Paper uses POMDP (Section III-A), meaning the agent observes
        features rather than the true state directly. This env is already
        observation-based (agent sees OHLCV + indicators, not hidden state),
        which satisfies the POMDP intent. No structural change needed.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        state_feature_cols: list,
        initial_amount: float = 1_000_000,   # paper uses $1,000,000
        hmax: int = 100,                      # paper Section III-A-2: k=100
        reward_alpha: float = 0.9,            # paper eq.1: tuned via simulations
        risk_free_rate: float = 0.0,          # paper eq.2: Rb = 0
        sharpe_window: int = 20,              # rolling window for Sharpe estimate
        close_col: str = "close_raw",
        date_col: str = "date",
        ticker_col: str = "ticker",
        print_verbosity: int = 0,
    ):
        super().__init__()

        self.df = df.copy()
        self.stock_dim = stock_dim
        self.state_feature_cols = state_feature_cols
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.reward_alpha = reward_alpha
        self.risk_free_rate = risk_free_rate
        self.sharpe_window = sharpe_window
        self.close_col = close_col
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.print_verbosity = print_verbosity

        # ── validate columns ──────────────────────────────────────────────
        required_cols = [self.date_col, self.ticker_col, self.close_col] + self.state_feature_cols
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # ── sort and index by date ─────────────────────────────────────────
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df = self.df.sort_values([self.date_col, self.ticker_col]).reset_index(drop=True)

        self.unique_dates = self.df[self.date_col].drop_duplicates().sort_values().tolist()
        self.date_to_data = {}

        for d in self.unique_dates:
            day_df = (
                self.df[self.df[self.date_col] == d]
                .sort_values(self.ticker_col)
                .reset_index(drop=True)
            )
            if len(day_df) != self.stock_dim:
                raise ValueError(
                    f"Date {d} has {len(day_df)} stocks, expected {self.stock_dim}. "
                    f"Ensure all {self.stock_dim} DOW-30 tickers are present every trading day."
                )
            self.date_to_data[d] = day_df

        # ── spaces ────────────────────────────────────────────────────────
        self.feature_dim = len(self.state_feature_cols)

        # State dim: 1 (cash) + N (shares) + N*F (features)
        # Paper Section III-A-1: F = 21 per stock (5 daily + 5 tech + 11 fund)
        self.state_dim = 1 + self.stock_dim + (self.stock_dim * self.feature_dim)

        # Action: continuous [-1, 1] per stock (paper Section III-A-2)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.stock_dim,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        # ── internal state (initialised in reset) ─────────────────────────
        self.day = None
        self.data = None
        self.terminal = None
        self.cash = None
        self.shares_held = None
        self.portfolio_value = None
        self.asset_memory = None
        self.reward_memory = None
        self.daily_return_memory = None
        self.actions_memory = None
        self.date_memory = None

        self.seed()
        self.reset()

    # ──────────────────────────────────────────────────────────────────────
    # Seeding
    # ──────────────────────────────────────────────────────────────────────
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _get_day_data(self):
        return self.date_to_data[self.unique_dates[self.day]]

    def _get_prices(self):
        return self.data[self.close_col].values.astype(np.float32)

    def _get_features(self):
        return self.data[self.state_feature_cols].values.astype(np.float32).flatten()

    def _get_state(self):
        """
        FIX 3 — Normalised observation vector.
        cash      → divided by initial_amount  (puts it in ~[0, 1] range)
        shares    → divided by hmax             (puts it in ~[0, N] range)
        features  → passed as-is (should be pre-normalised in your pipeline)

        Internal bookkeeping (self.cash, self.shares_held) stays in raw units.
        """
        norm_cash   = np.array([self.cash / self.initial_amount], dtype=np.float32)
        norm_shares = (self.shares_held / self.hmax).astype(np.float32)
        return np.concatenate([norm_cash, norm_shares, self._get_features()])

    def _calculate_portfolio_value(self, prices):
        return float(self.cash + np.sum(self.shares_held * prices))

    def _calculate_sharpe(self):
        """
        FIX 2 — Paper eq. 2: Sharpe = E[Ra - Rb] / σa
        Uses ddof=1 (unbiased sample std) and a tiny epsilon for stability.
        risk_free_rate (Rb) = 0.0 per paper.
        """
        if len(self.daily_return_memory) < 2:
            return 0.0

        returns = np.array(
            self.daily_return_memory[-self.sharpe_window:], dtype=np.float64
        )
        excess = returns - self.risk_free_rate

        # Need at least 2 points for ddof=1
        if len(excess) < 2:
            return 0.0

        std = excess.std(ddof=1)
        if std < 1e-9:
            return 0.0

        return float(excess.mean() / (std + 1e-9))

    def _sell_stock(self, index, action, prices):
        """
        Sell up to |action| shares of stock[index].
        Paper: no transaction costs.
        """
        if action >= 0:
            return 0

        shares_to_sell = min(abs(action), int(self.shares_held[index]))
        if shares_to_sell == 0:
            return 0

        proceeds = prices[index] * shares_to_sell
        self.cash += proceeds
        self.shares_held[index] -= shares_to_sell
        return shares_to_sell

    def _buy_stock(self, index, action, prices):
        """
        Buy up to action shares of stock[index], limited by available cash.
        Paper: no transaction costs.
        """
        if action <= 0:
            return 0

        price = prices[index]
        if price <= 0:
            return 0

        max_affordable = int(self.cash // price)
        shares_to_buy  = min(action, max_affordable)

        if shares_to_buy == 0:
            return 0

        self.cash -= price * shares_to_buy
        self.shares_held[index] += shares_to_buy
        return shares_to_buy

    def _get_info(self):
        return {
            "day"            : self.day,
            "date"           : self.unique_dates[self.day],
            "cash"           : float(self.cash),
            "shares_held"    : self.shares_held.copy(),
            "portfolio_value": float(self.portfolio_value),
            "daily_return"   : float(self.daily_return_memory[-1]),
            "sharpe_ratio"   : float(self._calculate_sharpe()),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.day      = 0
        self.terminal = False
        self.data     = self._get_day_data()

        self.cash         = float(self.initial_amount)
        self.shares_held  = np.zeros(self.stock_dim, dtype=np.int32)

        prices = self._get_prices()
        self.portfolio_value = self._calculate_portfolio_value(prices)

        self.asset_memory        = [self.portfolio_value]
        self.reward_memory       = [0.0]
        self.daily_return_memory = [0.0]
        self.actions_memory      = []
        self.date_memory         = [self.unique_dates[self.day]]

        return self._get_state(), {}

    def step(self, actions):
        """
        One trading day.

        FIX 4 — Action scaling: use np.round instead of truncation (astype(int))
                 so that action=0.005 → 0 shares (hold), not a spurious non-zero.

        FIX 1 — Reward scaling: normalise change_in_portfolio by initial_amount
                 so all three PSR terms are in the same numerical range.

        FIX 5 — Terminal guard: ensure self.data is updated before returning
                 the final observation so the agent sees the last day's state.
        """
        # ── terminal guard ────────────────────────────────────────────────
        if self.day >= len(self.unique_dates) - 1:
            self.terminal = True
            return self._get_state(), 0.0, True, False, self._get_info()

        # ── FIX 4: round to nearest integer share count ───────────────────
        actions        = np.clip(actions, -1.0, 1.0)
        scaled_actions = np.round(actions * self.hmax).astype(int)   # was: astype(int)

        # ── execute trades at begin-of-day prices ─────────────────────────
        begin_prices = self._get_prices()
        begin_value  = self._calculate_portfolio_value(begin_prices)

        # Sells first (frees cash), then buys — paper ordering
        sell_idx = np.where(scaled_actions < 0)[0]
        buy_idx  = np.where(scaled_actions > 0)[0]

        for idx in sell_idx[np.argsort(scaled_actions[sell_idx])]:
            self._sell_stock(idx, int(scaled_actions[idx]), begin_prices)

        for idx in buy_idx[np.argsort(-scaled_actions[buy_idx])]:
            self._buy_stock(idx, int(scaled_actions[idx]), begin_prices)

        # ── advance to next day ───────────────────────────────────────────
        self.day += 1
        self.data = self._get_day_data()          # FIX 5: always update data

        end_prices = self._get_prices()
        end_value  = self._calculate_portfolio_value(end_prices)

        # ── PSR reward components (paper eq. 1 & 2) ───────────────────────
        change_in_portfolio = end_value - begin_value

        daily_return = (
            change_in_portfolio / begin_value if begin_value > 0 else 0.0
        )

        self.daily_return_memory.append(float(daily_return))
        sharpe_ratio = self._calculate_sharpe()   # FIX 2 applied inside

        # FIX 1: normalise portfolio change so all three terms are comparable
        normalised_portfolio_change = change_in_portfolio / self.initial_amount

        # PSR = change_in_portfolio + sharpe_ratio + alpha * daily_return
        # (paper eq. 1, with normalised portfolio change)
        reward = (
            normalised_portfolio_change
            + sharpe_ratio
            + self.reward_alpha * daily_return
        )

        # ── update bookkeeping ────────────────────────────────────────────
        self.portfolio_value = end_value
        self.asset_memory.append(end_value)
        self.reward_memory.append(float(reward))
        self.actions_memory.append(scaled_actions.copy())
        self.date_memory.append(self.unique_dates[self.day])

        terminated = self.day >= len(self.unique_dates) - 1
        truncated  = False
        self.terminal = terminated

        if self.print_verbosity > 0 and (
            self.day % self.print_verbosity == 0 or terminated
        ):
            print(
                f"day={self.day}  date={self.unique_dates[self.day].date()}  "
                f"portfolio=${self.portfolio_value:,.2f}  "
                f"reward={reward:.6f}  cash=${self.cash:,.2f}  "
                f"sharpe={sharpe_ratio:.4f}  daily_ret={daily_return:.6f}"
            )

        return self._get_state(), float(reward), terminated, truncated, self._get_info()

    def render(self, mode="human"):
        info = self._get_info()
        print(
            f"Date: {info['date'].date()}  "
            f"Portfolio: ${info['portfolio_value']:,.2f}  "
            f"Cash: ${info['cash']:,.2f}  "
            f"Daily Return: {info['daily_return']:.6f}  "
            f"Sharpe: {info['sharpe_ratio']:.6f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Backtesting / evaluation helpers
    # ──────────────────────────────────────────────────────────────────────
    def save_asset_memory(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date"           : self.date_memory,
            "portfolio_value": self.asset_memory,
        })

    def save_reward_memory(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date"  : self.date_memory,
            "reward": self.reward_memory,
        })

    def save_action_memory(self) -> pd.DataFrame:
        if len(self.actions_memory) == 0:
            return pd.DataFrame()
        tickers    = self.date_to_data[self.unique_dates[0]][self.ticker_col].tolist()
        action_df  = pd.DataFrame(self.actions_memory, columns=tickers)
        action_df.insert(0, "date", self.date_memory[1:])
        return action_df

    def get_portfolio_return(self) -> float:
        """Cumulative return over the episode — used in evaluation."""
        if len(self.asset_memory) < 2:
            return 0.0
        return (self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0]

    def get_sharpe_ratio(self) -> float:
        """Episode-level Sharpe ratio (annualised, 252 trading days)."""
        returns = np.array(self.daily_return_memory[1:], dtype=np.float64)
        if len(returns) < 2:
            return 0.0
        excess = returns - self.risk_free_rate
        std    = excess.std(ddof=1)
        if std < 1e-9:
            return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    def get_max_drawdown(self) -> float:
        """Maximum drawdown over the episode."""
        values   = np.array(self.asset_memory, dtype=np.float64)
        peak     = np.maximum.accumulate(values)
        drawdown = (values - peak) / (peak + 1e-9)
        return float(drawdown.min())