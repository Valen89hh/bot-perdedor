import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Environment de trading para oro 1H con RL.

    Action Space (Discrete, 3 acciones):
        0 = HOLD
        1 = BUY (abrir long / cerrar short)
        2 = SELL (abrir short / cerrar long)

    Observation Space (Box):
        Features tecnicos normalizados + estado del agente
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_features: pd.DataFrame,
        df_prices: pd.DataFrame,
        initial_balance: float = 10000,
        commission: float = 0.0002,
        max_position_size: int = 1,
    ):
        super().__init__()

        self.df_features = df_features.values.astype(np.float32)
        self.df_prices = df_prices.loc[df_features.index].copy()
        self.closes = self.df_prices["close"].values.astype(np.float64)
        self.n_steps = len(self.df_features)
        self.feature_names = list(df_features.columns)
        self.timestamps = df_features.index.tolist()

        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position_size = max_position_size

        # Action: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation: features + [position, unrealized_pnl, time_in_pos, balance_norm]
        n_features = self.df_features.shape[1]
        n_extra = 4  # position state, unrealized pnl, time in position, balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + n_extra,),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.time_in_position = 0
        self.total_trades = 0
        self.winning_trades = 0

        self.trade_history = []
        self.portfolio_history = [self.initial_balance]
        self.action_history = []

        self._total_reward = 0.0
        self._prev_portfolio_value = self.initial_balance

        return self._get_observation(), {}

    def _get_observation(self):
        features = self.df_features[self.current_step]

        # Estado del agente normalizado
        position_state = float(self.position)
        unrealized_pnl = self._unrealized_pnl() / self.initial_balance
        time_norm = min(self.time_in_position / 100.0, 1.0)
        balance_norm = (self.balance / self.initial_balance) - 1.0

        extra = np.array(
            [position_state, unrealized_pnl, time_norm, balance_norm],
            dtype=np.float32,
        )
        return np.concatenate([features, extra])

    def _unrealized_pnl(self):
        if self.position == 0:
            return 0.0
        price = self.closes[self.current_step]
        if self.position == 1:  # long
            return (price - self.entry_price) * self.max_position_size
        else:  # short
            return (self.entry_price - price) * self.max_position_size

    def _portfolio_value(self):
        return self.balance + self._unrealized_pnl()

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.closes[self.current_step]
        commission_cost = 0.0
        trade_pnl = 0.0

        # Ejecutar accion
        if action == 1:  # BUY
            if self.position == -1:  # Cerrar short
                trade_pnl = (self.entry_price - current_price) * self.max_position_size
                commission_cost = current_price * self.commission * 2
                self.balance += trade_pnl - commission_cost
                self._record_trade("close_short", current_price, trade_pnl - commission_cost)
                self.position = 0
                self.entry_price = 0.0
                self.time_in_position = 0
            if self.position == 0:  # Abrir long
                self.position = 1
                self.entry_price = current_price
                commission_cost += current_price * self.commission
                self.balance -= commission_cost
                self.time_in_position = 0
                self._record_trade("open_long", current_price, 0)

        elif action == 2:  # SELL
            if self.position == 1:  # Cerrar long
                trade_pnl = (current_price - self.entry_price) * self.max_position_size
                commission_cost = current_price * self.commission * 2
                self.balance += trade_pnl - commission_cost
                self._record_trade("close_long", current_price, trade_pnl - commission_cost)
                self.position = 0
                self.entry_price = 0.0
                self.time_in_position = 0
            if self.position == 0:  # Abrir short
                self.position = -1
                self.entry_price = current_price
                commission_cost += current_price * self.commission
                self.balance -= commission_cost
                self.time_in_position = 0
                self._record_trade("open_short", current_price, 0)

        # HOLD o posicion abierta
        if self.position != 0:
            self.time_in_position += 1

        # Calcular reward
        current_portfolio = self._portfolio_value()
        reward = (current_portfolio - self._prev_portfolio_value) / self.initial_balance

        # Penalizacion por over-trading
        trade_ratio = self.total_trades / max(self.current_step + 1, 1)
        if trade_ratio > 0.3:
            reward -= 0.01

        # Penalizacion por posiciones estancadas (>100 horas)
        if self.time_in_position > 100:
            reward -= 0.001

        self._prev_portfolio_value = current_portfolio
        self._total_reward += reward

        # Guardar historial
        self.portfolio_history.append(current_portfolio)
        self.action_history.append(action)

        # Avanzar step
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Drawdown check - si pierde mas del 50% terminar
        if current_portfolio < self.initial_balance * 0.5:
            terminated = True

        # Terminal reward
        if terminated:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1] if len(self.portfolio_history) > 1 else [0]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
                reward += np.clip(sharpe / 10, -0.5, 0.5)

            # Max drawdown penalty
            peak = np.maximum.accumulate(self.portfolio_history)
            drawdowns = (np.array(self.portfolio_history) - peak) / peak
            max_dd = np.min(drawdowns)
            if max_dd < -0.2:
                reward += max_dd  # penalizacion

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "balance": self.balance,
            "portfolio_value": current_portfolio,
            "position": self.position,
            "total_trades": self.total_trades,
        }

        return obs, float(reward), terminated, truncated, info

    def _record_trade(self, trade_type, price, pnl):
        self.trade_history.append({
            "step": self.current_step,
            "timestamp": str(self.timestamps[self.current_step]),
            "type": trade_type,
            "price": float(price),
            "pnl": float(pnl),
        })
        if "close" in trade_type:
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1

    def get_metrics(self):
        """Retorna metricas del episodio."""
        portfolio = np.array(self.portfolio_history)
        returns = np.diff(portfolio) / portfolio[:-1] if len(portfolio) > 1 else np.array([0])

        total_return = (portfolio[-1] / portfolio[0] - 1) * 100
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100

        # Sharpe ratio anualizado (1H = 24*252 periodos por ano)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)

        # Max drawdown
        peak = np.maximum.accumulate(portfolio)
        drawdowns = (portfolio - peak) / peak
        max_drawdown = np.min(drawdowns) * 100

        # Profit factor
        winning_pnls = [t["pnl"] for t in self.trade_history if "close" in t["type"] and t["pnl"] > 0]
        losing_pnls = [t["pnl"] for t in self.trade_history if "close" in t["type"] and t["pnl"] < 0]
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "total_trades": self.total_trades,
            "final_balance": round(portfolio[-1], 2),
        }
