import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """
    Environment de trading para oro 1H con RL (solo longs).

    Action Space (Discrete, 3 acciones):
        0 = HOLD (no hacer nada)
        1 = OPEN LONG (solo si esta flat)
        2 = CLOSE LONG (solo si tiene posicion abierta)

    Si intenta OPEN estando en posicion o CLOSE estando flat, se fuerza HOLD.

    Observation Space (Box):
        Features tecnicos normalizados + estado del agente

    Rules:
        - Stop loss obligatorio: cierra automaticamente si pierde > -2%
        - Take profit bonus: +0.2 reward por step si ganancia > +3%
        - Cooldown de 5 velas despues de cerrar posicion
    """

    metadata = {"render_modes": ["human"]}

    STOP_LOSS_PCT = -2.0
    TAKE_PROFIT_PCT = 3.0
    TAKE_PROFIT_BONUS = 0.2
    COOLDOWN_STEPS = 5

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

        # Señal de tendencia (EMA9 vs EMA50)
        close_series = self.df_prices["close"]
        ema9 = close_series.ewm(span=9, adjust=False).mean()
        ema50 = close_series.ewm(span=50, adjust=False).mean()
        self.trend_signal = np.where(ema9.values > ema50.values, 1.0, -1.0).astype(np.float32)

        self.initial_balance = initial_balance
        self.commission = commission
        self.max_position_size = max_position_size

        # Action: 0=HOLD, 1=OPEN LONG, 2=CLOSE LONG
        self.action_space = spaces.Discrete(3)

        # Observation: features + [position, unrealized_pnl, time_in_pos, balance_norm, trend, cooldown_norm]
        n_features = self.df_features.shape[1]
        n_extra = 6
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
        self.position = 0  # 0=flat, 1=long
        self.entry_price = 0.0
        self.entry_trend = 0.0
        self.time_in_position = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.cooldown_remaining = 0

        self.trade_history = []
        self.portfolio_history = [self.initial_balance]
        self.action_history = []

        self._total_reward = 0.0
        self._prev_portfolio_value = self.initial_balance

        return self._get_observation(), {}

    def _get_observation(self):
        features = self.df_features[self.current_step]

        position_state = float(self.position)
        unrealized_pnl = self._unrealized_pnl() / self.initial_balance
        time_norm = min(self.time_in_position / 100.0, 1.0)
        balance_norm = (self.balance / self.initial_balance) - 1.0
        trend = self.trend_signal[self.current_step]
        cooldown_norm = min(self.cooldown_remaining / self.COOLDOWN_STEPS, 1.0)

        extra = np.array(
            [position_state, unrealized_pnl, time_norm, balance_norm, trend, cooldown_norm],
            dtype=np.float32,
        )
        return np.concatenate([features, extra])

    def _unrealized_pnl(self):
        if self.position == 0:
            return 0.0
        price = self.closes[self.current_step]
        return (price - self.entry_price) * self.max_position_size

    def _portfolio_value(self):
        return self.balance + self._unrealized_pnl()

    def _close_long(self, price):
        """Cierra posicion long y retorna net_pnl."""
        trade_pnl = (price - self.entry_price) * self.max_position_size
        commission_cost = price * self.commission * 2
        self.balance += trade_pnl - commission_cost
        net_pnl = trade_pnl - commission_cost
        self._record_trade("close_long", price, net_pnl)
        self.position = 0
        self.entry_price = 0.0
        self.entry_trend = 0.0
        self.time_in_position = 0
        self.cooldown_remaining = self.COOLDOWN_STEPS
        return net_pnl

    def step(self, action):
        assert self.action_space.contains(action)

        current_price = self.closes[self.current_step]
        reward = 0.0

        # --- Stop loss obligatorio ---
        if self.position == 1:
            unrealized_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            if unrealized_pct <= self.STOP_LOSS_PCT:
                entry_px = self.entry_price
                net_pnl = self._close_long(current_price)
                pnl_pct = (net_pnl / entry_px) * 100
                reward += pnl_pct * 3.0  # penalizar perdedor
                action = 0  # ya se cerro

        # Decrementar cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        # --- Forzar HOLD si accion invalida ---
        if action == 1 and (self.position == 1 or self.cooldown_remaining > 0):
            action = 0  # no puede abrir si ya tiene posicion o esta en cooldown
        if action == 2 and self.position == 0:
            action = 0  # no puede cerrar si no tiene posicion

        # --- Ejecutar accion ---
        if action == 1:  # OPEN LONG
            self.position = 1
            self.entry_price = current_price
            self.entry_trend = self.trend_signal[self.current_step]
            commission_cost = current_price * self.commission
            self.balance -= commission_cost
            self.time_in_position = 0
            self._record_trade("open_long", current_price, 0)

        elif action == 2:  # CLOSE LONG
            entry_px = self.entry_price
            net_pnl = self._close_long(current_price)
            pnl_pct = (net_pnl / entry_px) * 100
            # Reward por calidad del trade
            if pnl_pct > 0:
                reward += pnl_pct * 2.0   # amplificar ganadores
            else:
                reward += pnl_pct * 3.0   # penalizar perdedores mas fuerte
            # Bonus por cerrar ganador a favor de tendencia
            if net_pnl > 0 and self.entry_trend > 0:
                reward += 0.1

        # Incrementar tiempo en posicion
        if self.position == 1:
            self.time_in_position += 1

        # --- Step reward por posicion abierta ---
        if self.position == 1:
            unrealized_pct = ((current_price - self.entry_price) / self.entry_price) * 100
            if unrealized_pct > 0:
                reward += 0.01   # comodidad en ganador
            else:
                reward -= 0.02   # urgencia en perdedor

            # Take profit bonus
            if unrealized_pct >= self.TAKE_PROFIT_PCT:
                reward += self.TAKE_PROFIT_BONUS

        # Penalizacion por posiciones estancadas (>100 horas)
        if self.time_in_position > 100:
            reward -= 0.001

        # Portfolio tracking
        current_portfolio = self._portfolio_value()
        self._prev_portfolio_value = current_portfolio
        self._total_reward += reward

        self.portfolio_history.append(current_portfolio)
        self.action_history.append(action)

        # Avanzar step
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Drawdown check
        if current_portfolio < self.initial_balance * 0.5:
            terminated = True

        # Terminal reward (Sharpe + drawdown penalty)
        if terminated:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1] if len(self.portfolio_history) > 1 else [0]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6.5)
                reward += np.clip(sharpe * 5, -10, 10)

            peak = np.maximum.accumulate(self.portfolio_history)
            drawdowns = (np.array(self.portfolio_history) - peak) / peak
            max_dd = np.min(drawdowns)
            if max_dd < -0.2:
                reward += max_dd

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

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)

        peak = np.maximum.accumulate(portfolio)
        drawdowns = (portfolio - peak) / peak
        max_drawdown = np.min(drawdowns) * 100

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
