import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    """Loggea metricas de trading durante el entrenamiento."""

    def __init__(self, eval_env=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        # Loggear metricas cada vez que un episodio termina
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            infos = self.locals.get("infos", [])
            for info in infos:
                if "balance" in info:
                    self.logger.record("trading/balance", info["balance"])
                    self.logger.record("trading/portfolio_value", info["portfolio_value"])
                    self.logger.record("trading/total_trades", info["total_trades"])
                    self.logger.record("trading/position", info["position"])

        # Acceder al env para metricas detalladas
        if self.n_calls % 10000 == 0:
            try:
                env = self.training_env.envs[0]
                if hasattr(env, "get_metrics"):
                    metrics = env.get_metrics()
                    for key, value in metrics.items():
                        self.logger.record(f"trading/{key}", value)
                    if self.verbose > 0:
                        print(f"  Step {self.n_calls}: Return={metrics['total_return_pct']:.1f}%, "
                              f"WinRate={metrics['win_rate_pct']:.1f}%, "
                              f"Trades={metrics['total_trades']}")
            except Exception:
                pass

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print("Entrenamiento finalizado.")
