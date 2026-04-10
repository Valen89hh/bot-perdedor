import os
import sys
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import CONFIG
from src.utils.data_loader import load_data, split_data
from src.environment.features import compute_features, normalize_features
from src.environment.trading_env import TradingEnv
from src.agent.callbacks import TradingMetricsCallback


ALGORITHMS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def make_env(df_features, df_prices, config):
    """Crea un environment wrapeado con Monitor."""
    def _init():
        env = TradingEnv(
            df_features=df_features,
            df_prices=df_prices,
            initial_balance=config["initial_balance"],
            commission=config["commission"],
            max_position_size=config["max_position_size"],
        )
        env = Monitor(env)
        return env
    return _init


def create_model(config, train_env):
    """Crea un modelo nuevo desde cero."""
    algo_class = ALGORITHMS[config["algorithm"]]
    algo_kwargs = {
        "policy": config["policy"],
        "env": train_env,
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "device": config.get("device", "auto"),
        "verbose": 1,
        "tensorboard_log": config["logs_dir"],
        "policy_kwargs": config["policy_kwargs"],
    }

    if config["algorithm"] == "PPO":
        algo_kwargs.update({
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "n_epochs": config["n_epochs"],
            "gae_lambda": config["gae_lambda"],
            "clip_range": config["clip_range"],
            "ent_coef": config["ent_coef"],
        })
    elif config["algorithm"] == "A2C":
        algo_kwargs.update({
            "n_steps": config["n_steps"],
            "ent_coef": config["ent_coef"],
        })

    return algo_class(**algo_kwargs)


def create_callbacks(config, eval_env):
    """Crea callbacks para entrenamiento."""
    os.makedirs(config["models_dir"], exist_ok=True)
    os.makedirs(config["logs_dir"], exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config["models_dir"],
        log_path=config["logs_dir"],
        eval_freq=config["eval_freq"],
        n_eval_episodes=config.get("n_eval_episodes", 5),
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["models_dir"],
        name_prefix="rl_trading",
        save_total_limit=config.get("save_total_limit", 5),
    )
    trading_callback = TradingMetricsCallback(verbose=1)

    return CallbackList([eval_callback, checkpoint_callback, trading_callback])


def train(config=None):
    """Entrena el agente RL en una sola pasada sobre todo el dataset de train."""
    if config is None:
        config = CONFIG

    print("=" * 60)
    print("  RL Trading Bot - Entrenamiento (Solo Longs)")
    print("=" * 60)

    # Cargar datos
    print("\n[1/5] Cargando datos...")
    df = load_data(config["data_path"])
    train_df, val_df, _ = split_data(df, config["train_ratio"], config["val_ratio"])

    # Asegurar que existan los directorios de salida
    os.makedirs(config["models_dir"], exist_ok=True)
    os.makedirs(config["logs_dir"], exist_ok=True)

    # Features
    print("[2/5] Calculando features...")
    train_feat = compute_features(train_df)
    train_feat_norm, scaler = normalize_features(
        train_feat,
        save_path=os.path.join(config["models_dir"], "scaler.pkl"),
    )
    train_prices = train_df.loc[train_feat_norm.index]

    val_feat = compute_features(val_df)
    val_feat_norm, _ = normalize_features(val_feat, scaler=scaler)
    val_prices = val_df.loc[val_feat_norm.index]

    # Crear environments
    n_envs = config.get("n_envs", 4)
    print(f"[3/5] Creando environments (train: {len(train_feat_norm)} steps, {n_envs} envs paralelos)...")
    train_env = DummyVecEnv([make_env(train_feat_norm, train_prices, config) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(val_feat_norm, val_prices, config)])

    # Crear modelo y entrenar
    model = create_model(config, train_env)
    callbacks = create_callbacks(config, eval_env)

    print(f"\n[4/5] Entrenando {config['total_timesteps']:,} steps...")
    print("-" * 60)
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # Guardar modelo final
    print("\n[5/5] Guardando modelo...")
    model_path = os.path.join(config["models_dir"], "final_model")
    model.save(model_path)
    print(f"    Modelo guardado en: {model_path}")

    print("\nEntrenamiento completado.")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train()
