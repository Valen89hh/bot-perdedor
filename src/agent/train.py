import os
import sys
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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


def train(config=None):
    """Entrena el agente RL."""
    if config is None:
        config = CONFIG

    print("=" * 60)
    print("  RL Trading Bot - Entrenamiento")
    print("=" * 60)

    # Cargar datos
    print("\n[1/5] Cargando datos...")
    df = load_data(config["data_path"])
    train_df, val_df, _ = split_data(df, config["train_ratio"], config["val_ratio"])

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

    print(f"    Train: {len(train_feat_norm)} steps | Val: {len(val_feat_norm)} steps")

    # Environments
    print("[3/5] Creando environments...")
    train_env = DummyVecEnv([make_env(train_feat_norm, train_prices, config)])
    eval_env = DummyVecEnv([make_env(val_feat_norm, val_prices, config)])

    # Modelo
    print("[4/5] Inicializando modelo...")
    algo_class = ALGORITHMS[config["algorithm"]]
    algo_kwargs = {
        "policy": config["policy"],
        "env": train_env,
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
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

    model = algo_class(**algo_kwargs)

    # Callbacks
    os.makedirs(config["models_dir"], exist_ok=True)
    os.makedirs(config["logs_dir"], exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config["models_dir"],
        log_path=config["logs_dir"],
        eval_freq=config["eval_freq"],
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=config["models_dir"],
        name_prefix="rl_trading",
    )
    trading_callback = TradingMetricsCallback(verbose=1)

    callbacks = CallbackList([eval_callback, checkpoint_callback, trading_callback])

    # Entrenar
    print(f"[5/5] Entrenando {config['algorithm']} por {config['total_timesteps']:,} timesteps...")
    print("-" * 60)
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # Guardar modelo final
    model_path = os.path.join(config["models_dir"], "final_model")
    model.save(model_path)
    print(f"\nModelo guardado en: {model_path}")

    return model


if __name__ == "__main__":
    train()
