import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # --- Datos ---
    "data_path": os.path.join(BASE_DIR, "data", "XAU_1h_data.csv"),
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # --- Environment ---
    "initial_balance": 10000,
    "commission": 0.0002,
    "max_position_size": 1,
    "leverage": 1,

    # --- RL Algorithm ---
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "total_timesteps": 300_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 5,
    "save_freq": 10_000,

    # --- Network ---
    "device": "cpu",
    "n_envs": 4,
    "policy_kwargs": {
        "net_arch": [128, 64],
    },

    # --- Paths ---
    "models_dir": os.path.join(BASE_DIR, "models"),
    "logs_dir": os.path.join(BASE_DIR, "logs"),
    "results_dir": os.path.join(BASE_DIR, "results"),

    # --- Visualization ---
    "dashboard_host": "127.0.0.1",
    "dashboard_port": 5000,
}
