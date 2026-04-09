import os
import sys
import json
import numpy as np

from stable_baselines3 import PPO, A2C, DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import CONFIG
from src.utils.data_loader import load_data, split_data
from src.environment.features import compute_features, normalize_features
from src.environment.trading_env import TradingEnv

ALGORITHMS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def evaluate(config=None, model_path=None, dataset="test"):
    """Evalua el modelo entrenado en el dataset especificado."""
    if config is None:
        config = CONFIG

    print("=" * 60)
    print(f"  RL Trading Bot - Evaluacion ({dataset})")
    print("=" * 60)

    # Cargar datos
    print("\n[1/4] Cargando datos...")
    df = load_data(config["data_path"])
    train_df, val_df, test_df = split_data(df, config["train_ratio"], config["val_ratio"])

    if dataset == "test":
        eval_df = test_df
    elif dataset == "val":
        eval_df = val_df
    else:
        eval_df = train_df

    # Features
    print("[2/4] Calculando features...")
    import joblib
    scaler_path = os.path.join(config["models_dir"], "scaler.pkl")
    scaler = joblib.load(scaler_path)

    eval_feat = compute_features(eval_df)
    eval_feat_norm, _ = normalize_features(eval_feat, scaler=scaler)
    
    eval_prices = eval_df.loc[eval_feat_norm.index]

    # Cargar modelo
    print("[3/4] Cargando modelo...")
    if model_path is None:
        # Intentar final_model primero, luego best_model
        final_path = os.path.join(config["models_dir"], "best_model.zip")
        best_path = os.path.join(config["models_dir"], "best_model.zip")
        if os.path.exists(final_path):
            model_path = final_path
        else:
            model_path = best_path

    algo_class = ALGORITHMS[config["algorithm"]]
    model = algo_class.load(model_path)
    print(f"    Modelo: {model_path}")

    # Evaluar
    print("[4/4] Ejecutando evaluacion...")
    env = TradingEnv(
        df_features=eval_feat_norm,
        df_prices=eval_prices,
        initial_balance=config["initial_balance"],
        commission=config["commission"],
        max_position_size=config["max_position_size"],
    )

    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if truncated:
            break

    # Metricas
    metrics = env.get_metrics()

    # Buy & Hold baseline
    start_price = eval_prices["close"].iloc[0]
    end_price = eval_prices["close"].iloc[-1]
    bh_return = ((end_price / start_price) - 1) * 100

    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"  Return:         {metrics['total_return_pct']:>10.2f}%")
    print(f"  Buy & Hold:     {bh_return:>10.2f}%")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:>10.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown_pct']:>10.2f}%")
    print(f"  Win Rate:       {metrics['win_rate_pct']:>10.2f}%")
    print(f"  Profit Factor:  {metrics['profit_factor']:>10.4f}")
    print(f"  Total Trades:   {metrics['total_trades']:>10d}")
    print(f"  Final Balance:  ${metrics['final_balance']:>10,.2f}")
    print("=" * 60)

    # Exportar resultados
    os.makedirs(config["results_dir"], exist_ok=True)

    results = {
        "metrics": metrics,
        "buy_hold_return_pct": round(bh_return, 2),
        "trade_history": env.trade_history,
        "portfolio_history": [float(x) for x in env.portfolio_history],
        "action_history": [int(x) for x in env.action_history],
        "timestamps": [str(t) for t in env.timestamps],
        "prices": {
            "open": eval_prices["open"].tolist(),
            "high": eval_prices["high"].tolist(),
            "low": eval_prices["low"].tolist(),
            "close": eval_prices["close"].tolist(),
            "volume": eval_prices["volume"].tolist(),
            "datetime": [str(t) for t in eval_prices.index],
        },
    }

    results_path = os.path.join(config["results_dir"], f"evaluation_{dataset}.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"\nResultados guardados en: {results_path}")

    return results


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "test"
    evaluate(dataset=dataset)
