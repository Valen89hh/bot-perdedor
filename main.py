"""
RL Trading Bot - Entry Point
Uso:
    python main.py prepare     - Preprocesar datos y verificar features
    python main.py train       - Entrenar el agente RL
    python main.py evaluate    - Evaluar en test set
    python main.py visualize   - Abrir dashboard
"""
import sys
import os

from config import CONFIG


def cmd_prepare():
    from src.utils.data_loader import load_data, split_data
    from src.environment.features import compute_features, normalize_features

    print("=" * 60)
    print("  Preparando datos...")
    print("=" * 60)

    df = load_data(CONFIG["data_path"])
    print(f"\nDatos cargados: {len(df)} filas")
    print(f"Rango: {df.index[0]} a {df.index[-1]}")
    print(f"Columnas: {list(df.columns)}")

    train, val, test = split_data(df, CONFIG["train_ratio"], CONFIG["val_ratio"])
    print(f"\nSplit: Train={len(train)} | Val={len(val)} | Test={len(test)}")

    print("\nCalculando features...")
    feat = compute_features(train)
    print(f"Features: {feat.shape[1]} indicadores, {feat.shape[0]} filas (tras warmup)")
    print(f"Columnas: {list(feat.columns)}")

    feat_norm, scaler = normalize_features(
        feat,
        save_path=os.path.join(CONFIG["models_dir"], "scaler.pkl"),
    )
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    print(f"\nScaler guardado en: {CONFIG['models_dir']}/scaler.pkl")
    print("\nDatos listos para entrenamiento.")


def cmd_train():
    from src.agent.train import train
    train(CONFIG)


def cmd_evaluate():
    dataset = sys.argv[2] if len(sys.argv) > 2 else "test"
    from src.agent.evaluate import evaluate
    evaluate(CONFIG, dataset=dataset)


def cmd_visualize():
    from src.visualization.chart_server import run_server
    run_server()


COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "visualize": cmd_visualize,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print(f"Comandos disponibles: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
