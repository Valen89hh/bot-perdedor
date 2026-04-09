import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
import joblib
import os


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores tecnicos sobre datos OHLCV."""
    feat = pd.DataFrame(index=df.index)

    # --- Tendencia ---
    feat["ema_9"] = ta.ema(df["close"], length=9)
    feat["ema_21"] = ta.ema(df["close"], length=21)
    feat["ema_50"] = ta.ema(df["close"], length=50)
    feat["sma_200"] = ta.sma(df["close"], length=200)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    feat["macd_line"] = macd.iloc[:, 0]
    feat["macd_signal"] = macd.iloc[:, 1]
    feat["macd_hist"] = macd.iloc[:, 2]

    # --- Momentum ---
    feat["rsi_14"] = ta.rsi(df["close"], length=14)

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    feat["stoch_k"] = stoch.iloc[:, 0]
    feat["stoch_d"] = stoch.iloc[:, 1]

    feat["roc_10"] = ta.roc(df["close"], length=10)

    # --- Volatilidad ---
    bbands = ta.bbands(df["close"], length=20, std=2)
    feat["bb_upper"] = bbands.iloc[:, 2]
    feat["bb_middle"] = bbands.iloc[:, 1]
    feat["bb_lower"] = bbands.iloc[:, 0]
    feat["bb_pct"] = (df["close"] - feat["bb_lower"]) / (feat["bb_upper"] - feat["bb_lower"])

    feat["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    feat["volatility"] = df["close"].pct_change().rolling(20).std()

    # --- Volumen ---
    feat["obv"] = ta.obv(df["close"], df["volume"])
    vol_sma = df["volume"].rolling(20).mean()
    feat["volume_ratio"] = df["volume"] / vol_sma.replace(0, 1)

    # --- Precio ---
    feat["log_return"] = np.log(df["close"] / df["close"].shift(1))
    feat["price_vs_ema9"] = (df["close"] - feat["ema_9"]) / df["close"]
    feat["price_vs_ema21"] = (df["close"] - feat["ema_21"]) / df["close"]
    feat["price_vs_ema50"] = (df["close"] - feat["ema_50"]) / df["close"]
    feat["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Drop filas NaN del warmup (SMA 200 necesita 200 velas)
    feat = feat.dropna()

    return feat


def normalize_features(feat: pd.DataFrame, scaler=None, save_path=None):
    """Normaliza features con StandardScaler. Retorna (df_normalizado, scaler)."""
    if scaler is None:
        scaler = StandardScaler()
        values = scaler.fit_transform(feat.values)
    else:
        values = scaler.transform(feat.values)

    result = pd.DataFrame(values, index=feat.index, columns=feat.columns)

    if save_path:
        joblib.dump(scaler, save_path)

    return result, scaler


def get_feature_names():
    """Retorna la lista de nombres de features."""
    return [
        "ema_9", "ema_21", "ema_50", "sma_200",
        "macd_line", "macd_signal", "macd_hist",
        "rsi_14", "stoch_k", "stoch_d", "roc_10",
        "bb_upper", "bb_middle", "bb_lower", "bb_pct",
        "atr_14", "volatility",
        "obv", "volume_ratio",
        "log_return", "price_vs_ema9", "price_vs_ema21", "price_vs_ema50",
        "hl_range",
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import CONFIG
    from src.utils.data_loader import load_data

    df = load_data(CONFIG["data_path"])
    feat = compute_features(df)
    print(f"Features shape: {feat.shape}")
    print(f"Columnas: {list(feat.columns)}")
    print(f"\nEstadisticas:\n{feat.describe()}")

    norm, scaler = normalize_features(feat)
    print(f"\nNormalizadas (media ~0, std ~1):\n{norm.describe().loc[['mean','std']]}")
