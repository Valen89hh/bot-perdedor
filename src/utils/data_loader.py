import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Carga el CSV del oro, parsea fechas y normaliza columnas."""
    df = pd.read_csv(
        filepath,
        sep=";",
        parse_dates=["Date"],
        date_format="%Y.%m.%d %H:%M",
    )
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "datetime"})
    df = df.set_index("datetime")
    df = df.sort_index()

    # Eliminar duplicados de índice
    df = df[~df.index.duplicated(keep="first")]

    # Forward fill para gaps de mercado
    df = df.ffill()

    # Eliminar filas con NaN restantes (inicio del dataset)
    df = df.dropna()

    # Validar integridad
    assert df.index.is_monotonic_increasing, "Los datos no están en orden cronológico"
    assert not df.index.has_duplicates, "Hay fechas duplicadas"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "Hay precios <= 0"

    return df


def split_data(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """Split temporal sin shuffle: train / validation / test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


if __name__ == "__main__":
    from config import CONFIG

    df = load_data(CONFIG["data_path"])
    print(f"Total filas: {len(df)}")
    print(f"Rango: {df.index[0]} a {df.index[-1]}")
    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nPrimeras filas:\n{df.head()}")
    print(f"\nÚltimas filas:\n{df.tail()}")

    train, val, test = split_data(df)
    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
