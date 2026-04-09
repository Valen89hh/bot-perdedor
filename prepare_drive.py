"""
Copia solo los archivos necesarios a una carpeta lista para subir a Google Drive.
Excluye venv, logs, models, results, __pycache__, .git

Uso: python prepare_drive.py
"""

import shutil
import os

SRC = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(os.path.dirname(SRC), "rl-trading-bot-drive")

INCLUDE = [
    "data",
    "src",
    "config.py",
    "main.py",
    "requirements.txt",
    "train_colab.ipynb",
    "CLAUDE.md",
]

if os.path.exists(DST):
    shutil.rmtree(DST)

os.makedirs(DST)

for item in INCLUDE:
    src_path = os.path.join(SRC, item)
    dst_path = os.path.join(DST, item)
    if not os.path.exists(src_path):
        print(f"  SKIP (no existe): {item}")
        continue
    if os.path.isdir(src_path):
        shutil.copytree(
            src_path, dst_path,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    else:
        shutil.copy2(src_path, dst_path)
    print(f"  OK: {item}")

# Crear carpetas vacias que el proyecto espera
for folder in ["models", "results", "logs"]:
    os.makedirs(os.path.join(DST, folder), exist_ok=True)

# Mostrar resultado
total = 0
for dirpath, _, filenames in os.walk(DST):
    for f in filenames:
        total += os.path.getsize(os.path.join(dirpath, f))

print(f"\nListo: {DST}")
print(f"Tamano total: {total / 1024 / 1024:.1f} MB")
print(f"\nSube esta carpeta a Google Drive en: My Drive/TECSUP/rl-trading-bot")
