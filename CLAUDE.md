# RL Trading Bot - XAU/USD 1H

## Proyecto
Bot de trading entrenado con Reinforcement Learning (PPO) usando datos historicos del oro (XAU/USD) en temporalidad 1H, con dashboard interactivo de velas japonesas.

## Stack
- Python 3.12 + venv
- Stable-Baselines3 (PPO) + Gymnasium
- pandas + pandas-ta (indicadores tecnicos)
- Flask + SocketIO (dashboard)
- Plotly.js (graficos de velas)

## Estructura
```
rl-trading-bot/
├── config.py                          # Hiperparametros y paths
├── main.py                            # CLI: prepare, train, evaluate, visualize
├── src/
│   ├── utils/data_loader.py           # Carga CSV (sep=;), split temporal
│   ├── environment/
│   │   ├── features.py                # 24 indicadores tecnicos + normalizacion
│   │   └── trading_env.py             # Gymnasium env (HOLD/BUY/SELL)
│   ├── agent/
│   │   ├── callbacks.py               # Metricas de trading para TensorBoard
│   │   ├── train.py                   # Entrenamiento PPO
│   │   └── evaluate.py                # Evaluacion + export JSON
│   └── visualization/
│       ├── chart_server.py            # Flask API server
│       └── templates/dashboard.html   # Dashboard Plotly.js
├── data/XAU_1h_data.csv               # ~124K filas, 2004-2025, sep=;
├── models/                            # Modelos guardados (.zip) + scaler.pkl
├── logs/                              # TensorBoard logs
└── results/                           # JSON de evaluacion
```

## Comandos
```bash
source venv/Scripts/activate   # Activar venv
python main.py prepare         # Preprocesar datos y features
python main.py train           # Entrenar PPO (500K steps, ~41 min)
python main.py evaluate        # Evaluar en test set
python main.py visualize       # Dashboard en http://127.0.0.1:5000
```

## Estado actual
- [x] Setup proyecto + venv + dependencias
- [x] Data loader (CSV sep=;, fecha %Y.%m.%d %H:%M)
- [x] Feature engineering (24 indicadores)
- [x] Trading environment (Gymnasium, validado con check_env)
- [x] Entrenamiento PPO completado (500K steps, best_model + final_model guardados)
- [ ] Evaluacion en test set
- [ ] Visualizacion en dashboard

## Notas
- CSV usa separador `;`, no `,`
- Formato fecha: `%Y.%m.%d %H:%M`
- Split temporal SIN shuffle: 70% train / 15% val / 15% test
- Primer entrenamiento: reward negativo (-21.2 avg), esperado para un primer modelo
- Iterar reward shaping y features para mejorar rendimiento
