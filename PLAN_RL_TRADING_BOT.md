# 🤖 Ultra Plan: Bot de Trading con Aprendizaje por Refuerzo

## Contexto del Proyecto

Crear un bot de trading entrenado con **Reinforcement Learning (RL)** usando datos históricos del **oro (XAU/USD) en temporalidad 1H**. El sistema debe incluir una visualización interactiva con gráfico de velas japonesas que muestre las operaciones del agente durante y después del entrenamiento.

---

## 📁 Estructura del Proyecto

```
rl-trading-bot/
├── data/
│   └── XAU_1h.csv              # CSV histórico del oro (OHLCV)
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── trading_env.py       # Gymnasium Environment personalizado
│   │   └── features.py          # Cálculo de indicadores técnicos
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── train.py             # Script de entrenamiento
│   │   ├── evaluate.py          # Evaluación y métricas
│   │   └── callbacks.py         # Callbacks para logging
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── chart_server.py      # Servidor web para visualización
│   │   └── templates/
│   │       └── dashboard.html   # Dashboard con gráfico de velas
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py       # Carga y preprocesamiento del CSV
├── models/                      # Modelos entrenados guardados
├── logs/                        # TensorBoard logs
├── results/                     # Resultados de backtesting
├── requirements.txt
├── config.py                    # Configuración centralizada
└── main.py                      # Entry point principal
```

---

## 🔧 Stack Tecnológico

| Componente | Tecnología | Propósito |
|---|---|---|
| Lenguaje | Python 3.11+ | Base del proyecto |
| RL Framework | Stable-Baselines3 | Algoritmos RL (PPO, A2C, DQN) |
| Environment | Gymnasium | Entorno de trading personalizado |
| Datos | Pandas + NumPy | Manipulación de datos |
| Indicadores | pandas-ta o ta-lib | RSI, MACD, Bollinger, EMA, etc. |
| Visualización | Plotly (gráfico de velas) | Charts interactivos |
| Dashboard | Flask + Socket.IO | Servidor web en tiempo real |
| Frontend | Plotly.js + HTML/CSS/JS | Interfaz del dashboard |
| Logging | TensorBoard | Métricas de entrenamiento |
| Serialización | Pickle / SB3 save/load | Guardar/cargar modelos |

---

## 📋 Fases de Desarrollo (en orden)

---

### FASE 1: Preparación de Datos y Feature Engineering
**Prioridad: CRÍTICA — Todo depende de esto**

#### 1.1 — Data Loader (`src/utils/data_loader.py`)
- Leer el CSV del oro con columnas: `datetime, open, high, low, close, volume`
- Parsear fechas correctamente y setear como index
- Manejar datos faltantes (forward fill para gaps de mercado)
- Normalizar nombres de columnas a lowercase
- Validar integridad: sin duplicados, orden cronológico, sin gaps extremos
- Split temporal: 70% train, 15% validation, 15% test (SIN shuffle, es serie temporal)

#### 1.2 — Feature Engineering (`src/environment/features.py`)
Calcular indicadores técnicos que serán el **observation space** del agente:

```
Indicadores a implementar:
├── Tendencia
│   ├── EMA 9, 21, 50 (Medias Móviles Exponenciales)
│   ├── SMA 200 (Media Móvil Simple)
│   └── MACD (línea, señal, histograma)
├── Momentum
│   ├── RSI (14 períodos)
│   ├── Stochastic Oscillator (%K, %D)
│   └── ROC (Rate of Change)
├── Volatilidad
│   ├── Bollinger Bands (upper, middle, lower, %B)
│   ├── ATR (Average True Range, 14 períodos)
│   └── Volatilidad histórica (std de returns)
├── Volumen
│   ├── OBV (On Balance Volume)
│   └── Volume SMA ratio
└── Precio
    ├── Returns logarítmicos
    ├── Precio relativo a EMAs (distancia normalizada)
    └── High-Low range normalizado
```

- Normalizar TODOS los features con `StandardScaler` o `MinMaxScaler`
- Guardar el scaler fitted para usarlo en inferencia
- Dropear las primeras N filas donde los indicadores son NaN (warmup period)

---

### FASE 2: Gymnasium Environment
**Prioridad: CRÍTICA — El corazón del sistema**

#### 2.1 — Trading Environment (`src/environment/trading_env.py`)

```python
# Esquema conceptual del Environment

class TradingEnv(gymnasium.Env):
    """
    Environment de trading para oro 1H con RL.
    
    Action Space (Discrete, 3 acciones):
        0 = HOLD (no hacer nada)
        1 = BUY (abrir posición long / cerrar short)
        2 = SELL (abrir posición short / cerrar long)
    
    Observation Space (Box):
        Vector de features normalizados:
        - Indicadores técnicos (calculados en features.py)
        - Estado de la posición actual (flat/long/short)
        - PnL no realizado normalizado
        - Tiempo en la posición actual (normalizado)
        - Balance actual normalizado
    
    Reward Function:
        - Cambio en PnL realizado + no realizado (step reward)
        - Penalización por over-trading (exceso de operaciones)
        - Penalización por drawdown excesivo
        - Bonus por operaciones ganadoras consecutivas
    """
```

**Parámetros clave del environment:**
- `initial_balance`: 10000 USD (configurable)
- `commission`: 0.0002 (spread simulado del oro)
- `max_position_size`: 1 lote (simplificado)
- `max_steps`: longitud del dataset de train
- `leverage`: 1:1 inicialmente (sin apalancamiento)

**Reward shaping detallado:**
```
reward = (
    delta_portfolio_value           # Cambio en valor del portafolio
    - commission_cost               # Costo de comisión si operó
    - 0.001 * is_holding_too_long   # Penalización por posiciones estancadas
    - 0.01 * excessive_trading      # Penalización si opera >30% de los steps
)

# Al final del episodio (terminal reward):
terminal_reward = (
    sharpe_ratio_bonus              # Bonus por buen Sharpe ratio
    + max_drawdown_penalty          # Penalización por drawdown >20%
)
```

**Datos que el environment debe trackear para visualización:**
```python
self.trade_history = []  # Lista de trades: {step, type, price, pnl}
self.portfolio_history = []  # Balance en cada step
self.action_history = []  # Acción tomada en cada step
```

#### 2.2 — Validación del Environment
- Usar `gymnasium.utils.env_checker.check_env()` para validar
- Test manual: correr 100 steps con acciones random, verificar que no crashee
- Verificar que observation shape sea consistente
- Verificar que rewards estén en rango razonable (-1 a 1 idealmente)

---

### FASE 3: Entrenamiento del Agente RL
**Prioridad: ALTA**

#### 3.1 — Configuración (`config.py`)

```python
# Hiperparámetros sugeridos para empezar
CONFIG = {
    "algorithm": "PPO",  # Empezar con PPO, es el más estable
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,        # Discount factor
    "gae_lambda": 0.95,   # GAE lambda
    "clip_range": 0.2,    # PPO clip
    "ent_coef": 0.01,     # Entropy coefficient (exploración)
    "total_timesteps": 500_000,  # Timesteps totales
    "eval_freq": 10_000,  # Evaluar cada N steps
    "save_freq": 50_000,  # Guardar modelo cada N steps
    
    # Network architecture
    "policy_kwargs": {
        "net_arch": [256, 256, 128]  # 3 capas hidden
    }
}
```

#### 3.2 — Script de Entrenamiento (`src/agent/train.py`)
- Crear el environment con wrappers de SB3:
  - `Monitor` para logging
  - `DummyVecEnv` para vectorización
  - `VecNormalize` para normalizar observations y rewards
- Instanciar el modelo PPO con los hiperparámetros
- Configurar callbacks:
  - `EvalCallback`: evaluar en validation set periódicamente
  - `CheckpointCallback`: guardar checkpoints
  - `CustomCallback`: loggear métricas custom (win rate, profit factor, etc.)
- Entrenar con `model.learn()`
- Guardar modelo final y VecNormalize stats

#### 3.3 — Callbacks Personalizados (`src/agent/callbacks.py`)

```python
class TradingMetricsCallback(BaseCallback):
    """
    Callback que loggea métricas de trading durante entrenamiento:
    - Win rate (% de trades ganadores)
    - Profit factor (gross profit / gross loss)
    - Max drawdown
    - Sharpe ratio
    - Número total de trades
    - PnL acumulado
    
    También guarda el trade_history del environment para
    la visualización posterior.
    """
```

#### 3.4 — Evaluación (`src/agent/evaluate.py`)
- Cargar modelo entrenado
- Correr en test set (datos no vistos)
- Generar métricas completas:
  - Total return, Sharpe ratio, Sortino ratio
  - Max drawdown, Calmar ratio
  - Win rate, profit factor
  - Número de trades, duración promedio
- Exportar `trade_history` y `portfolio_history` a JSON para visualización
- Comparar contra baseline: Buy & Hold del mismo período

---

### FASE 4: Visualización con Gráfico de Velas
**Prioridad: ALTA — Lo que quieres ver en pantalla**

#### 4.1 — Backend del Dashboard (`src/visualization/chart_server.py`)

Servidor Flask que:
- Sirve el HTML del dashboard
- Endpoint `/api/candles` → datos OHLCV del CSV
- Endpoint `/api/trades` → historial de trades del agente
- Endpoint `/api/portfolio` → evolución del balance
- Endpoint `/api/metrics` → métricas resumen
- WebSocket con Socket.IO para actualizaciones en tiempo real durante entrenamiento

#### 4.2 — Frontend Dashboard (`src/visualization/templates/dashboard.html`)

**Layout del Dashboard:**

```
┌─────────────────────────────────────────────────────────┐
│  RL Trading Bot — Gold XAU/USD 1H          [Train] [Eval]│
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │                                                 │    │
│  │         GRÁFICO DE VELAS JAPONESAS              │    │
│  │         (Plotly.js candlestick chart)            │    │
│  │                                                 │    │
│  │    🟢 Markers de BUY (triángulo verde arriba)   │    │
│  │    🔴 Markers de SELL (triángulo rojo abajo)    │    │
│  │    ── Líneas conectando entrada → salida        │    │
│  │       (verde si ganó, roja si perdió)           │    │
│  │                                                 │    │
│  │    Overlays: EMA 9, EMA 21, Bollinger Bands     │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │         EQUITY CURVE (evolución del balance)     │    │
│  │         Línea de balance + drawdown shaded       │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │ Win Rate │ Profit F.│ Sharpe   │ Drawdown │          │
│  │  62.4%   │   1.85   │  1.42    │  -8.3%   │          │
│  └──────────┴──────────┴──────────┴──────────┘          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Training Progress: ████████░░ 80% (400K/500K)  │    │
│  │  Episode Reward: 2.34  |  Avg: 1.87  |  Best: 4.1│   │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**Especificaciones del gráfico de velas:**
- Usar `Plotly.js` con tipo `candlestick`
- Zoom/pan interactivo con rangeslider
- Markers de operaciones:
  - **BUY**: triángulo verde apuntando arriba en la vela correspondiente
  - **SELL**: triángulo rojo apuntando abajo
  - Línea punteada conectando cada par entry-exit
  - Color de la línea: verde si PnL > 0, rojo si PnL < 0
- Overlays opcionales (toggleables): EMAs, Bollinger, volumen
- Tooltip al pasar sobre un marker: precio, hora, PnL del trade

---

### FASE 5: Integración y Pipeline Completo
**Prioridad: MEDIA**

#### 5.1 — Main Entry Point (`main.py`)

```bash
# Uso desde terminal:
python main.py prepare      # Preprocesar datos y features
python main.py train         # Entrenar el agente (abre dashboard automáticamente)
python main.py evaluate      # Evaluar en test set
python main.py visualize     # Abrir dashboard con último modelo
python main.py optimize      # Optuna hyperparameter search (futuro)
```

#### 5.2 — Flujo completo:
```
CSV → DataLoader → FeatureEngineering → TradingEnv → PPO Agent
                                                         │
                                          ┌──────────────┤
                                          ▼              ▼
                                      Training      Evaluation
                                          │              │
                                          ▼              ▼
                                   Callbacks ────→ trade_history.json
                                                         │
                                                         ▼
                                                  Flask Dashboard
                                                         │
                                                         ▼
                                              Plotly Candlestick +
                                              Trade Markers + Metrics
```

---

## 🚀 Instrucciones para Claude Code

### Prompt de inicio sugerido:

> Estoy construyendo un bot de trading con Reinforcement Learning. Sigue el plan en `PLAN_RL_TRADING_BOT.md`. Mi CSV está en `data/gold_1h.csv` con datos del oro en temporalidad 1H. Empecemos por la **Fase 1**: crear el data loader y el feature engineering. Después continuamos fase por fase.

### Orden de implementación recomendado:

```
1. requirements.txt (instalar dependencias)
2. config.py
3. src/utils/data_loader.py + tests básicos
4. src/environment/features.py + verificar que los indicadores se calculan bien
5. src/environment/trading_env.py + validar con env_checker
6. src/agent/callbacks.py
7. src/agent/train.py + entrenar primer modelo
8. src/agent/evaluate.py + generar resultados
9. src/visualization/chart_server.py + dashboard.html
10. main.py (integrar todo)
```

### Tips para trabajar con Claude Code:

- **Fase por fase**: No pidas todo de golpe. Ve fase por fase y valida cada una
- **Muestra el CSV**: Al inicio dale unas líneas de tu CSV para que entienda el formato
- **Itera el reward**: La función de recompensa es lo MÁS importante. Pide variantes y experimenta
- **Pide tests**: Pide que cree tests simples para cada componente
- **Debuggea visualmente**: Usa el dashboard desde la Fase 4 para entender qué está haciendo el agente

---

## ⚠️ Advertencias Importantes

1. **No esperes rentabilidad real al inicio.** El primer modelo será malo. La clave es iterar el reward shaping y los features.
2. **Cuidado con overfitting.** Si el agente performa genial en train pero mal en test, está memorizando, no aprendiendo.
3. **Paper trading primero.** NUNCA conectes a dinero real hasta tener meses de paper trading estable.
4. **El mercado cambia.** Un modelo entrenado en datos 2020-2023 puede no funcionar en 2024+. Necesitarás reentrenamiento periódico.
5. **Binarias ≠ Trading legítimo.** Si puedes, enfócate en mercados regulados.

---

## 📊 Requirements.txt

```
gymnasium==0.29.1
stable-baselines3==2.3.2
pandas==2.2.0
numpy==1.26.4
pandas-ta==0.3.14b1
plotly==5.18.0
flask==3.0.0
flask-socketio==5.3.6
tensorboard==2.15.1
scikit-learn==1.4.0
python-dotenv==1.0.0
```
