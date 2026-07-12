"""
Application configuration for the Kalshi BTC 15-minute prediction system.

Centralizes paths, API endpoints, model hyperparameters, and trading thresholds
so all modules share a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "kalka.duckdb"
MODEL_DIR = DATA_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------
BINANCE_BASE_URL = "https://data-api.binance.vision"
BINANCE_FUTURES_URL = "https://fapi.binance.com"
# ---------------------------------------------------------------------------
# Coinbase
# ---------------------------------------------------------------------------
COINBASE_BASE_URL = "https://api.exchange.coinbase.com"
COINBASE_PRODUCT_ID = "BTC-USD"

# ---------------------------------------------------------------------------
# Kraken
# ---------------------------------------------------------------------------
KRAKEN_BASE_URL = "https://api.kraken.com"
KRAKEN_PAIR = "XBTUSD"

# ---------------------------------------------------------------------------
# Shared market settings
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
CANDLE_INTERVAL = "1m"
CANDLE_LIMIT = 500  # candles per request
EXCHANGES = ("binance", "coinbase", "kraken")

# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_TICKER = "KXBTC15M"

# ---------------------------------------------------------------------------
# CF Benchmarks BRTI (Kalshi settlement index)
# ---------------------------------------------------------------------------
BRTI_INDEX_ID = "BRTI"
SETTLEMENT_SECONDS = 60  # Kalshi averages 60 one-second BRTI prints
# Optional: KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH env vars for live BRTI feed

# ---------------------------------------------------------------------------
# Market window
# ---------------------------------------------------------------------------
WINDOW_MINUTES = 15

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Trading thresholds
# ---------------------------------------------------------------------------
YES_THRESHOLD = 0.55   # recommend BUY YES above this
NO_THRESHOLD = 0.45    # recommend BUY NO below this
CONFIDENCE_THRESHOLD = 0.55  # minimum conviction to output YES/NO label
CONVICTION_THRESHOLD = CONFIDENCE_THRESHOLD  # alias
MIN_EDGE_THRESHOLD = 0.05  # minimum executable edge to signal ENTER

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 10
