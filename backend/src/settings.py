import datetime as dt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DUCKDB_PATH = str(DATA_DIR / "candles.duckdb")

# The asset being forecast, and the basket whose first principal component
# serves as the "market" factor in the CAPM stage.
TARGET = "BTC-USD"
FACTOR_PRODUCTS = ["ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"]
PRODUCTS = [TARGET, *FACTOR_PRODUCTS]

COINBASE_URL_TEMPLATE = "https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

# How much candle history to keep (for seasonality, replay, and fit-window comparisons)
STORE_WINDOW = dt.timedelta(days=30)
# How much of it the production model is fitted on
FIT_WINDOW = dt.timedelta(hours=72)
# Fit windows compared by the replay backtest
FIT_WINDOW_CANDIDATES = (dt.timedelta(hours=24), dt.timedelta(hours=72), dt.timedelta(days=7))
# How much history the candle chart shows
CHART_WINDOW = dt.timedelta(hours=72)

# Coinbase omits minutes with no trades; carry the last price forward at most this many
# minutes (a no-trade minute is a zero return). Longer holes are dropped from the panel.
MAX_FFILL_MINUTES = 5

# 120 minutes so the cone always covers the next *and* the following top-of-hour
# deadline (the next one can be up to ~61 minutes past the cone's origin).
FORECAST_HORIZON_DEFAULT = 120
FORECAST_HORIZON_MAX = 240
N_SIMULATIONS = 1000
# Which forecaster draws the dashboard cone and strike table: "ensemble" (equal-weight
# quantile average of factor, GJR-only and EWMA; best or tied-best on real-data
# scoring at 15-60 minutes) or "factor". Scoring always logs every model separately.
CONE_MODEL = "ensemble"

# Strike table: Kalshi's strike spacing, and the minimum gap (in percentage points)
# between the model's probability and the price before a trade is worth taking.
LADDER_STRIKE_STEP = 100.0
LADDER_EDGE_POINTS = 10

# Adaptive ensembles (scored challengers). Each resolved outcome updates a discounted
# score per member: S <- FORGETTING * S + score; weights = softmax(S). 0.99 remembers
# roughly the last 100 resolved forecasts, so weights can shift with market regimes
# without collapsing onto one model.
ENSEMBLE_FORGETTING = 0.99
# Loss-based variant: score = -ETA * (member pinball loss / mean member loss)
ENSEMBLE_LOSS_ETA = 1.0

# Time-of-day volatility profile
SEASON_BUCKET_MINUTES = 15

# Forecast scoring
LOG_EVERY_MINUTES = 5
LOG_HORIZONS = (1, 15, 30, 60)
REPLAY_STEP_MINUTES = 30
FORECAST_LOG_RETENTION = dt.timedelta(days=90)

# Worker cadence: how often to check for new/late bars, and to refresh the forming bar
SYNC_EVERY_SECONDS = 15
LIVE_POLL_SECONDS = 10
