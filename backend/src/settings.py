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
WINDOW = dt.timedelta(hours=72)

# Coinbase omits minutes with no trades; carry the last price forward at most this many
# minutes (a no-trade minute is a zero return). Longer holes are dropped from the panel.
MAX_FFILL_MINUTES = 5

FORECAST_HORIZON_DEFAULT = 60
FORECAST_HORIZON_MAX = 240
N_SIMULATIONS = 1000
