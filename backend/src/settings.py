import datetime as dt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = str(DATA_DIR / "candles.duckdb")

PRODUCTS = ["BTC-USD", "ETH-USD", "SPX-USD"]
COINBASE_URL_TEMPLATE = "https://api.exchange.coinbase.com/products/{PRODUCT}/candles"
WINDOW = dt.timedelta(hours=72)



