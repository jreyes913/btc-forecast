"""Data-loading glue for the GARCH-FHS backtest (Coinbase 1-minute candles).

Candles are stored as gzipped CSVs, one per product, with Coinbase's bucket OPEN time
in the ``time`` column (UTC).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import PRODUCTS

DATA_DIR = Path(__file__).resolve().parent / "data"
FIELDS = ("close", "high", "low")


def candle_path(product: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{product}.csv.gz"


def load_candles(product: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(candle_path(product, data_dir))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


def load_panel(products=PRODUCTS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Open-time-indexed panel with (field, product) columns, like CandleStore.read_panel."""
    frames = {p: load_candles(p, data_dir).set_index("time") for p in products}
    panel = pd.concat({f: pd.DataFrame({p: frames[p][f] for p in products}) for f in FIELDS},
                      axis=1)
    return panel.sort_index()


def close_time_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Re-index candles by CLOSE time (open + 1 min): the close known at that instant."""
    return pd.DataFrame(
        {"timestamp": candles["time"] + pd.Timedelta(minutes=1), "close": candles["close"]}
    )
