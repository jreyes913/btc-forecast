"""Download the pre-registered data window from Coinbase Exchange (public API).

Usage (from backend/):  python -m research.fetch_candles
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from research.data_io import DATA_DIR, candle_path
from src.services.candles import CoinbaseClient
from src.settings import PRODUCTS

# Pre-registered (see research/PREREGISTRATION.md): 30 days of history for the current
# method's seasonality profile before the 7-day GARCH warm-up, through the last outcome.
START = dt.datetime(2026, 7, 2, 0, 0, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 15, 2, 0, tzinfo=dt.UTC)
CHUNK = dt.timedelta(days=2)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = CoinbaseClient()
    for product in PRODUCTS:
        path = candle_path(product)
        if path.exists():
            print(f"{product}: already downloaded ({path.name})")
            continue
        t0, parts, start = time.time(), [], START
        while start <= END:
            end = min(start + CHUNK - dt.timedelta(minutes=1), END)
            parts.append(client.fetch_minute_candles(product, start, end))
            start = end + dt.timedelta(minutes=1)
        df = pd.concat(parts, ignore_index=True).drop(columns=["product"])
        df = df.drop_duplicates("time").sort_values("time")
        df.to_csv(path, index=False, compression="gzip")
        expected = int((END - START).total_seconds() // 60) + 1
        print(f"{product}: {len(df):,} candles of {expected:,} minutes "
              f"({expected - len(df):,} minutes without trades) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
