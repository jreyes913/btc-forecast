import dataclasses as dc
import datetime as dt
import threading
import time as tm

import duckdb as dd
import pandas as pd
import requests as rq

from src.settings import COINBASE_URL_TEMPLATE, PRODUCTS, WINDOW

TIME = "time"
LOW = "low"
HIGH = "high"
OPEN = "open"
CLOSE = "close"
VOLUME = "volume"
PRODUCT = "product"
COLS = [TIME, LOW, HIGH, OPEN, CLOSE, VOLUME]


@dc.dataclass
class Candles:
    con: dd.DuckDBPyConnection
    products: list[str] = dc.field(default_factory=lambda: list(PRODUCTS))
    db_lock: threading.Lock = dc.field(default_factory=threading.Lock)
    dfs: dict[str, pd.DataFrame] = dc.field(default_factory=dict)

    def fetch_minute_candles(
        self, product: str, start_time: dt.datetime, end_time: dt.datetime
    ) -> pd.DataFrame:
        url = COINBASE_URL_TEMPLATE.format(PRODUCT=product)
        candle_rows = []
        current_time_step = start_time

        while current_time_step < end_time:
            next_time_step = min(
                current_time_step + dt.timedelta(minutes=299), end_time
            )
            params = {
                "granularity": 60,
                "start": current_time_step.isoformat(),
                "end": next_time_step.isoformat(),
            }
            request = rq.get(url, params=params, timeout=10)
            request.raise_for_status()
            candle_rows += request.json()
            current_time_step = next_time_step
            tm.sleep(0.2)

        df = pd.DataFrame(candle_rows, columns=COLS)
        df[TIME] = pd.to_datetime(df[TIME], unit="s", utc=True)
        df[PRODUCT] = product
        return df

    def save_to_db(self) -> None:
        if not self.dfs:
            return
        combined_df = pd.concat(self.dfs.values(), ignore_index=True)
        with self.db_lock:
            self.con.execute("CREATE OR REPLACE TABLE candles AS SELECT * FROM combined_df")

    def update_minute_candles(self, current_time_step: dt.datetime) -> None:
            cutoff = current_time_step - WINDOW
            for product in self.products:
                new_df = self.fetch_minute_candles(
                    product, current_time_step - dt.timedelta(minutes=5), current_time_step
                )
                existing_df = self.dfs.get(product, pd.DataFrame(columns=COLS + [PRODUCT]))
                stale_trimmed_df = existing_df.loc[existing_df[TIME] > cutoff]

                updated_df = (
                    pd.concat([stale_trimmed_df, new_df])
                    .drop_duplicates(TIME, keep="last")
                    .sort_values(by=[TIME])
                    .reset_index(drop=True)
                )
                self.dfs[product] = updated_df

            self.save_to_db()

    def initialize(self) -> None:
        current_time_step = dt.datetime.now(dt.UTC)
        for product in self.products:
            while True:
                try:
                    df = self.fetch_minute_candles(
                        product, current_time_step - WINDOW, current_time_step
                    )
                    df = (
                        df.drop_duplicates(TIME, keep="last")
                        .sort_values(by=[TIME])
                        .reset_index(drop=True)
                    )
                    self.dfs[product] = df
                    print(f"Initial 72h {product} loaded ({len(df)} rows)")
                    break
                except rq.RequestException as err:
                    print(f"Initial download failed for {product} ({err}). Retrying in 10s...")
                    tm.sleep(10)

        self.save_to_db()

    def run(self) -> None:
        self.initialize()
        while True:
            tm.sleep(60 - tm.time() % 60 + 2)
            current_time_step = dt.datetime.now(dt.UTC)
            try:
                self.update_minute_candles(current_time_step)
                print(f"Updated all products at {current_time_step.isoformat()}")
            except rq.RequestException as err:
                print("Fetch failed, will retry next minute:", err)
