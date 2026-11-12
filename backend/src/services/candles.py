"""Coinbase 1-minute candle ingestion into a rolling DuckDB window.

DuckDB is the single source of truth. Each tick fetches only the bars missing since the
last stored bar for each product (so restarts and outages backfill themselves), upserts
them, and prunes anything older than ``WINDOW`` so the table stays bounded.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time as tm
from collections.abc import Callable, Iterable

import duckdb as dd
import pandas as pd
import requests as rq

from src.settings import COINBASE_URL_TEMPLATE, PRODUCTS, WINDOW

log = logging.getLogger(__name__)

COLS = ["time", "low", "high", "open", "close", "volume"]  # Coinbase response order
MAX_BARS_PER_REQUEST = 300
REQUEST_SPACING_S = 0.2
RETRY_STATUSES = {429, 500, 502, 503, 504}
CHECKPOINT_EVERY = dt.timedelta(hours=1)


def floor_minute(ts: dt.datetime) -> dt.datetime:
    return ts.replace(second=0, microsecond=0)


class CandleStore:
    """Thread-safe wrapper around one DuckDB connection holding the ``candles`` table."""

    def __init__(self, con: dd.DuckDBPyConnection):
        self.con = con
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute("SET TimeZone = 'UTC'")
            self.con.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    product VARCHAR NOT NULL,
                    time TIMESTAMPTZ NOT NULL,
                    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                    PRIMARY KEY (product, time)
                )
                """
            )

    def upsert(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = df[["product", "time", "open", "high", "low", "close", "volume"]]
        with self.lock:
            self.con.register("new_rows", rows)
            try:
                self.con.execute("INSERT OR REPLACE INTO candles SELECT * FROM new_rows")
            finally:
                self.con.unregister("new_rows")

    def prune(self, cutoff: dt.datetime) -> None:
        with self.lock:
            self.con.execute("DELETE FROM candles WHERE time < ?", [cutoff])

    def checkpoint(self) -> None:
        with self.lock:
            self.con.execute("CHECKPOINT")

    def last_time(self, product: str) -> dt.datetime | None:
        with self.lock:
            # .df() rather than fetchall(): DuckDB's Python conversion of TIMESTAMPTZ
            # scalars needs pytz, while the pandas path does not.
            last = self.con.execute(
                "SELECT max(time) AS t FROM candles WHERE product = ?", [product]
            ).df()["t"].iloc[0]
        return None if pd.isna(last) else last.to_pydatetime()

    def read_candles(self, product: str, since: dt.datetime | None = None) -> pd.DataFrame:
        query = "SELECT time, open, high, low, close, volume FROM candles WHERE product = ?"
        params: list = [product]
        if since is not None:
            query += " AND time > ?"
            params.append(since)
        with self.lock:
            return self.con.execute(query + " ORDER BY time", params).df()

    def read_closes(self, products: Iterable[str]) -> pd.DataFrame:
        """Close prices as a wide frame: index = time (UTC), one column per product."""
        products = list(products)
        with self.lock:
            df = self.con.execute(
                "SELECT time, product, close FROM candles WHERE list_contains(?, product)",
                [products],
            ).df()
        wide = df.pivot(index="time", columns="product", values="close").sort_index()
        return wide.reindex(columns=products)


class CoinbaseClient:
    def __init__(self, session: rq.Session | None = None, max_retries: int = 4):
        self.session = session or rq.Session()
        self.max_retries = max_retries

    def _get(self, url: str, params: dict) -> list:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code not in RETRY_STATUSES:
                    resp.raise_for_status()
                    return resp.json()
                err: Exception = rq.HTTPError(f"HTTP {resp.status_code}", response=resp)
            except (rq.ConnectionError, rq.Timeout) as exc:
                err = exc
            if attempt == self.max_retries:
                raise err
            tm.sleep(2**attempt)
        raise AssertionError("unreachable")

    def fetch_minute_candles(
        self, product: str, start: dt.datetime, end: dt.datetime
    ) -> pd.DataFrame:
        """Bars with ``start <= time <= end``, in chunks of at most 300 bars per request."""
        url = COINBASE_URL_TEMPLATE.format(PRODUCT=product)
        rows: list = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + dt.timedelta(minutes=MAX_BARS_PER_REQUEST - 1), end)
            params = {
                "granularity": 60,
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
            }
            rows += self._get(url, params)
            chunk_start = chunk_end + dt.timedelta(minutes=1)
            tm.sleep(REQUEST_SPACING_S)

        df = pd.DataFrame(rows, columns=COLS)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["product"] = product
        df = df.loc[(df["time"] >= start) & (df["time"] <= end)]
        return df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


class CandleWorker:
    """Keeps ``store`` in sync with Coinbase and calls ``on_update`` after each tick."""

    def __init__(
        self,
        store: CandleStore,
        client: CoinbaseClient | None = None,
        products: Iterable[str] = PRODUCTS,
        on_update: Callable[[], None] | None = None,
    ):
        self.store = store
        self.client = client or CoinbaseClient()
        self.products = list(products)
        self.on_update = on_update
        self._last_checkpoint = dt.datetime.now(dt.UTC)

    def sync_product(self, product: str, now: dt.datetime) -> int:
        """Fetch every closed bar missing for ``product``; returns the number fetched."""
        current_minute = floor_minute(now)
        last_closed = current_minute - dt.timedelta(minutes=1)  # current bar is still forming
        window_start = current_minute - WINDOW

        last = self.store.last_time(product)
        # Re-fetch a couple of bars of overlap: Coinbase occasionally revises the latest bar.
        start = window_start if last is None else max(last - dt.timedelta(minutes=2), window_start)
        if start > last_closed:
            return 0

        df = self.client.fetch_minute_candles(product, start, last_closed)
        self.store.upsert(df)
        return len(df)

    def tick(self, now: dt.datetime | None = None) -> dict[str, int | None]:
        """Sync every product independently; a failure in one doesn't skip the others."""
        now = now or dt.datetime.now(dt.UTC)
        fetched: dict[str, int | None] = {}
        for product in self.products:
            try:
                fetched[product] = self.sync_product(product, now)
            except (rq.RequestException, ValueError) as err:
                log.warning("Sync failed for %s: %s", product, err)
                fetched[product] = None

        self.store.prune(floor_minute(now) - WINDOW)
        if now - self._last_checkpoint >= CHECKPOINT_EVERY:
            self.store.checkpoint()
            self._last_checkpoint = now
        return fetched

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            fetched = self.tick()
            log.info("Synced candles: %s", fetched)
            if self.on_update is not None:
                try:
                    self.on_update()
                except Exception:  # keep ingesting even if the model blows up
                    log.exception("on_update callback failed")
            # Wake a couple of seconds after each minute boundary so the bar has closed.
            stop.wait(60 - tm.time() % 60 + 2)
