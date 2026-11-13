"""Coinbase 1-minute candle ingestion into a rolling DuckDB store.

DuckDB is the single source of truth for *closed* bars. Each sync:

1. fills the **tail**, from the last stored bar to the last closed minute, so restarts
   and outages backfill themselves; a fresh install gets the most recent bars first;
2. extends the **head** backwards towards ``STORE_WINDOW``, a few requests at a time,
   so a long history fills in without delaying the first forecast;
3. prunes anything older than ``STORE_WINDOW`` so the table stays bounded.

The bar that is still forming is never stored. ``LivePoller`` keeps it in memory for
the chart only.
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

from src.settings import (
    COINBASE_URL_TEMPLATE,
    FIT_WINDOW,
    LIVE_POLL_SECONDS,
    PRODUCTS,
    STORE_WINDOW,
    SYNC_EVERY_SECONDS,
    TARGET,
)

log = logging.getLogger(__name__)

COLS = ["time", "low", "high", "open", "close", "volume"]  # Coinbase response order
MAX_BARS_PER_REQUEST = 300
HEAD_REQUESTS_PER_SYNC = 12
REQUEST_SPACING_S = 0.2
RETRY_STATUSES = {429, 500, 502, 503, 504}
CHECKPOINT_EVERY = dt.timedelta(hours=1)
ONE_MIN = dt.timedelta(minutes=1)


def floor_minute(ts: dt.datetime) -> dt.datetime:
    return ts.replace(second=0, microsecond=0)


class CandleStore:
    """Thread-safe wrapper around one DuckDB connection holding the ``candles`` table."""

    def __init__(self, con: dd.DuckDBPyConnection):
        self.con = con
        self.lock = threading.RLock()
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

    def _scalar_time(self, agg: str, product: str) -> dt.datetime | None:
        with self.lock:
            # .df() rather than fetchall(): DuckDB's Python conversion of TIMESTAMPTZ
            # scalars needs pytz, while the pandas path does not.
            value = (
                self.con.execute(
                    f"SELECT {agg}(time) AS t FROM candles WHERE product = ?", [product]
                )
                .df()["t"]
                .iloc[0]
            )
        return None if pd.isna(value) else value.to_pydatetime()

    def last_time(self, product: str) -> dt.datetime | None:
        return self._scalar_time("max", product)

    def first_time(self, product: str) -> dt.datetime | None:
        return self._scalar_time("min", product)

    def read_candles(self, product: str, since: dt.datetime | None = None) -> pd.DataFrame:
        query = "SELECT time, open, high, low, close, volume FROM candles WHERE product = ?"
        params: list = [product]
        if since is not None:
            query += " AND time > ?"
            params.append(since)
        with self.lock:
            return self.con.execute(query + " ORDER BY time", params).df()

    def read_panel(
        self,
        products: Iterable[str],
        fields: Iterable[str] = ("close", "high", "low"),
        since: dt.datetime | None = None,
        until: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Wide frame indexed by time with (field, product) MultiIndex columns."""
        products, fields = list(products), list(fields)
        query = (
            f"SELECT time, product, {', '.join(fields)} FROM candles "
            "WHERE list_contains(?, product)"
        )
        params: list = [products]
        if since is not None:
            query += " AND time > ?"
            params.append(since)
        if until is not None:
            query += " AND time <= ?"
            params.append(until)
        with self.lock:
            df = self.con.execute(query, params).df()
        wide = df.pivot(index="time", columns="product", values=fields).sort_index()
        cols = pd.MultiIndex.from_product([fields, products])
        return wide.reindex(columns=cols)

    def read_closes(self, products: Iterable[str], **kwargs) -> pd.DataFrame:
        """Close prices as a wide frame: index = time (UTC), one column per product."""
        return self.read_panel(products, fields=("close",), **kwargs)["close"]


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
            chunk_end = min(chunk_start + (MAX_BARS_PER_REQUEST - 1) * ONE_MIN, end)
            params = {
                "granularity": 60,
                "start": chunk_start.isoformat(),
                "end": chunk_end.isoformat(),
            }
            rows += self._get(url, params)
            chunk_start = chunk_end + ONE_MIN
            tm.sleep(REQUEST_SPACING_S)

        df = pd.DataFrame(rows, columns=COLS)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["product"] = product
        df = df.loc[(df["time"] >= start) & (df["time"] <= end)]
        return df.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


class CandleWorker:
    """Keeps ``store`` in sync with Coinbase and calls ``on_update`` when bars arrive."""

    def __init__(
        self,
        store: CandleStore,
        client: CoinbaseClient | None = None,
        products: Iterable[str] = PRODUCTS,
        on_update: Callable[[], None] | None = None,
        store_window: dt.timedelta = STORE_WINDOW,
        initial_window: dt.timedelta = FIT_WINDOW,
    ):
        self.store = store
        self.client = client or CoinbaseClient()
        self.products = list(products)
        self.on_update = on_update
        self.store_window = store_window
        self.initial_window = initial_window
        self._head_done: set[str] = set()
        self._last_checkpoint = dt.datetime.now(dt.UTC)

    def sync_tail(self, product: str, now: dt.datetime) -> int:
        """Fetch closed bars after the last stored one. Returns the number fetched."""
        current_minute = floor_minute(now)
        last_closed = current_minute - ONE_MIN  # the current bar is still forming
        last = self.store.last_time(product)
        if last is not None and last >= last_closed:
            return 0  # up to date: no request
        if last is None:
            start = current_minute - self.initial_window  # newest data first on a fresh DB
        else:
            # Re-fetch two bars of overlap: Coinbase occasionally revises the latest bar.
            start = max(last - 2 * ONE_MIN, current_minute - self.store_window)
        df = self.client.fetch_minute_candles(product, start, last_closed)
        self.store.upsert(df)
        return len(df)

    def sync_head(self, product: str, now: dt.datetime) -> int:
        """Extend history backwards towards ``store_window``, a bounded chunk per call."""
        if product in self._head_done:
            return 0
        first = self.store.first_time(product)
        window_start = floor_minute(now) - self.store_window
        if first is None:
            return 0
        if first <= window_start + ONE_MIN:
            self._head_done.add(product)
            return 0
        end = first - ONE_MIN
        start = max(
            window_start, end - (HEAD_REQUESTS_PER_SYNC * MAX_BARS_PER_REQUEST - 1) * ONE_MIN
        )
        df = self.client.fetch_minute_candles(product, start, end)
        if df.empty:
            # Nothing older exists (e.g. a recent listing); stop asking.
            self._head_done.add(product)
            return 0
        self.store.upsert(df)
        return len(df)

    def tick(self, now: dt.datetime | None = None) -> dict[str, int | None]:
        """Sync every product independently; a failure in one doesn't skip the others.

        Returns the number of *new tail* bars per product (None on failure).
        """
        now = now or dt.datetime.now(dt.UTC)
        fetched: dict[str, int | None] = {}
        for product in self.products:
            try:
                fetched[product] = self.sync_tail(product, now)
            except (rq.RequestException, ValueError) as err:
                log.warning("Sync failed for %s: %s", product, err)
                fetched[product] = None
        for product in self.products:
            try:
                self.sync_head(product, now)
            except (rq.RequestException, ValueError) as err:
                log.warning("History backfill failed for %s: %s", product, err)

        self.store.prune(floor_minute(now) - self.store_window)
        if now - self._last_checkpoint >= CHECKPOINT_EVERY:
            self.store.checkpoint()
            self._last_checkpoint = now
        return fetched

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            fetched = self.tick()
            if any(fetched.values()):
                log.info("New candles: %s", fetched)
                if self.on_update is not None:
                    try:
                        self.on_update()
                    except Exception:  # keep ingesting even if the model blows up
                        log.exception("on_update callback failed")
            stop.wait(SYNC_EVERY_SECONDS)


class LivePoller:
    """Polls the bar that is still forming, for display only. It is never stored."""

    def __init__(self, client: CoinbaseClient | None = None, product: str = TARGET):
        self.client = client or CoinbaseClient(max_retries=1)
        self.product = product
        self._lock = threading.Lock()
        self._bar: dict | None = None

    def poll(self, now: dt.datetime | None = None) -> dict | None:
        now = now or dt.datetime.now(dt.UTC)
        current_minute = floor_minute(now)
        df = self.client.fetch_minute_candles(
            self.product, current_minute - ONE_MIN, current_minute
        )
        forming = df.loc[df["time"] == current_minute]
        bar = None
        if not forming.empty:
            row = forming.iloc[-1]
            bar = {k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}
            bar["time"] = row["time"].to_pydatetime()
        with self._lock:
            self._bar = bar
        return bar

    @property
    def bar(self) -> dict | None:
        with self._lock:
            return self._bar

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.poll()
            except rq.RequestException as err:
                log.debug("Live poll failed: %s", err)
            stop.wait(LIVE_POLL_SECONDS)
