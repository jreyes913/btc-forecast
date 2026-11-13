import datetime as dt

import duckdb as dd
import pandas as pd
import pytest
import requests as rq

from src.services.candles import (
    HEAD_REQUESTS_PER_SYNC,
    MAX_BARS_PER_REQUEST,
    CandleStore,
    CandleWorker,
    CoinbaseClient,
    LivePoller,
)

NOW = dt.datetime(2026, 11, 12, 12, 0, 30, tzinfo=dt.UTC)  # mid-way through the 12:00 bar
MIN = dt.timedelta(minutes=1)
STORE_WINDOW = dt.timedelta(days=5)
INITIAL = dt.timedelta(hours=1)


def bars(product, start, end):
    times = pd.date_range(start, end, freq="1min")
    return pd.DataFrame(
        {"time": times, "low": 1.0, "high": 2.0, "open": 1.5, "close": 1.5,
         "volume": 1.0, "product": product}
    )


class FakeClient:
    def __init__(self, fail=(), listed_since=None):
        self.calls = []
        self.fail = set(fail)
        self.listed_since = listed_since

    def fetch_minute_candles(self, product, start, end):
        self.calls.append((product, start, end))
        if product in self.fail:
            raise rq.ConnectionError("boom")
        if self.listed_since is not None:
            start = max(start, self.listed_since)
        if start > end:
            return bars(product, end, end).iloc[0:0]
        return bars(product, start, end)


@pytest.fixture
def store():
    return CandleStore(dd.connect(":memory:"))


def worker(store, client, products=("BTC-USD",)):
    return CandleWorker(
        store, client, products=list(products), store_window=STORE_WINDOW, initial_window=INITIAL
    )


def test_fresh_db_fetches_recent_bars_first_without_forming_bar(store):
    client = FakeClient()
    w = worker(store, client)
    w.sync_tail("BTC-USD", NOW)

    _, start, end = client.calls[0]
    assert end == dt.datetime(2026, 11, 12, 11, 59, tzinfo=dt.UTC)  # 12:00 is still forming
    assert start == dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC) - INITIAL
    assert len(store.read_candles("BTC-USD")) == 60


def test_head_backfills_in_bounded_chunks_until_window_is_full(store):
    client = FakeClient()
    w = worker(store, client)
    w.sync_tail("BTC-USD", NOW)
    span = HEAD_REQUESTS_PER_SYNC * MAX_BARS_PER_REQUEST * MIN

    w.sync_head("BTC-USD", NOW)
    _, start, end = client.calls[-1]
    assert end - start == span - MIN  # one bounded chunk per call

    for _ in range(20):
        w.sync_head("BTC-USD", NOW)
    first = store.first_time("BTC-USD")
    assert first == dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC) - STORE_WINDOW
    n_calls = len(client.calls)
    w.sync_head("BTC-USD", NOW)
    assert len(client.calls) == n_calls  # done: no more requests

    times = store.read_candles("BTC-USD")["time"]
    assert times.is_unique and times.diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_head_stops_when_no_older_data_exists(store):
    listed = dt.datetime(2026, 11, 12, 10, 0, tzinfo=dt.UTC)
    client = FakeClient(listed_since=listed)
    w = worker(store, client)
    store.upsert(bars("BTC-USD", listed, "2026-11-12 11:59Z"))
    w.sync_head("BTC-USD", NOW)
    n_calls = len(client.calls)
    w.sync_head("BTC-USD", NOW)
    assert len(client.calls) == n_calls


def test_up_to_date_products_make_no_request(store):
    store.upsert(bars("BTC-USD", "2026-11-12 11:00Z", "2026-11-12 11:59Z"))
    client = FakeClient()
    assert worker(store, client).sync_tail("BTC-USD", NOW) == 0
    assert client.calls == []


def test_tail_resumes_from_last_stored_bar_with_overlap(store):
    store.upsert(bars("BTC-USD", "2026-11-12 10:00Z", "2026-11-12 11:00Z"))
    client = FakeClient()
    worker(store, client).sync_tail("BTC-USD", NOW)
    _, start, _ = client.calls[0]
    assert start == dt.datetime(2026, 11, 12, 10, 58, tzinfo=dt.UTC)


def test_upsert_is_idempotent_and_replaces_revised_bars(store):
    df = bars("BTC-USD", "2026-11-12 11:00Z", "2026-11-12 11:04Z")
    store.upsert(df)
    store.upsert(df.assign(close=9.0))
    out = store.read_candles("BTC-USD")
    assert len(out) == 5 and out["close"].eq(9.0).all()


def test_old_rows_are_pruned(store):
    store.upsert(bars("BTC-USD", "2026-11-01 00:00Z", "2026-11-01 00:10Z"))
    worker(store, FakeClient()).tick(NOW)
    oldest = store.read_candles("BTC-USD")["time"].min()
    assert oldest >= pd.Timestamp(NOW).floor("min") - STORE_WINDOW


def test_one_product_failing_does_not_block_others(store):
    w = worker(store, FakeClient(fail={"ETH-USD"}), products=["ETH-USD", "BTC-USD"])
    fetched = w.tick(NOW)
    assert fetched["ETH-USD"] is None
    assert fetched["BTC-USD"] == 60


def test_read_panel_is_wide_with_field_product_columns(store):
    store.upsert(bars("ETH-USD", "2026-11-12 11:00Z", "2026-11-12 11:02Z"))
    store.upsert(bars("BTC-USD", "2026-11-12 11:01Z", "2026-11-12 11:02Z"))
    panel = store.read_panel(["BTC-USD", "ETH-USD"])
    assert list(panel["close"].columns) == ["BTC-USD", "ETH-USD"]
    assert set(panel.columns.get_level_values(0)) == {"close", "high", "low"}
    assert panel.index.is_monotonic_increasing and len(panel) == 3
    until = store.read_closes(["BTC-USD"], until=dt.datetime(2026, 11, 12, 11, 1, tzinfo=dt.UTC))
    assert list(until.index) == [pd.Timestamp("2026-11-12 11:01Z")]


def test_live_bar_is_kept_in_memory_only(store):
    client = FakeClient()
    poller = LivePoller(client)
    bar = poller.poll(NOW)
    assert bar["time"] == dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC)
    assert poller.bar == bar
    assert store.last_time("BTC-USD") is None


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise rq.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FlakySession:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def get(self, url, params, timeout):
        return FakeResponse(self.statuses.pop(0))


def test_client_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("src.services.candles.tm.sleep", lambda s: None)
    client = CoinbaseClient(session=FlakySession([429, 503, 200]))
    start = dt.datetime(2026, 11, 12, 11, 0, tzinfo=dt.UTC)
    assert client.fetch_minute_candles("BTC-USD", start, start).empty


def test_client_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("src.services.candles.tm.sleep", lambda s: None)
    client = CoinbaseClient(session=FlakySession([429] * 3), max_retries=2)
    start = dt.datetime(2026, 11, 12, 11, 0, tzinfo=dt.UTC)
    with pytest.raises(rq.HTTPError):
        client.fetch_minute_candles("BTC-USD", start, start)
