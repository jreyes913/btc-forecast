import datetime as dt

import duckdb as dd
import pandas as pd
import pytest
import requests as rq

from src.services.candles import CandleStore, CandleWorker, CoinbaseClient
from src.settings import WINDOW

NOW = dt.datetime(2026, 11, 12, 12, 0, 30, tzinfo=dt.UTC)  # mid-way through the 12:00 bar


def bars(product, start, end):
    times = pd.date_range(start, end, freq="1min")
    return pd.DataFrame(
        {"time": times, "low": 1.0, "high": 2.0, "open": 1.5, "close": 1.5,
         "volume": 1.0, "product": product}
    )


class FakeClient:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def fetch_minute_candles(self, product, start, end):
        self.calls.append((product, start, end))
        if product in self.fail:
            raise rq.ConnectionError("boom")
        return bars(product, start, end)


@pytest.fixture
def store():
    return CandleStore(dd.connect(":memory:"))


def test_initial_sync_fetches_full_window_without_forming_bar(store):
    client = FakeClient()
    CandleWorker(store, client, products=["BTC-USD"]).tick(NOW)

    _, start, end = client.calls[0]
    assert end == dt.datetime(2026, 11, 12, 11, 59, tzinfo=dt.UTC)  # 12:00 is still forming
    assert start == dt.datetime(2026, 11, 12, 12, 0, tzinfo=dt.UTC) - WINDOW
    assert len(store.read_candles("BTC-USD")) == 72 * 60


def test_sync_resumes_from_last_stored_bar(store):
    store.upsert(bars("BTC-USD", "2026-11-12 10:00Z", "2026-11-12 11:00Z"))
    client = FakeClient()
    CandleWorker(store, client, products=["BTC-USD"]).tick(NOW)

    _, start, _ = client.calls[0]
    assert start == dt.datetime(2026, 11, 12, 10, 58, tzinfo=dt.UTC)  # 2-minute overlap
    candles = store.read_candles("BTC-USD")
    assert candles["time"].is_unique
    assert candles["time"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_upsert_is_idempotent_and_replaces_revised_bars(store):
    df = bars("BTC-USD", "2026-11-12 11:00Z", "2026-11-12 11:04Z")
    store.upsert(df)
    revised = df.assign(close=9.0)
    store.upsert(revised)
    out = store.read_candles("BTC-USD")
    assert len(out) == 5 and out["close"].eq(9.0).all()


def test_old_rows_are_pruned(store):
    store.upsert(bars("BTC-USD", "2026-11-01 00:00Z", "2026-11-01 00:10Z"))
    CandleWorker(store, FakeClient(), products=["BTC-USD"]).tick(NOW)
    oldest = store.read_candles("BTC-USD")["time"].min()
    assert oldest >= pd.Timestamp(NOW).floor("min") - WINDOW


def test_one_product_failing_does_not_block_others(store):
    worker = CandleWorker(store, FakeClient(fail={"ETH-USD"}), products=["ETH-USD", "BTC-USD"])
    fetched = worker.tick(NOW)
    assert fetched["ETH-USD"] is None
    assert fetched["BTC-USD"] == 72 * 60


def test_read_closes_is_wide_and_ordered(store):
    store.upsert(bars("ETH-USD", "2026-11-12 11:00Z", "2026-11-12 11:02Z"))
    store.upsert(bars("BTC-USD", "2026-11-12 11:01Z", "2026-11-12 11:02Z"))
    wide = store.read_closes(["BTC-USD", "ETH-USD"])
    assert list(wide.columns) == ["BTC-USD", "ETH-USD"]
    assert wide.index.is_monotonic_increasing and len(wide) == 3


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
