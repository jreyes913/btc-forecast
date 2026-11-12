import duckdb as dd
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import create_app
from src.services.candles import CandleStore
from src.settings import PRODUCTS


@pytest.fixture
def client(synthetic_closes):
    store = CandleStore(dd.connect(":memory:"))
    for product in PRODUCTS:
        s = synthetic_closes[product]
        store.upsert(pd.DataFrame({
            "product": product, "time": s.index, "open": s.values, "high": s.values,
            "low": s.values, "close": s.values, "volume": 1.0,
        }))
    with TestClient(create_app(store=store, start_worker=False)) as c:
        yield c


def test_candles_and_since(client):
    all_rows = client.get("/candles").json()
    assert len(all_rows) == 72 * 60
    newer = client.get("/candles", params={"since": all_rows[-3]["time"]}).json()
    assert len(newer) == 2


def test_unknown_product_404(client):
    assert client.get("/candles", params={"product": "SPX-USD"}).status_code == 404


def test_forecast_shape(client):
    body = client.get("/forecast", params={"horizon": 15}).json()
    assert len(body["forecast"]) == 16  # anchor point + 15 steps
    assert set(body["metrics"]["pca_loadings"]) == set(PRODUCTS[1:])
    assert body["diagnostics"]["resid_converged"]
    last = body["forecast"][-1]
    assert last["p025"] < last["p16"] < last["p50"] < last["p84"] < last["p975"]


def test_forecast_is_cached(client, monkeypatch):
    first = client.get("/forecast", params={"horizon": 5}).json()
    cache = client.app.state.cache
    monkeypatch.setattr(cache, "_fit", lambda *a, **k: pytest.fail("refit on unchanged data"))
    assert client.get("/forecast", params={"horizon": 5}).json() == first


def test_forecast_503_when_empty():
    store = CandleStore(dd.connect(":memory:"))
    with TestClient(create_app(store=store, start_worker=False)) as c:
        assert c.get("/forecast").status_code == 503
        assert c.get("/backtest").status_code == 503
