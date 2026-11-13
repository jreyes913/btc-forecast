import duckdb as dd
import numpy as np
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
    monkeypatch.setattr(
        "src.services.forecast_cache.fit_and_forecast",
        lambda *a, **k: pytest.fail("refit on unchanged data"),
    )
    assert client.get("/forecast", params={"horizon": 5}).json() == first


def test_forecast_503_when_empty():
    store = CandleStore(dd.connect(":memory:"))
    with TestClient(create_app(store=store, start_worker=False)) as c:
        assert c.get("/forecast").status_code == 503
        assert c.get("/backtest").status_code == 503


def test_live_reports_cone_origin(client):
    body = client.get("/live").json()
    assert body["bar"] is None and body["cone_origin_time"] is None
    client.get("/forecast")
    body = client.get("/live").json()
    assert body["cone_origin_time"] == body["last_closed_time"]
    assert body["cone_origin_price"] > 0


def test_backtest_summary_by_source(client):
    import numpy as np
    from scipy.stats import norm

    from app import compute_summary
    from src.services.quant import Q_GRID
    from src.services.scoring import ForecastRecord

    app = client.app
    origins = pd.date_range("2026-11-01 01:00", periods=40, freq="30min", tz="UTC")
    closes = app.state.store.read_closes(["BTC-USD"])["BTC-USD"]
    records = [
        ForecastRecord(t, model, 15, "live", float(closes.loc[t]), 400.0, norm.ppf(Q_GRID) * 20)
        for t in origins
        for model in ("factor", "ewma")
    ]
    app.state.forecast_log.insert(records)
    app.state.backtest = compute_summary(app)

    body = client.get("/backtest", params={"source": "live"}).json()
    rows = {r["model"]: r for r in body["rows"]}
    assert rows["factor"]["n"] == 40 and rows["factor"]["brier"] is not None
    assert "dm_p_value" in rows["ewma"]
    assert body["calibration"] and "replay_progress" in body
    assert client.get("/backtest", params={"source": "replay"}).json()["rows"] == []
    assert client.get("/backtest", params={"source": "bogus"}).status_code == 422
    assert np.isfinite(rows["factor"]["qlike"])


def test_ensemble_cone_switch(client):
    from src.services.forecast_cache import ForecastCache
    from src.settings import FACTOR_PRODUCTS, TARGET

    app = client.app
    app.state.cache = ForecastCache(app.state.store, TARGET, FACTOR_PRODUCTS, cone_model="ensemble")
    body = client.get("/forecast", params={"horizon": 30}).json()
    assert body["diagnostics"]["cone_model"] == "ensemble"
    for p in body["forecast"][1:]:
        assert p["p025"] < p["p16"] < p["p50"] < p["p84"] < p["p975"]

    # The factor result used for scoring stays the pure factor model
    factor = app.state.cache.get_factor(30)
    cone = app.state.cache.get(30)
    assert factor.cone_model == "factor" and cone.cone_model == "ensemble"
    assert not np.allclose(factor.cum_return_var, cone.cum_return_var)


def test_invalid_cone_model_rejected(client):
    from src.services.forecast_cache import ForecastCache

    with pytest.raises(ValueError):
        ForecastCache(client.app.state.store, "BTC-USD", [], cone_model="bogus")
