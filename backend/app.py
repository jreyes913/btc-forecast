import dataclasses as dc
import datetime as dt
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb as dd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.models.api import CandleSchema, ForecastResponse
from src.services.backtest import run_backtest
from src.services.candles import CandleStore, CandleWorker
from src.services.forecast_cache import ForecastCache
from src.services.quant import ForecastResult, InsufficientDataError
from src.settings import (
    DATA_DIR,
    DUCKDB_PATH,
    FACTOR_PRODUCTS,
    FORECAST_HORIZON_DEFAULT,
    FORECAST_HORIZON_MAX,
    PRODUCTS,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("btc-forecast")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
BACKTEST_EVERY = dt.timedelta(hours=1)


def forecast_to_response(res: ForecastResult) -> dict:
    step = dt.timedelta(minutes=1)
    points = [
        {"time": res.last_time, "mean": res.last_close,
         **{k: res.last_close for k in res.price_quantiles}}
    ] + [
        {
            "time": res.last_time + (i + 1) * step,
            "mean": float(res.price_mean[i]),
            **{k: float(v[i]) for k, v in res.price_quantiles.items()},
        }
        for i in range(res.horizon)
    ]
    hist = res.historical_volatility
    return {
        "metrics": {
            "alpha_bps": res.alpha_bps,
            "beta": res.beta,
            "capm_r2": res.capm_r2,
            "pca_explained_variance_ratio": res.pca_explained_variance_ratio,
            "pca_loadings": res.pca_loadings,
            "resid_std_bps": res.resid_std_bps,
            "factor_variance_share": res.factor_variance_share,
            "resid_garch_params": res.resid_garch_params,
            "factor_garch_params": res.factor_garch_params,
        },
        "diagnostics": {
            "n_obs": res.n_obs,
            "last_bar_time": res.last_time,
            "resid_converged": res.resid_converged,
            "factor_converged": res.factor_converged,
        },
        "historical_volatility": [
            {"time": t, "total_bps": float(tot), "idiosyncratic_bps": float(idio)}
            for t, tot, idio in zip(hist.index, hist["total"], hist["idiosyncratic"], strict=True)
        ],
        "forecast": points,
    }


def create_app(store: CandleStore | None = None, start_worker: bool = True) -> FastAPI:
    """``store`` and ``start_worker`` exist so tests can run against an in-memory DB."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        con = None
        if app.state.store is None:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            con = dd.connect(DUCKDB_PATH)
            app.state.store = CandleStore(con)
        app.state.cache = ForecastCache(
            lambda: app.state.store.read_closes(PRODUCTS), TARGET, FACTOR_PRODUCTS
        )

        stop = threading.Event()
        if start_worker:
            def warm_cache():
                try:
                    app.state.cache.get(FORECAST_HORIZON_DEFAULT)
                except (LookupError, InsufficientDataError) as err:
                    log.info("Forecast not ready yet: %s", err)

            worker = CandleWorker(app.state.store, on_update=warm_cache)
            threading.Thread(target=worker.run, args=(stop,), daemon=True).start()
            threading.Thread(target=backtest_loop, args=(app, stop), daemon=True).start()
        yield
        stop.set()
        if con is not None:
            con.close()

    def backtest_loop(app: FastAPI, stop: threading.Event) -> None:
        stop.wait(120)  # let ingestion catch up first
        while not stop.is_set():
            try:
                closes = app.state.store.read_closes(PRODUCTS)
                if len(closes) > 49 * 60:
                    rows = run_backtest(closes, TARGET, FACTOR_PRODUCTS)
                    app.state.backtest = {
                        "computed_at": dt.datetime.now(dt.UTC),
                        "rows": [dc.asdict(r) for r in rows],
                    }
                    log.info("Backtest refreshed (%d rows)", len(rows))
            except Exception:
                log.exception("Backtest failed")
            stop.wait(BACKTEST_EVERY.total_seconds())

    app = FastAPI(title="BTC forecast", lifespan=lifespan)
    app.state.store = store
    app.state.backtest = None

    @app.get("/candles", response_model=list[CandleSchema])
    def get_candles(
        product: str = Query(default=TARGET),
        since: dt.datetime | None = Query(default=None),
    ):
        if product not in PRODUCTS:
            raise HTTPException(status_code=404, detail=f"Unknown product {product!r}")
        return app.state.store.read_candles(product, since).to_dict(orient="records")

    @app.get("/forecast", response_model=ForecastResponse)
    def get_forecast(
        horizon: int = Query(default=FORECAST_HORIZON_DEFAULT, ge=1, le=FORECAST_HORIZON_MAX),
    ):
        try:
            res = app.state.cache.get(horizon)
        except (LookupError, InsufficientDataError) as err:
            raise HTTPException(status_code=503, detail=f"Model not ready: {err}") from err
        return forecast_to_response(res)

    @app.get("/backtest")
    def get_backtest():
        if app.state.backtest is None:
            raise HTTPException(status_code=503, detail="Backtest needs ~49h of data; not run yet.")
        return app.state.backtest

    # Serve the built React frontend at the root URL (must be AFTER all API routes)
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


app = create_app()
