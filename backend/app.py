import datetime as dt
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb as dd
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.models.api import CandleSchema, ForecastResponse, LadderResponse, LiveResponse
from src.services.candles import CandleStore, CandleWorker, LivePoller
from src.services.evaluation import Replayer, baseline_records, factor_records, window_label
from src.services.forecast_cache import ForecastCache
from src.services.ladder import build_ladder
from src.services.quant import ForecastResult, InsufficientDataError
from src.services.scoring import REFERENCE_MODEL, ForecastLog, summarize
from src.settings import (
    CHART_WINDOW,
    DATA_DIR,
    DUCKDB_PATH,
    FACTOR_PRODUCTS,
    FIT_WINDOW,
    FORECAST_HORIZON_DEFAULT,
    FORECAST_HORIZON_MAX,
    FORECAST_LOG_RETENTION,
    LADDER_EDGE_POINTS,
    LADDER_STRIKE_STEP,
    LOG_EVERY_MINUTES,
    PRODUCTS,
    TARGET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("btc-forecast")

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
SUMMARY_EVERY = dt.timedelta(minutes=5)
SOURCES = ("all", "live", "replay")


def forecast_to_response(res: ForecastResult) -> dict:
    step = dt.timedelta(minutes=1)
    points = [
        {
            "time": res.last_time,
            "mean": res.last_close,
            **{k: res.last_close for k in res.price_quantiles},
        }
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
            "last_close": res.last_close,
            "resid_converged": res.resid_converged,
            "factor_converged": res.factor_converged,
            "seasonality_active": res.seasonality_active,
            "fit_window_hours": res.fit_window_hours,
            "cone_model": res.cone_model,
        },
        "historical_volatility": [
            {
                "time": t,
                "total_bps": float(tot),
                "idiosyncratic_bps": float(idio),
                "typical_bps": float(typ),
            }
            for t, tot, idio, typ in zip(
                hist.index, hist["total"], hist["idiosyncratic"], hist["typical"], strict=True
            )
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
        app.state.forecast_log = ForecastLog(app.state.store)
        app.state.cache = ForecastCache(app.state.store, TARGET, FACTOR_PRODUCTS)
        app.state.replayer = Replayer(app.state.store, app.state.forecast_log)

        stop = threading.Event()
        if start_worker:
            worker = CandleWorker(app.state.store, on_update=lambda: on_new_bars(app))
            threads = [
                (worker.run, "candle-worker"),
                (app.state.live.run, "live-poller"),
                (app.state.replayer.run, "replay"),
                (lambda s: summary_loop(app, s), "summary"),
            ]
            for target, name in threads:
                threading.Thread(target=target, args=(stop,), name=name, daemon=True).start()
        yield
        stop.set()
        if con is not None:
            con.close()

    app = FastAPI(title="BTC forecast", lifespan=lifespan)
    app.state.store = store
    app.state.live = LivePoller()
    app.state.backtest = None
    app.state.last_logged_origin = None

    @app.get("/candles", response_model=list[CandleSchema])
    def get_candles(
        product: str = Query(default=TARGET),
        since: dt.datetime | None = Query(default=None),
    ):
        if product not in PRODUCTS:
            raise HTTPException(status_code=404, detail=f"Unknown product {product!r}")
        if since is None:
            last = app.state.store.last_time(product)
            since = last - CHART_WINDOW if last is not None else None
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

    @app.get("/live", response_model=LiveResponse)
    def get_live():
        latest = app.state.cache.latest(FORECAST_HORIZON_DEFAULT)
        return {
            "bar": app.state.live.bar,
            "last_closed_time": app.state.store.last_time(TARGET),
            "cone_origin_time": latest.last_time if latest else None,
            "cone_origin_price": latest.last_close if latest else None,
        }

    @app.get("/ladder", response_model=LadderResponse)
    def get_ladder(which: str = Query(default="next", pattern="^(next|following)$")):
        try:
            cone = app.state.cache.get(FORECAST_HORIZON_DEFAULT)
        except (LookupError, InsufficientDataError) as err:
            raise HTTPException(status_code=503, detail=f"Model not ready: {err}") from err
        bar = app.state.live.bar
        # Only anchor on the live bar if it's newer than the cone's origin.
        live_price = bar["close"] if bar and bar["time"] > cone.last_time else None
        return build_ladder(
            cone,
            dt.datetime.now(dt.UTC),
            live_price,
            which,
            step=LADDER_STRIKE_STEP,
            edge=LADDER_EDGE_POINTS,
        )

    @app.get("/backtest")
    def get_backtest(source: str = Query(default="all", pattern="^(all|live|replay)$")):
        if app.state.backtest is None:
            raise HTTPException(status_code=503, detail="Scoring summary not computed yet.")
        summary = app.state.backtest["by_source"][source]
        return {
            "computed_at": app.state.backtest["computed_at"],
            "source": source,
            "replay_progress": app.state.replayer.progress,
            **summary,
        }

    # Serve the built React frontend at the root URL (must be AFTER all API routes)
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


def on_new_bars(app: FastAPI) -> None:
    """Refit for the newest bar, and log a live forecast every LOG_EVERY_MINUTES."""
    try:
        app.state.cache.get(FORECAST_HORIZON_DEFAULT)  # warm the dashboard cone
        # Always score the pure factor model, even when the cone is the ensemble.
        res = app.state.cache.get_factor(FORECAST_HORIZON_DEFAULT)
    except (LookupError, InsufficientDataError) as err:
        log.info("Forecast not ready yet: %s", err)
        return
    origin = res.last_time
    if origin.minute % LOG_EVERY_MINUTES or origin == app.state.last_logged_origin:
        return
    closes = app.state.store.read_closes(
        [TARGET], since=(origin - FIT_WINDOW - pd.Timedelta(minutes=5)).to_pydatetime()
    )
    records = factor_records(res, "live") + baseline_records(closes, origin, res.last_close, "live")
    app.state.forecast_log.insert(records)
    app.state.last_logged_origin = origin


def compute_summary(app: FastAPI) -> dict:
    flog: ForecastLog = app.state.forecast_log
    flog.resolve()
    df = flog.read_resolved()
    by_source = {}
    for source in SOURCES:
        sub = df if source == "all" else df.loc[df["source"] == source]
        if source == "all":
            # Replay-only window variants would otherwise mix into the live comparison.
            sub = sub.loc[~sub["model"].str.contains("@")] if not sub.empty else sub
        summary = summarize(sub)
        rec = summary["recommended_fit_window"]
        if rec == REFERENCE_MODEL:
            summary["recommended_fit_window"] = window_label(FIT_WINDOW)
        elif rec is not None:
            summary["recommended_fit_window"] = rec.split("@", 1)[1]
        by_source[source] = summary
    return {"computed_at": dt.datetime.now(dt.UTC), "by_source": by_source}


def summary_loop(app: FastAPI, stop: threading.Event) -> None:
    stop.wait(60)
    while not stop.is_set():
        try:
            app.state.forecast_log.prune(dt.datetime.now(dt.UTC) - FORECAST_LOG_RETENTION)
            app.state.backtest = compute_summary(app)
        except Exception:
            log.exception("Scoring summary failed")
        stop.wait(SUMMARY_EVERY.total_seconds())


app = create_app()
