import datetime as dt
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb as dd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from src.models.api import CandleSchema
from src.services.candles import Candles
from src.services.quant import BtcFactorGarchModel
from src.settings import DUCKDB_PATH

con = dd.connect(DUCKDB_PATH)
candles_worker = Candles(con=con)
quant_model = BtcFactorGarchModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=candles_worker.run, daemon=True)
    thread.start()
    yield
    con.close()


app = FastAPI(lifespan=lifespan)


@app.get("/candles", response_model=list[CandleSchema])
def get_candles(product: str = Query(default="BTC-USD")):
    with candles_worker.db_lock:
        tables = con.execute("SHOW TABLES").fetchall()
        if ("candles",) not in tables:
            raise HTTPException(
                status_code=503,
                detail="Initial candle data is still loading. Try again shortly.",
            )

        df = con.execute(
            "SELECT time, low, high, open, close, volume FROM candles WHERE product = ? ORDER BY time ASC",
            [product],
        ).df()
        return df.to_dict(orient="records")


@app.get("/forecast")
def get_forecast(horizon: int = Query(default=60, ge=1, le=240)):
    with candles_worker.db_lock:
        tables = con.execute("SHOW TABLES").fetchall()
        if ("candles",) not in tables:
            raise HTTPException(status_code=503, detail="Candle data still loading.")

        btc_df = con.execute("SELECT time, close FROM candles WHERE product = 'BTC-USD'").df()
        eth_df = con.execute("SELECT time, close FROM candles WHERE product = 'ETH-USD'").df()
        sp500_df = con.execute("SELECT time, close FROM candles WHERE product = 'SPX-USD'").df()

    if btc_df.empty or eth_df.empty or sp500_df.empty:
        raise HTTPException(status_code=503, detail="Waiting for all three product series.")

    result = quant_model.fit_and_forecast(btc_df, eth_df, sp500_df, horizon=horizon)

    last_row = btc_df.sort_values("time").iloc[-1]
    last_time = last_row["time"]
    last_close = float(last_row["close"])

    paths = quant_model.project_price_path(last_close=last_close, result=result)
    future_times = [
        (last_time + dt.timedelta(minutes=i + 1)).isoformat() for i in range(horizon)
    ]

    forecast_points = [
        {
            "time": last_time.isoformat(),
            "expected_price": last_close,
            "upper_1sd": last_close,
            "lower_1sd": last_close,
        }
    ] + [
        {
            "time": future_times[i],
            "expected_price": paths["expected_price"][i],
            "upper_1sd": paths["upper_1sd"][i],
            "lower_1sd": paths["lower_1sd"][i],
        }
        for i in range(horizon)
    ]

    return {
        "metrics": {
            "alpha": result.alpha,
            "beta": result.beta,
            "pca_explained_variance_ratio": result.pca_explained_variance_ratio,
            "pca_loadings": result.pca_loadings,
            "residual_std_error": result.residual_std_error,
            "garch_params": result.garch_params,
        },
        "historical_volatility": result.historical_volatility,
        "forecast": forecast_points,
    }


# Serve the built React frontend at the root URL (must be AFTER all API routes)
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
