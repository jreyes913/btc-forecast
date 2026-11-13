"""Builds scored-forecast records, live and replay, and runs the historical replay."""

from __future__ import annotations

import datetime as dt
import logging
import threading

import numpy as np
import pandas as pd

from src.services.baselines import ewma_forecast, gjr_forecast
from src.services.candles import CandleStore
from src.services.quant import (
    ForecastResult,
    InsufficientDataError,
    build_return_panel,
    fit_and_forecast,
)
from src.services.scoring import REFERENCE_MODEL, ForecastLog, ForecastRecord
from src.services.seasonality import SeasonalProfile, estimate_profile
from src.settings import (
    FACTOR_PRODUCTS,
    FIT_WINDOW,
    FIT_WINDOW_CANDIDATES,
    LOG_HORIZONS,
    PRODUCTS,
    REPLAY_STEP_MINUTES,
    SEASON_BUCKET_MINUTES,
    STORE_WINDOW,
    TARGET,
)

log = logging.getLogger(__name__)

REPLAY_SIMS = 500
REPLAY_BATCH = 4


def window_label(window: dt.timedelta) -> str:
    hours = window / dt.timedelta(hours=1)
    return (
        f"{int(hours // 24)}d"
        if hours >= 24 and hours % 24 == 0 and hours > 72
        else f"{int(hours)}h"
    )


def model_windows(production: dt.timedelta = FIT_WINDOW) -> dict[str, dt.timedelta]:
    """The production window is logged as ``factor``; other candidates as ``factor@<w>``."""
    out = {REFERENCE_MODEL: production}
    for w in FIT_WINDOW_CANDIDATES:
        if w != production:
            out[f"{REFERENCE_MODEL}@{window_label(w)}"] = w
    return out


def factor_records(
    result: ForecastResult, source: str, horizons=LOG_HORIZONS, model: str = REFERENCE_MODEL
) -> list[ForecastRecord]:
    return [
        ForecastRecord(
            origin_time=result.last_time,
            model=model,
            horizon=h,
            source=source,
            origin_price=result.last_close,
            var_bps2=float(result.cum_return_var[h - 1]),
            q=result.cum_q_grid[h - 1],
        )
        for h in horizons
        if h <= result.horizon
    ]


def baseline_records(
    closes: pd.DataFrame,
    origin_time: pd.Timestamp,
    origin_price: float,
    source: str,
    horizons=LOG_HORIZONS,
    n_sims: int = REPLAY_SIMS,
) -> list[ForecastRecord]:
    """``gjr`` and ``ewma`` on raw BTC returns over the production fit window."""
    window = closes.loc[closes.index > origin_time - FIT_WINDOW]
    btc = build_return_panel(window[[TARGET]])[TARGET]
    gen = np.random.default_rng(int(origin_time.timestamp()))
    out = []
    for name, forecasts in (
        ("gjr", gjr_forecast(btc, horizons, n_sims, gen)),
        ("ewma", ewma_forecast(btc, horizons)),
    ):
        out += [
            ForecastRecord(origin_time, name, h, source, origin_price, var, q)
            for h, (q, var) in forecasts.items()
        ]
    return out


def profile_before(store: CandleStore, day_start: pd.Timestamp) -> SeasonalProfile:
    """Seasonal profile from history strictly before ``day_start`` (no look-ahead)."""
    closes = store.read_closes(
        PRODUCTS,
        since=(day_start - STORE_WINDOW).to_pydatetime(),
        until=(day_start - pd.Timedelta(minutes=1)).to_pydatetime(),
    )
    if closes.empty:
        return SeasonalProfile.flat(PRODUCTS, SEASON_BUCKET_MINUTES)
    return estimate_profile(build_return_panel(closes), SEASON_BUCKET_MINUTES)


class Replayer:
    """Re-runs every model at past origins (oldest first) so scoring starts large.

    Origins sit on a ``REPLAY_STEP_MINUTES`` grid, starting once the longest candidate
    fit window of history exists. Each origin only sees data up to itself.
    """

    def __init__(self, store: CandleStore, forecast_log: ForecastLog):
        self.store = store
        self.log = forecast_log
        self.windows = model_windows()
        self.longest = max(self.windows.values())
        self.max_h = max(LOG_HORIZONS)
        self._profiles: dict[pd.Timestamp, SeasonalProfile] = {}
        self.progress = {"done": 0, "total": 0}

    def pending_origins(self) -> list[pd.Timestamp]:
        first, last = self.store.first_time(TARGET), self.store.last_time(TARGET)
        if first is None or last is None:
            return []
        start = pd.Timestamp(first + self.longest).ceil(f"{REPLAY_STEP_MINUTES}min")
        end = pd.Timestamp(last) - pd.Timedelta(minutes=self.max_h)
        if start > end:
            self.progress = {"done": 0, "total": 0}
            return []
        grid = pd.date_range(start, end, freq=f"{REPLAY_STEP_MINUTES}min")
        done = self.log.origins("replay")
        pending = [t for t in grid if t not in done]
        self.progress = {"done": len(grid) - len(pending), "total": len(grid)}
        return pending

    def process(self, origin: pd.Timestamp) -> int:
        day = origin.floor("D")
        if day not in self._profiles:
            self._profiles[day] = profile_before(self.store, day)
        profile = self._profiles[day]

        panel = self.store.read_panel(
            PRODUCTS,
            since=(origin - self.longest - pd.Timedelta(minutes=5)).to_pydatetime(),
            until=origin.to_pydatetime(),
        )
        closes = panel["close"]
        btc = closes[TARGET].dropna()
        if btc.empty or btc.index[-1] != origin:
            # No bar exactly at the origin: log a sentinel-free skip by marking nothing.
            return 0

        records: list[ForecastRecord] = []
        for name, window in self.windows.items():
            try:
                res = fit_and_forecast(
                    panel,
                    TARGET,
                    FACTOR_PRODUCTS,
                    self.max_h,
                    profile=profile,
                    fit_window=window,
                    n_sims=REPLAY_SIMS,
                )
            except InsufficientDataError:
                continue
            records += factor_records(res, "replay", model=name)
        if not any(r.model == REFERENCE_MODEL for r in records):
            return 0
        records += baseline_records(closes, origin, float(btc.iloc[-1]), "replay")
        self.log.insert(records)
        return len(records)

    def run(self, stop: threading.Event) -> None:
        stop.wait(30)  # let the first sync land
        skipped: set[pd.Timestamp] = set()
        while not stop.is_set():
            pending = [t for t in self.pending_origins() if t not in skipped]
            if not pending:
                stop.wait(300)
                continue
            for origin in pending[:REPLAY_BATCH]:
                try:
                    if self.process(origin) == 0:
                        skipped.add(origin)
                except Exception:
                    log.exception("Replay failed at %s", origin)
                    skipped.add(origin)
            stop.wait(1)
