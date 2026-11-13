"""Caches one model fit per (last closed bar, horizon) so requests never trigger refits.

The seasonal profile is estimated from the full stored history and refreshed hourly;
each fit uses only the last ``FIT_WINDOW`` of data.
"""

from __future__ import annotations

import datetime as dt
import threading

import pandas as pd

from src.services.candles import CandleStore
from src.services.quant import ForecastResult, build_return_panel, fit_and_forecast
from src.services.seasonality import SeasonalProfile, estimate_profile
from src.settings import FIT_WINDOW, SEASON_BUCKET_MINUTES, STORE_WINDOW

PROFILE_REFRESH = dt.timedelta(hours=1)


class ForecastCache:
    def __init__(
        self,
        store: CandleStore,
        target: str,
        factors: list[str],
        fit_window: dt.timedelta = FIT_WINDOW,
    ):
        self.store = store
        self.target = target
        self.factors = factors
        self.products = [target, *factors]
        self.fit_window = fit_window
        self._results: dict[int, ForecastResult] = {}  # horizon -> latest result
        self._profile: SeasonalProfile | None = None
        self._profile_at: pd.Timestamp | None = None
        self._lock = threading.Lock()

    def profile(self, now: pd.Timestamp) -> SeasonalProfile:
        if self._profile is None or now - self._profile_at >= PROFILE_REFRESH:
            closes = self.store.read_closes(
                self.products, since=(now - STORE_WINDOW).to_pydatetime()
            )
            self._profile = (
                estimate_profile(build_return_panel(closes), SEASON_BUCKET_MINUTES)
                if not closes.empty
                else SeasonalProfile.flat(self.products, SEASON_BUCKET_MINUTES)
            )
            self._profile_at = now
        return self._profile

    def latest(self, horizon: int) -> ForecastResult | None:
        with self._lock:
            return self._results.get(horizon)

    def get(self, horizon: int) -> ForecastResult:
        """Latest forecast for ``horizon``; refits only if a new bar has closed since."""
        with self._lock:  # serialize fits: concurrent requests wait for one fit
            last = self.store.last_time(self.target)
            if last is None:
                raise LookupError("no candles stored yet")
            last_time = pd.Timestamp(last)

            cached = self._results.get(horizon)
            if cached is not None and cached.last_time == last_time:
                return cached
            panel = self.store.read_panel(
                self.products,
                since=(last_time - self.fit_window - pd.Timedelta(minutes=5)).to_pydatetime(),
            )
            result = fit_and_forecast(
                panel,
                self.target,
                self.factors,
                horizon,
                profile=self.profile(last_time),
                fit_window=self.fit_window,
            )
            self._results[horizon] = result
            return result
