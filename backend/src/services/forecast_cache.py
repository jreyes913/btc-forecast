"""Caches one model fit per (last closed bar, horizon) so requests never trigger refits."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pandas as pd

from src.services.quant import ForecastResult, fit_and_forecast


class ForecastCache:
    def __init__(
        self,
        load_closes: Callable[[], pd.DataFrame],
        target: str,
        factors: list[str],
        fit: Callable[..., ForecastResult] = fit_and_forecast,
    ):
        self._load_closes = load_closes
        self._target = target
        self._factors = factors
        self._fit = fit
        self._results: dict[int, ForecastResult] = {}  # horizon -> latest result
        self._lock = threading.Lock()

    def get(self, horizon: int) -> ForecastResult:
        """Latest forecast for ``horizon``; refits only if a new bar has closed since."""
        with self._lock:  # serialize fits: concurrent requests wait for one fit
            closes = self._load_closes()
            target_closes = closes[self._target].dropna() if self._target in closes else None
            if target_closes is None or target_closes.empty:
                raise LookupError("no candles stored yet")
            last_time = target_closes.index[-1]

            cached = self._results.get(horizon)
            if cached is not None and cached.last_time == last_time:
                return cached
            result = self._fit(closes, self._target, self._factors, horizon)
            self._results[horizon] = result
            return result
