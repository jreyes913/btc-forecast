"""Caches one model fit per (last closed bar, horizon) so requests never trigger refits.

The seasonal profile is estimated from the full stored history and refreshed hourly;
each fit uses only the last ``FIT_WINDOW`` of data.

Two results are kept per horizon: the pure factor-model forecast (always the one that
is logged for scoring as ``factor``) and the cone shown on the dashboard, which is the
same object unless ``CONE_MODEL == "ensemble"``.
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import threading

import numpy as np
import pandas as pd

from src.services.baselines import ewma_paths, gjr_paths
from src.services.candles import CandleStore
from src.services.quant import (
    BPS,
    Q_GRID,
    QUANTILES,
    ForecastResult,
    build_return_panel,
    fit_and_forecast,
)
from src.services.scoring import combine_quantiles
from src.services.seasonality import SeasonalProfile, estimate_profile
from src.settings import (
    CONE_MODEL,
    FIT_WINDOW,
    N_SIMULATIONS,
    SEASON_BUCKET_MINUTES,
    STORE_WINDOW,
)

PROFILE_REFRESH = dt.timedelta(hours=1)


def ensemble_result(
    factor: ForecastResult, btc_returns: pd.Series, n_sims: int = N_SIMULATIONS
) -> ForecastResult:
    """Equal-weight quantile average of the factor, GJR-only and EWMA forecasts."""
    keys = list(QUANTILES)
    levels = np.concatenate([np.array([QUANTILES[k] for k in keys]), Q_GRID])
    h = factor.horizon
    gen = np.random.default_rng(int(factor.last_time.timestamp()))

    factor_q = np.hstack([np.column_stack([factor.cum_return_quantiles[k] for k in keys]),
                          factor.cum_q_grid])
    gjr_q, gjr_var = gjr_paths(btc_returns, h, n_sims, gen, levels)
    ewma_q, ewma_var = ewma_paths(btc_returns, h, levels)

    q, var = combine_quantiles(
        np.stack([factor_q, gjr_q, ewma_q], axis=1),
        np.column_stack([factor.cum_return_var, gjr_var, ewma_var]),
    )
    cum_q = {k: q[:, i] for i, k in enumerate(keys)}
    p50 = cum_q["p50"]
    return dc.replace(
        factor,
        cum_return_quantiles=cum_q,
        price_quantiles={k: factor.last_close * np.exp(v / BPS) for k, v in cum_q.items()},
        price_mean=factor.last_close * np.exp(p50 / BPS + 0.5 * var / BPS**2),
        cum_return_var=var,
        cum_q_grid=q[:, len(keys):],
        cone_model="ensemble",
    )


class ForecastCache:
    def __init__(
        self,
        store: CandleStore,
        target: str,
        factors: list[str],
        fit_window: dt.timedelta = FIT_WINDOW,
        cone_model: str = CONE_MODEL,
    ):
        if cone_model not in ("factor", "ensemble"):
            raise ValueError(f"CONE_MODEL must be 'factor' or 'ensemble', got {cone_model!r}")
        self.store = store
        self.target = target
        self.factors = factors
        self.products = [target, *factors]
        self.fit_window = fit_window
        self.cone_model = cone_model
        # horizon -> (factor result, cone result)
        self._results: dict[int, tuple[ForecastResult, ForecastResult]] = {}
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
        """Latest cone for ``horizon`` without refitting (None if never fitted)."""
        with self._lock:
            pair = self._results.get(horizon)
            return pair[1] if pair else None

    def get(self, horizon: int) -> ForecastResult:
        """The dashboard cone for ``horizon``; refits only if a new bar has closed."""
        return self._get(horizon)[1]

    def get_factor(self, horizon: int) -> ForecastResult:
        """The pure factor-model forecast, used for scoring whatever the cone shows."""
        return self._get(horizon)[0]

    def _get(self, horizon: int) -> tuple[ForecastResult, ForecastResult]:
        with self._lock:  # serialize fits: concurrent requests wait for one fit
            last = self.store.last_time(self.target)
            if last is None:
                raise LookupError("no candles stored yet")
            last_time = pd.Timestamp(last)

            cached = self._results.get(horizon)
            if cached is not None and cached[0].last_time == last_time:
                return cached
            panel = self.store.read_panel(
                self.products,
                since=(last_time - self.fit_window - pd.Timedelta(minutes=5)).to_pydatetime(),
            )
            factor = fit_and_forecast(
                panel,
                self.target,
                self.factors,
                horizon,
                profile=self.profile(last_time),
                fit_window=self.fit_window,
            )
            cone = factor
            if self.cone_model == "ensemble":
                closes = panel["close"]
                window = closes.loc[closes.index > last_time - self.fit_window]
                btc = build_return_panel(window[[self.target]])[self.target]
                cone = ensemble_result(factor, btc)
            self._results[horizon] = (factor, cone)
            return factor, cone
