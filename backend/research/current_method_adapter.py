"""The project's current forecasting method, exposed as a ``garch_fhs.ExternalForecaster``.

"Current method" = the production cone (``CONE_MODEL = "ensemble"``): an equal-weight
quantile average (Vincentization) of

* ``factor`` - PCA-CAPM + range-informed GJR-GARCH-X with time-of-day adjustment and
  filtered historical simulation (``src.services.quant.fit_and_forecast``),
* ``gjr``    - arch AR(1)-GJR-GARCH skew-t on BTC returns (``baselines.gjr_paths``),
* ``ewma``   - RiskMetrics EWMA with Gaussian quantiles (``baselines.ewma_paths``),

each fitted on the last ``FIT_WINDOW`` (72 h) before the origin, exactly as
``ForecastCache`` does in production. The seasonal profile is re-estimated at every
origin from the ``STORE_WINDOW`` (30 days) before it, matching production's hourly
refresh (origins are hourly).

Timestamp convention
--------------------
Coinbase candle ``time`` is the bucket OPEN time; its close is known one minute later.
The backtest's price series is indexed by CLOSE time (open + 1 min), so its value at t
is the close known at t. The panel passed here keeps Coinbase's open-time index, and
for an origin t the adapter only uses bars with open time <= t - 1 min.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from src.services.baselines import ewma_paths, gjr_paths
from src.services.quant import BPS, build_return_panel, fit_and_forecast
from src.services.scoring import combine_quantiles
from src.services.seasonality import estimate_profile
from src.settings import (
    FACTOR_PRODUCTS,
    FIT_WINDOW,
    PRODUCTS,
    SEASON_BUCKET_MINUTES,
    STORE_WINDOW,
    TARGET,
)

ONE_MIN = pd.Timedelta(minutes=1)
# 199 evenly spaced quantile levels: CRPS from these approximates the true CRPS.
LEVELS = np.linspace(0.005, 0.995, 199)


def make_current_method(
    panel: pd.DataFrame,
    fit_window: dt.timedelta = FIT_WINDOW,
    profile_window: dt.timedelta = STORE_WINDOW,
):
    """Build the forecaster.

    ``panel``: open-time-indexed (UTC) frame with (field, product) MultiIndex columns,
    fields close/high/low, products ``PRODUCTS`` (the layout of
    ``CandleStore.read_panel``). It may extend past the evaluation period: every call
    slices it to what was known at the origin.
    """
    panel = panel.sort_index()
    index = panel.index
    fit_window = pd.Timedelta(fit_window)
    profile_window = pd.Timedelta(profile_window)

    def current_method(
        train: pd.Series, horizon: int, n_sims: int, rng: np.random.Generator
    ) -> np.ndarray:
        origin = pd.Timestamp(train.index[-1])  # close time of the origin bar
        last_open = origin - ONE_MIN  # newest candle whose close is known at origin
        end = index.searchsorted(last_open, side="right")
        known = panel.iloc[:end]
        last_time = known.index[-1]

        # Seasonal profile from the 30 days up to the origin (production refreshes hourly).
        prof_start = known.index.searchsorted(last_time - profile_window, side="right")
        prof_closes = known["close"].iloc[prof_start:][PRODUCTS]
        profile = estimate_profile(build_return_panel(prof_closes), SEASON_BUCKET_MINUTES)

        fit_start = known.index.searchsorted(last_time - fit_window - 5 * ONE_MIN, side="right")
        fit_panel = known.iloc[fit_start:]

        factor_seed, gjr_seed = (int(s) for s in rng.integers(0, 2**31 - 1, size=2))
        factor = fit_and_forecast(
            fit_panel,
            TARGET,
            FACTOR_PRODUCTS,
            horizon,
            profile=profile,
            fit_window=fit_window,
            n_sims=n_sims,
            seed=factor_seed,
            extra_levels=LEVELS,
        )

        # Baselines on raw BTC returns over the same window, as ForecastCache does.
        closes = fit_panel["close"]
        window = closes.loc[closes.index > last_time - fit_window]
        btc = build_return_panel(window[[TARGET]])[TARGET]
        gjr_q, gjr_var = gjr_paths(btc, horizon, n_sims, np.random.default_rng(gjr_seed), LEVELS)
        ewma_q, ewma_var = ewma_paths(btc, horizon, LEVELS)

        q_bps, _ = combine_quantiles(
            np.stack([factor.cum_q_extra, gjr_q, ewma_q], axis=1),  # (horizon, 3, 199)
            np.column_stack([factor.cum_return_var, gjr_var, ewma_var]),
        )
        # Rows = quantile levels, columns = horizons; anchored on the shared series.
        return float(np.log(train.iloc[-1])) + q_bps.T / BPS

    return current_method
