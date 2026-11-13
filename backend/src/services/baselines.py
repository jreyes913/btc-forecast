"""Benchmark forecasters the factor model is scored against.

* ``gjr``: AR(1)-GJR-GARCH(1,1) with skew-t fitted directly to BTC returns via ``arch``
  (no factor stage, no seasonality, no range term). An independent implementation, so
  it also guards against bugs in our own estimator.
* ``ewma``: RiskMetrics EWMA variance (lambda = 0.94) with Gaussian quantiles.

Both return, per horizon, quantiles of the cumulative log return (bps) on ``Q_GRID``
and the forecast variance, the same format the factor model logs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy.stats import norm

from src.services.quant import Q_GRID

EWMA_LAMBDA = 0.94

HorizonForecasts = dict[int, tuple[np.ndarray, float]]  # horizon -> (quantiles, variance)


def _inverse_cdf_rng(fit: ARCHModelResult, gen: np.random.Generator) -> Callable:
    """Seeded standardized innovations (arch's simulator ignores ``random_state`` for
    skew-t, so we draw via the inverse CDF instead)."""
    dist = fit.model.distribution
    names = dist.parameter_names()
    dist_params = fit.params[names].values if names else None
    return lambda size: dist.ppf(gen.uniform(size=size), dist_params)


def gjr_forecast(
    btc_returns: pd.Series, horizons: Iterable[int], n_sims: int, gen: np.random.Generator
) -> HorizonForecasts:
    horizons = list(horizons)
    fit = arch_model(
        btc_returns, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False
    ).fit(disp="off", options={"maxiter": 500})
    fc = fit.forecast(
        horizon=max(horizons),
        method="simulation",
        simulations=n_sims,
        reindex=False,
        rng=_inverse_cdf_rng(fit, gen),
    )
    cum = np.cumsum(fc.simulations.values[-1], axis=1)
    return {h: (np.quantile(cum[:, h - 1], Q_GRID), float(cum[:, h - 1].var())) for h in horizons}


def ewma_forecast(btc_returns: pd.Series, horizons: Iterable[int]) -> HorizonForecasts:
    var1 = float((btc_returns**2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().iloc[-1])
    z = norm.ppf(Q_GRID)
    return {h: (z * np.sqrt(h * var1), h * var1) for h in horizons}
