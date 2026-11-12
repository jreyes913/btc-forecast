"""Two-stage factor model for 1-minute BTC returns.

Stage 1: PC1 of the standardized factor-basket returns is the market factor, and BTC
returns are regressed on it: ``r_btc = alpha + beta * f + eps``.
Stage 2: the factor gets a GARCH(1,1)-t, and the idiosyncratic residual an
AR(1)-GJR-GARCH(1,1) with skew-t innovations.

Forecast bands come from Monte Carlo paths of *both* components, so they include factor
risk (beta^2 * Var(f)) as well as idiosyncratic risk. OLS residuals are orthogonal to the
regressor, so the two components are simulated independently.

All internal returns are in basis points (log return * 1e4), which keeps 1-minute
variances in a range where GARCH MLE converges cleanly.
"""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Callable

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from src.settings import MAX_FFILL_MINUTES, N_SIMULATIONS

BPS = 1e4
QUANTILES = {"p025": 0.025, "p16": 0.16, "p50": 0.50, "p84": 0.84, "p975": 0.975}
MIN_OBS = 500


class InsufficientDataError(ValueError):
    pass


def build_return_panel(closes: pd.DataFrame, max_ffill: int = MAX_FFILL_MINUTES) -> pd.DataFrame:
    """Aligned 1-minute log returns (bps) on a complete UTC minute grid.

    Coinbase omits minutes with no trades. A missing minute means "no trade", so the last
    price is carried forward (a zero return) for up to ``max_ffill`` minutes. Longer holes
    are dropped instead of being treated as a single 1-minute return.
    """
    if closes.empty:
        return closes.copy()
    grid = pd.date_range(closes.index.min(), closes.index.max(), freq="1min")
    prices = closes.sort_index().reindex(grid).ffill(limit=max_ffill)
    returns = np.log(prices).diff() * BPS
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


@dc.dataclass(frozen=True)
class FactorFit:
    pc1: pd.Series
    loadings: dict[str, float]
    explained_variance_ratio: float


def fit_factor(factor_returns: pd.DataFrame) -> FactorFit:
    """First principal component of the standardized factor returns.

    The sign is fixed so the first column's loading is positive, so PC1 means "the
    market went up" for every fit.
    """
    z = (factor_returns - factor_returns.mean()) / factor_returns.std(ddof=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(z.values, rowvar=False))
    w = eigvecs[:, -1]  # eigh sorts ascending, so the last column is PC1
    if w[0] < 0:
        w = -w
    pc1 = pd.Series(z.values @ w, index=factor_returns.index, name="pc1")
    return FactorFit(
        pc1=pc1,
        loadings={col: float(v) for col, v in zip(factor_returns.columns, w, strict=True)},
        explained_variance_ratio=float(eigvals[-1] / eigvals.sum()),
    )


@dc.dataclass(frozen=True)
class CapmFit:
    alpha: float  # bps per minute
    beta: float  # bps of BTC per unit of PC1
    r2: float
    resid: pd.Series


def fit_capm(y: pd.Series, f: pd.Series) -> CapmFit:
    X = np.column_stack([np.ones(len(f)), f.values])
    coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ coef
    r2 = 1.0 - resid.var() / y.values.var()
    return CapmFit(
        alpha=float(coef[0]),
        beta=float(coef[1]),
        r2=float(r2),
        resid=pd.Series(resid, index=y.index, name="resid"),
    )


def fit_resid_garch(resid: pd.Series) -> ARCHModelResult:
    model = arch_model(
        resid, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False
    )
    return model.fit(disp="off", options={"maxiter": 500})


def fit_factor_garch(pc1: pd.Series) -> ARCHModelResult:
    model = arch_model(pc1, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    return model.fit(disp="off", options={"maxiter": 500})


def _inverse_cdf_rng(fit: ARCHModelResult, gen: np.random.Generator) -> Callable:
    """Seeded standardized innovations for ``fit``'s distribution (arch's own simulator
    ignores ``random_state`` for skew-t, so we draw via the inverse CDF instead)."""
    dist = fit.model.distribution
    dist_params = fit.params[dist.parameter_names()].values if dist.parameter_names() else None
    return lambda size: dist.ppf(gen.uniform(size=size), dist_params)


def simulate_returns(
    fit: ARCHModelResult, horizon: int, n_sims: int, gen: np.random.Generator
) -> np.ndarray:
    """Simulated future returns, shape (n_sims, horizon)."""
    fc = fit.forecast(
        horizon=horizon,
        method="simulation",
        simulations=n_sims,
        reindex=False,
        rng=_inverse_cdf_rng(fit, gen),
    )
    return fc.simulations.values[-1]


@dc.dataclass
class ForecastResult:
    last_time: pd.Timestamp
    last_close: float
    horizon: int
    n_obs: int
    # Stage 1
    alpha_bps: float
    beta: float
    capm_r2: float
    pca_explained_variance_ratio: float
    pca_loadings: dict[str, float]
    resid_std_bps: float
    # Stage 2
    resid_garch_params: dict[str, float]
    factor_garch_params: dict[str, float]
    resid_converged: bool
    factor_converged: bool
    # Share of the horizon-H cumulative return variance explained by the factor
    factor_variance_share: float
    # Conditional volatility history (bps per minute): total and idiosyncratic
    historical_volatility: pd.DataFrame
    # Forecast paths: cumulative log return (bps) quantiles and price paths
    cum_return_quantiles: dict[str, np.ndarray]
    price_quantiles: dict[str, np.ndarray]
    price_mean: np.ndarray
    cum_return_var: np.ndarray  # forecast variance of cumulative return (bps^2) per step


def fit_and_forecast(
    closes: pd.DataFrame,
    target: str,
    factors: list[str],
    horizon: int,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = None,
) -> ForecastResult:
    """Fit both stages on ``closes`` (wide close prices) and forecast ``horizon`` minutes.

    ``seed`` defaults to the last bar's timestamp, so a given data snapshot always produces
    the same bands.
    """
    returns = build_return_panel(closes[[target, *factors]])
    if len(returns) < MIN_OBS:
        raise InsufficientDataError(f"need {MIN_OBS} aligned returns, have {len(returns)}")

    factor = fit_factor(returns[factors])
    capm = fit_capm(returns[target], factor.pc1)
    resid_fit = fit_resid_garch(capm.resid)
    factor_fit = fit_factor_garch(factor.pc1)

    last_close_series = closes[target].dropna()
    last_time = last_close_series.index[-1]
    last_close = float(last_close_series.iloc[-1])
    if seed is None:
        seed = int(pd.Timestamp(last_time).timestamp())
    gen = np.random.default_rng(seed)

    f_sims = simulate_returns(factor_fit, horizon, n_sims, gen)
    e_sims = simulate_returns(resid_fit, horizon, n_sims, gen)
    cum = np.cumsum(capm.alpha + capm.beta * f_sims + e_sims, axis=1)  # bps
    cum_factor = np.cumsum(capm.beta * f_sims, axis=1)

    cum_q = {k: np.quantile(cum, q, axis=0) for k, q in QUANTILES.items()}
    price_q = {k: last_close * np.exp(v / BPS) for k, v in cum_q.items()}
    cum_var = cum.var(axis=0)

    sigma_f = factor_fit.conditional_volatility
    sigma_e = resid_fit.conditional_volatility
    hist = pd.DataFrame(
        {
            "total": np.sqrt(capm.beta**2 * sigma_f**2 + sigma_e**2),
            "idiosyncratic": sigma_e,
        }
    ).dropna()

    def finite_params(fit: ARCHModelResult) -> dict[str, float]:
        return {k: float(v) for k, v in fit.params.items() if np.isfinite(v)}

    return ForecastResult(
        last_time=pd.Timestamp(last_time),
        last_close=last_close,
        horizon=horizon,
        n_obs=len(returns),
        alpha_bps=capm.alpha,
        beta=capm.beta,
        capm_r2=capm.r2,
        pca_explained_variance_ratio=factor.explained_variance_ratio,
        pca_loadings=factor.loadings,
        resid_std_bps=float(capm.resid.std()),
        resid_garch_params=finite_params(resid_fit),
        factor_garch_params=finite_params(factor_fit),
        resid_converged=resid_fit.convergence_flag == 0,
        factor_converged=factor_fit.convergence_flag == 0,
        factor_variance_share=float(cum_factor[:, -1].var() / cum[:, -1].var()),
        historical_volatility=hist,
        cum_return_quantiles=cum_q,
        price_quantiles=price_q,
        price_mean=last_close * np.exp(cum / BPS).mean(axis=0),
        cum_return_var=cum_var,
    )
