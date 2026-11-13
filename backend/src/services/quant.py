"""Two-stage factor model for 1-minute BTC returns.

Pipeline (returns in basis points: log return x 1e4):

1. Returns and Parkinson high/low range variances on a complete minute grid.
2. Divide out each series' time-of-day volatility profile s(t).
3. Market factor: PC1 of the standardized basket returns.
4. CAPM: ``r_btc = alpha + beta * f + eps``.
5. GJR-GARCH-X for the factor and for the residual, each with a range-variance term.
6. Filtered historical simulation: both components are driven by jointly resampled
   historical shocks (real shape, co-jumps preserved).
7. Multiply the simulated BTC path by s(t) for the future minutes it covers.

Drift is set to zero in the forecast: alpha and the AR constant are estimated and
reported but not simulated. In-sample drift over a few days is noise, not signal.
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt

import numpy as np
import pandas as pd

from src.services.garch import GarchXFit, fit_garch_x, simulate_fhs
from src.services.seasonality import SeasonalProfile
from src.settings import MAX_FFILL_MINUTES, N_SIMULATIONS

BPS = 1e4
PARKINSON = 1.0 / (4.0 * np.log(2.0))
QUANTILES = {"p025": 0.025, "p16": 0.16, "p50": 0.50, "p84": 0.84, "p975": 0.975}
# Finer grid logged for scoring (probabilities at arbitrary strikes are interpolated on it)
Q_GRID = np.array([0.01, 0.025, 0.05, 0.10, 0.16, 0.25, 0.50, 0.75, 0.84, 0.90, 0.95, 0.975, 0.99])
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


def build_range_panel(high: pd.DataFrame, low: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Parkinson range variance per minute, (ln(H/L))^2 / (4 ln 2), in bps^2.

    Minutes without a bar (no trades) have zero range.
    """
    rv = PARKINSON * (np.log(high / low) * BPS) ** 2
    return rv.reindex(index).fillna(0.0).clip(lower=0.0)


@dc.dataclass(frozen=True)
class FactorFit:
    pc1: pd.Series
    loadings: dict[str, float]
    explained_variance_ratio: float
    scale: pd.Series  # std of each input series used for standardization


def fit_factor(factor_returns: pd.DataFrame) -> FactorFit:
    """First principal component of the standardized factor returns.

    The sign is fixed so the first column's loading is positive, so PC1 means "the
    market went up" for every fit.
    """
    scale = factor_returns.std(ddof=0)
    z = (factor_returns - factor_returns.mean()) / scale
    eigvals, eigvecs = np.linalg.eigh(np.cov(z.values, rowvar=False))
    w = eigvecs[:, -1]  # eigh sorts ascending, so the last column is PC1
    if w[0] < 0:
        w = -w
    pc1 = pd.Series(z.values @ w, index=factor_returns.index, name="pc1")
    return FactorFit(
        pc1=pc1,
        loadings={col: float(v) for col, v in zip(factor_returns.columns, w, strict=True)},
        explained_variance_ratio=float(eigvals[-1] / eigvals.sum()),
        scale=scale,
    )


@dc.dataclass(frozen=True)
class CapmFit:
    alpha: float  # bps per minute (seasonally adjusted)
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


@dc.dataclass
class ForecastResult:
    last_time: pd.Timestamp
    last_close: float
    horizon: int
    n_obs: int
    fit_window_hours: float
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
    seasonality_active: bool
    # Share of the horizon-H cumulative return variance explained by the factor
    factor_variance_share: float
    # Conditional volatility history (bps per minute): total, idiosyncratic, and the
    # "typical for this time of day" level
    historical_volatility: pd.DataFrame
    # Forecast paths
    cum_return_quantiles: dict[str, np.ndarray]  # bps, keys of QUANTILES
    price_quantiles: dict[str, np.ndarray]
    price_mean: np.ndarray
    cum_return_var: np.ndarray  # forecast variance of the cumulative return (bps^2) per step
    cum_q_grid: np.ndarray  # (horizon, len(Q_GRID)) cumulative return quantiles, bps
    cone_model: str = "factor"  # which forecaster produced the bands


def _split_panel(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    if isinstance(panel.columns, pd.MultiIndex):
        fields = panel.columns.get_level_values(0)
        high = panel["high"] if "high" in fields else None
        low = panel["low"] if "low" in fields else None
        return panel["close"], high, low
    return panel, None, None


def fit_and_forecast(
    panel: pd.DataFrame,
    target: str,
    factors: list[str],
    horizon: int,
    profile: SeasonalProfile | None = None,
    fit_window: dt.timedelta | None = None,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = None,
) -> ForecastResult:
    """Fit on ``panel`` and forecast ``horizon`` minutes ahead.

    ``panel`` is either a wide close-price frame (one column per product) or the
    (field, product) frame from ``CandleStore.read_panel`` with close/high/low. Without
    high/low the range term is dropped. ``seed`` defaults to the last bar's timestamp so
    a given data snapshot always produces the same forecast.
    """
    closes, high, low = _split_panel(panel)
    cols = [target, *factors]
    if fit_window is not None:
        cutoff = closes.index.max() - fit_window
        keep = closes.index > cutoff
        closes = closes.loc[keep]
        high = high.loc[keep] if high is not None else None
        low = low.loc[keep] if low is not None else None

    returns = build_return_panel(closes[cols])
    if len(returns) < MIN_OBS:
        raise InsufficientDataError(f"need {MIN_OBS} aligned returns, have {len(returns)}")
    rv = build_range_panel(high[cols], low[cols], returns.index) if high is not None else None

    # --- Seasonal adjustment --------------------------------------------------------
    profile = profile or SeasonalProfile.flat(cols)
    s = profile.scales(returns)
    r_adj = returns / s
    rv_adj = rv / s**2 if rv is not None else None

    # --- Stage 1: factor and CAPM ----------------------------------------------------
    factor = fit_factor(r_adj[factors])
    capm = fit_capm(r_adj[target], factor.pc1)

    # --- Stage 2: GJR-GARCH-X for both components, aligned on index[1:] ---------------
    if rv_adj is not None:
        # Range proxy for the factor: mean of the basket's standardized range variances.
        x_factor = (rv_adj[factors] / factor.scale**2).mean(axis=1).to_numpy()
        x_resid = rv_adj[target].to_numpy()
    else:
        x_factor = x_resid = None
    resid_fit = fit_garch_x(capm.resid.to_numpy(), x_resid, ar=True)
    factor_fit = fit_garch_x(
        factor.pc1.to_numpy()[1:], None if x_factor is None else x_factor[1:], ar=False
    )
    fit_index = returns.index[1:]

    # --- Simulation -----------------------------------------------------------------
    last_close_series = closes[target].dropna()
    last_time = pd.Timestamp(last_close_series.index[-1])
    last_close = float(last_close_series.iloc[-1])
    if seed is None:
        seed = int(last_time.timestamp())
    gen = np.random.default_rng(seed)

    f_sims, e_sims = simulate_fhs([factor_fit, resid_fit], horizon, n_sims, gen)
    future = pd.date_range(last_time + pd.Timedelta(minutes=1), periods=horizon, freq="1min")
    s_future = profile.scale_at(target, future)
    steps_factor = capm.beta * f_sims * s_future
    steps = steps_factor + e_sims * s_future  # zero drift: no alpha
    cum = np.cumsum(steps, axis=1)
    cum_factor = np.cumsum(steps_factor, axis=1)

    cum_q = {k: np.quantile(cum, q, axis=0) for k, q in QUANTILES.items()}
    price_q = {k: last_close * np.exp(v / BPS) for k, v in cum_q.items()}

    hist = _historical_volatility(capm.beta, factor_fit, resid_fit, profile, target, fit_index)

    def report(fit: GarchXFit) -> dict[str, float]:
        out = {k: v for k, v in fit.params.items() if np.isfinite(v)}
        out["persistence"] = fit.persistence()
        return out

    span = (closes.index.max() - closes.index.min()) / pd.Timedelta(hours=1)
    return ForecastResult(
        last_time=last_time,
        last_close=last_close,
        horizon=horizon,
        n_obs=len(returns),
        fit_window_hours=float(fit_window / pd.Timedelta(hours=1)) if fit_window else float(span),
        alpha_bps=capm.alpha,
        beta=capm.beta,
        capm_r2=capm.r2,
        pca_explained_variance_ratio=factor.explained_variance_ratio,
        pca_loadings=factor.loadings,
        resid_std_bps=float(capm.resid.std()),
        resid_garch_params=report(resid_fit),
        factor_garch_params=report(factor_fit),
        resid_converged=resid_fit.converged,
        factor_converged=factor_fit.converged,
        seasonality_active=profile.active,
        factor_variance_share=float(cum_factor[:, -1].var() / cum[:, -1].var()),
        historical_volatility=hist,
        cum_return_quantiles=cum_q,
        price_quantiles=price_q,
        price_mean=last_close * np.exp(cum / BPS).mean(axis=0),
        cum_return_var=cum.var(axis=0),
        cum_q_grid=np.quantile(cum, Q_GRID, axis=0).T,
    )


def _historical_volatility(
    beta: float,
    factor_fit: GarchXFit,
    resid_fit: GarchXFit,
    profile: SeasonalProfile,
    target: str,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    s = profile.scale_at(target, index)
    total_adj2 = beta**2 * factor_fit.sigma2 + resid_fit.sigma2
    return pd.DataFrame(
        {
            "total": s * np.sqrt(total_adj2),
            "idiosyncratic": s * np.sqrt(resid_fit.sigma2),
            "typical": s * np.sqrt(total_adj2.mean()),
        },
        index=index,
    ).dropna()
