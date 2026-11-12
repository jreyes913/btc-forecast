"""Rolling-origin backtest of the factor model's forecast bands and variance forecasts.

At each origin the model is fitted on a trailing window only, so nothing from the future
can influence the forecast. The forecast is then scored against the realized cumulative
BTC log return over each horizon, and compared with two baselines:

* ``ewma``: RiskMetrics EWMA variance (lambda = 0.94) with Gaussian bands
* ``gjr``:  AR(1)-GJR-GARCH(1,1) skew-t fitted directly to BTC returns (no factor stage)

Metrics per (model, horizon):

* ``cov68`` / ``cov95``: share of realized returns inside the 68% / 95% bands
* ``qlike``: mean of log(s2) + r^2 / s2. A volatility loss that is robust to noisy
  realized proxies; lower is better.
* ``mse``: mean squared error of the variance forecast against r^2
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt

import numpy as np
import pandas as pd
from arch import arch_model

from src.services.quant import (
    BPS,
    InsufficientDataError,
    build_return_panel,
    fit_and_forecast,
    simulate_returns,
)
from src.settings import MAX_FFILL_MINUTES

Z68 = 0.994457883  # Gaussian quantile for a 68% central band (p16-p84)
Z95 = 1.959963985
EWMA_LAMBDA = 0.94


@dc.dataclass(frozen=True)
class BacktestRow:
    model: str
    horizon: int
    n: int
    cov68: float
    cov95: float
    qlike: float
    mse: float


def _score(
    model: str, horizon: int, real: np.ndarray, bands: np.ndarray, var: np.ndarray
) -> BacktestRow:
    """``bands`` columns: lo95, lo68, hi68, hi95 (cumulative log return, bps)."""
    var = np.maximum(var, 1e-12)
    return BacktestRow(
        model=model,
        horizon=horizon,
        n=len(real),
        cov68=float(np.mean((real >= bands[:, 1]) & (real <= bands[:, 2]))),
        cov95=float(np.mean((real >= bands[:, 0]) & (real <= bands[:, 3]))),
        qlike=float(np.mean(np.log(var) + real**2 / var)),
        mse=float(np.mean((real**2 - var) ** 2)),
    )


def _quantile_bands(cum: np.ndarray, h: int) -> list[float]:
    return list(np.quantile(cum[:, h - 1], [0.025, 0.16, 0.84, 0.975]))


def _gaussian_bands(var: float) -> list[float]:
    sd = np.sqrt(var)
    return [-Z95 * sd, -Z68 * sd, Z68 * sd, Z95 * sd]


def run_backtest(
    closes: pd.DataFrame,
    target: str,
    factors: list[str],
    horizons: tuple[int, ...] = (1, 15, 60),
    train: dt.timedelta = dt.timedelta(hours=48),
    step_minutes: int = 30,
    n_sims: int = 300,
) -> list[BacktestRow]:
    closes = closes.sort_index()
    max_h = max(horizons)
    grid = pd.date_range(closes.index.min(), closes.index.max(), freq="1min")
    btc = closes[target].reindex(grid).ffill(limit=MAX_FFILL_MINUTES)

    origins = grid[(grid >= grid[0] + train) & (grid <= grid[-1] - pd.Timedelta(minutes=max_h))]
    origins = origins[::step_minutes]

    records: dict[str, dict[int, list]] = {
        m: {h: [] for h in horizons} for m in ("factor", "gjr", "ewma")
    }
    for t in origins:
        if np.isnan(btc.loc[t]):
            continue
        window = closes.loc[(closes.index > t - train) & (closes.index <= t)]
        try:
            res = fit_and_forecast(window, target, factors, max_h, n_sims=n_sims)
        except InsufficientDataError:
            continue

        btc_ret = build_return_panel(window[[target]])[target]
        gjr = arch_model(
            btc_ret, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="skewt", rescale=False
        ).fit(disp="off", options={"maxiter": 500})
        gen = np.random.default_rng(int(t.timestamp()))
        gjr_cum = np.cumsum(simulate_returns(gjr, max_h, n_sims, gen), axis=1)

        ewma_var = (btc_ret**2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean().iloc[-1]

        for h in horizons:
            end_px = btc.get(t + pd.Timedelta(minutes=h))
            if end_px is None or np.isnan(end_px):
                continue
            real = float(np.log(end_px / btc.loc[t]) * BPS)
            q = res.cum_return_quantiles
            records["factor"][h].append(
                (real, [q["p025"][h - 1], q["p16"][h - 1], q["p84"][h - 1], q["p975"][h - 1]],
                 res.cum_return_var[h - 1])
            )
            records["gjr"][h].append((real, _quantile_bands(gjr_cum, h), gjr_cum[:, h - 1].var()))
            records["ewma"][h].append((real, _gaussian_bands(h * ewma_var), h * ewma_var))

    rows = []
    for model, by_h in records.items():
        for h, recs in by_h.items():
            if not recs:
                continue
            real = np.array([r[0] for r in recs])
            bands = np.array([r[1] for r in recs])
            var = np.array([r[2] for r in recs])
            rows.append(_score(model, h, real, bands, var))
    return rows


def format_table(rows: list[BacktestRow]) -> str:
    header = f"{'model':<8}{'h':>4}{'n':>5}{'cov68':>8}{'cov95':>8}{'qlike':>9}{'mse':>14}"
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: (r.horizon, r.model)):
        lines.append(
            f"{r.model:<8}{r.horizon:>4}{r.n:>5}{r.cov68:>8.2f}{r.cov95:>8.2f}"
            f"{r.qlike:>9.3f}{r.mse:>14.1f}"
        )
    return "\n".join(lines)
