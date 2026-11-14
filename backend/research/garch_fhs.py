from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from arch.bootstrap import optimal_block_length
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit


# =====================================================================
# Data preparation: regular 1-minute close series
# =====================================================================

def prepare_minute_closes(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    max_ffill: int = 5,
) -> pd.Series:
    """
    Build a strictly regular 1-minute close series (UTC DatetimeIndex).

    - Sorts, de-duplicates (keeps last), reindexes to a 1-min grid.
    - Forward-fills gaps of up to `max_ffill` minutes.
    - If longer gaps remain, keeps the longest contiguous segment and warns,
      so no simulated return ever spans an unobserved gap.

    The timestamp must identify the candle whose CLOSE is used. Confirm the
    exchange convention (open-time vs close-time) before using this.
    """
    s = (
        df[[timestamp_col, close_col]]
        .assign(**{timestamp_col: pd.to_datetime(df[timestamp_col], utc=True)})
        .drop_duplicates(subset=timestamp_col, keep="last")
        .set_index(timestamp_col)[close_col]
        .astype(float)
        .sort_index()
    )
    s = s[s > 0]

    grid = pd.date_range(s.index[0], s.index[-1], freq="1min")
    s = s.reindex(grid)
    n_missing = int(s.isna().sum())
    s = s.ffill(limit=max_ffill)

    if s.isna().any():
        valid = s.notna().to_numpy()
        segment_id = np.cumsum(~valid)
        lengths = pd.Series(valid).groupby(segment_id).sum()
        best = lengths.idxmax()
        mask = (segment_id == best) & valid
        warnings.warn(
            f"Gaps longer than {max_ffill} min found; keeping longest "
            f"contiguous segment ({int(mask.sum())} of {len(s)} minutes)."
        )
        s = s[mask]

    s.attrs["missing_minutes_before_fill"] = n_missing
    s.name = "close"
    return s


# =====================================================================
# GARCH(1,1) estimation (Gaussian QML)
# =====================================================================

@dataclass
class GARCHFit:
    mu: float
    mu_se: float
    omega: float
    alpha: float
    beta: float
    residuals: np.ndarray
    conditional_variance: np.ndarray
    standardized_residuals: np.ndarray
    optimizer_params: np.ndarray


def unpack_garch(params: np.ndarray) -> tuple[float, float, float]:
    omega = np.exp(params[0])
    alpha = 0.999 * expit(params[1])
    beta = (0.999 - alpha) * expit(params[2])
    return omega, alpha, beta


def garch_variance(
    residuals: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """Conditional variances via a linear filter; var[0] = sample variance."""
    n = len(residuals)
    variance = np.empty(n)
    variance[0] = max(float(np.var(residuals)), 1e-12)
    if n > 1:
        inputs = omega + alpha * residuals[:-1] ** 2
        variance[1:], _ = lfilter(
            [1.0], [1.0, -beta], inputs, zi=[beta * variance[0]]
        )
    return np.maximum(variance, 1e-12)


def _standardize(residuals: np.ndarray, variance: np.ndarray) -> np.ndarray:
    z = residuals / np.sqrt(variance)
    z = z - np.mean(z)
    return z / np.std(z, ddof=1)


def fit_garch(
    returns: np.ndarray, initial_params: np.ndarray | None = None
) -> GARCHFit:
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) < 30:
        raise ValueError("At least 30 one-dimensional returns are required.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns contain non-finite values.")

    mu = float(np.mean(returns))
    mu_se = float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
    residuals = returns - mu
    sample_variance = max(float(np.var(residuals)), 1e-12)

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = unpack_garch(params)
        v = garch_variance(residuals, omega, alpha, beta)
        return float(0.5 * np.sum(np.log(v) + residuals**2 / v))

    if initial_params is None:
        initial_params = np.array([np.log(sample_variance * 0.05), -2.5, 2.0])

    result = minimize(
        objective, initial_params, method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    omega, alpha, beta = unpack_garch(result.x)
    variance = garch_variance(residuals, omega, alpha, beta)

    return GARCHFit(
        mu=mu, mu_se=mu_se, omega=omega, alpha=alpha, beta=beta,
        residuals=residuals, conditional_variance=variance,
        standardized_residuals=_standardize(residuals, variance),
        optimizer_params=result.x.copy(),
    )


def rebuild_garch_state(returns: np.ndarray, previous: GARCHFit) -> GARCHFit:
    """Reuse previous parameters, but rebuild state from the CURRENT window."""
    returns = np.asarray(returns, dtype=float)
    mu = float(np.mean(returns))
    mu_se = float(np.std(returns, ddof=1) / np.sqrt(len(returns)))
    residuals = returns - mu
    variance = garch_variance(
        residuals, previous.omega, previous.alpha, previous.beta
    )
    return GARCHFit(
        mu=mu, mu_se=mu_se, omega=previous.omega, alpha=previous.alpha,
        beta=previous.beta, residuals=residuals,
        conditional_variance=variance,
        standardized_residuals=_standardize(residuals, variance),
        optimizer_params=previous.optimizer_params.copy(),
    )


# =====================================================================
# Drift and simulators. All return per-step log returns (n_sims, horizon).
# =====================================================================

def draw_drift(model, n_simulations, drift_shrinkage, drift_uncertainty, rng):
    if not 0 <= drift_shrinkage <= 1:
        raise ValueError("drift_shrinkage must be between 0 and 1.")
    mean = drift_shrinkage * model.mu
    if drift_uncertainty:
        return rng.normal(loc=mean, scale=model.mu_se, size=n_simulations)
    return np.full(n_simulations, mean)


def simulate_garch_fhs(model, horizon, n_simulations, drift_shrinkage,
                       drift_uncertainty, rng) -> np.ndarray:
    """GARCH(1,1) volatility + bootstrapped standardized residuals (FHS)."""
    if horizon < 1 or n_simulations < 1:
        raise ValueError("Horizon and simulation count must be positive.")
    drift = draw_drift(model, n_simulations, drift_shrinkage,
                       drift_uncertainty, rng)

    # sigma^2_{T+1} from sigma^2_T and eps_T
    variance = np.full(
        n_simulations,
        model.omega + model.alpha * model.residuals[-1] ** 2
        + model.beta * model.conditional_variance[-1],
    )
    out = np.empty((n_simulations, horizon))
    z_hist = model.standardized_residuals
    for t in range(horizon):
        z = rng.choice(z_hist, size=n_simulations, replace=True)
        residual = np.sqrt(variance) * z
        out[:, t] = drift + residual
        variance = model.omega + model.alpha * residual**2 + model.beta * variance
    return out


def simulate_iid_bootstrap(model, horizon, n_simulations, drift_shrinkage,
                           drift_uncertainty, rng) -> np.ndarray:
    """Baseline: i.i.d. bootstrap of demeaned historical returns."""
    drift = draw_drift(model, n_simulations, drift_shrinkage,
                       drift_uncertainty, rng)
    resid = rng.choice(model.residuals, size=(n_simulations, horizon),
                       replace=True)
    return drift[:, None] + resid


def simulate_gaussian(model, horizon, n_simulations, drift_shrinkage,
                      drift_uncertainty, rng) -> np.ndarray:
    """Baseline: Gaussian random walk with constant volatility."""
    drift = draw_drift(model, n_simulations, drift_shrinkage,
                       drift_uncertainty, rng)
    sigma = float(np.std(model.residuals, ddof=1))
    return rng.normal(loc=drift[:, None], scale=sigma,
                      size=(n_simulations, horizon))


# =====================================================================
# Scoring and inference
# =====================================================================

def crps_ensemble_matrix(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Empirical CRPS per horizon.
    samples (m, horizon), observations (horizon,) -> (horizon,).
    With m == 1 (a point forecast) this reduces exactly to absolute error.
    """
    samples = np.asarray(samples, dtype=float)
    observations = np.asarray(observations, dtype=float)
    m = samples.shape[0]
    sorted_samples = np.sort(samples, axis=0)
    first = np.mean(np.abs(samples - observations[None, :]), axis=0)
    coefficients = 2 * np.arange(1, m + 1) - m - 1
    second = (coefficients @ sorted_samples) / m**2
    return first - second


def block_bootstrap_ci(values, block_length, n_bootstrap=2_000,
                       confidence=0.99, rng=None) -> tuple[float, float]:
    """Moving-block bootstrap percentile CI for a mean (vectorized)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    length = min(max(1, block_length), n)
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=(n_bootstrap, n_blocks))
    idx = (starts[:, :, None] + np.arange(length)).reshape(n_bootstrap, -1)[:, :n]
    means = values[idx].mean(axis=1)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return float(lo), float(hi)


def estimate_block_length(differences, minimum: int = 1) -> int:
    """Politis-White (circular) estimate, floored at `minimum`, capped at n."""
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n <= 1:
        return 1
    try:
        estimate = optimal_block_length(differences)
        raw = float(estimate["circular"].iloc[0])
        if not np.isfinite(raw) or raw < 1:
            raise ValueError("Invalid estimated block length")
        length = int(np.ceil(raw))
    except (ValueError, FloatingPointError, IndexError, KeyError):
        length = int(np.ceil(np.sqrt(n)))
    return min(n, max(minimum, length))


# =====================================================================
# Rolling-origin backtest
# =====================================================================

# An external forecaster: given ONLY the training closes (up to and including
# the origin), return log-price samples of shape (m, horizon), where column
# h-1 is the forecast for the close h minutes after the origin.
# m == 1 is allowed (point forecast): CRPS becomes absolute error and
# coverage is reported as NaN.
ExternalForecaster = Callable[[pd.Series, int, int, np.random.Generator], np.ndarray]

BUILTIN_SIMULATORS = {
    "garch": simulate_garch_fhs,
    "iid": simulate_iid_bootstrap,
    "gaussian": simulate_gaussian,
}


def _verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "garch_better"
    if ci_high < 0:
        return "garch_worse"
    return "inconclusive"


def rolling_backtest(
    prices: pd.Series,
    eval_horizons: tuple[int, ...] = (1, 15, 30, 60),
    horizon_buckets: dict[str, tuple[int, int]] | None = None,
    min_train: int = 10_080,
    rolling_window: int | None = 10_080,
    step: int = 60,
    n_simulations: int = 2_000,
    drift_shrinkage: float = 0.0,
    drift_uncertainty: bool = True,
    external_models: dict[str, ExternalForecaster] | None = None,
    n_bootstrap: int = 2_000,
    confidence: float = 0.99,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare GARCH-FHS against IID-bootstrap, Gaussian and any external models.

    All models are scored on the same origins and realized outcomes.
    CRPS is on log prices. Inference: per bucket, average paired CRPS
    differences (other - garch) over the bucket's horizons per origin, then a
    moving-block bootstrap CI across origins. Positive = GARCH better.
    Secondary: the same paired test on absolute error of the median forecast
    (log price), which is the fair comparison against point forecasters.
    """
    prices = prices.dropna().astype(float)
    if (prices <= 0).any():
        raise ValueError("Prices must be positive.")
    if step < 1:
        raise ValueError("step must be positive.")
    if min_train < 31:
        raise ValueError("min_train must be at least 31.")

    horizon = int(max(eval_horizons))
    if horizon < 1:
        raise ValueError("eval_horizons must be positive.")

    if rolling_window is not None:
        if rolling_window < 31:
            raise ValueError("rolling_window must be at least 31.")
        min_train = max(min_train, rolling_window)

    if len(prices) < min_train + horizon:
        raise ValueError("Insufficient data for requested backtest.")

    if horizon_buckets is None:
        horizon_buckets = {f"h{h}": (h, h) for h in sorted(set(eval_horizons))}

    buckets = {}
    for name, (lower, upper) in horizon_buckets.items():
        lower, upper = max(1, lower), min(horizon, upper)
        if lower <= upper:
            buckets[name] = (lower, upper)
    if not buckets:
        raise ValueError("No valid horizon buckets.")

    external_models = external_models or {}
    clash = set(external_models) & set(BUILTIN_SIMULATORS)
    if clash:
        raise ValueError(f"External model names clash with built-ins: {clash}")

    model_names = tuple(BUILTIN_SIMULATORS) + tuple(external_models)
    rng = np.random.default_rng(seed)
    q_levels = np.array([0.005, 0.025, 0.16, 0.84, 0.975, 0.995])

    records = {n: {k: [] for k in ("crps", "ae", "cov68", "cov95", "cov99")}
               for n in model_names}

    previous_model = None
    fit_failures = 0
    total_origins = 0
    persistence = []

    for origin in range(min_train - 1, len(prices) - horizon, step):
        end = origin + 1
        if rolling_window is None:
            training = prices.iloc[:end]
        else:
            training = prices.iloc[max(0, end - rolling_window):end]

        actual_log = np.log(prices.iloc[end:end + horizon].to_numpy())
        rets = np.diff(np.log(training.to_numpy()))

        try:
            model = fit_garch(
                rets,
                initial_params=(previous_model.optimizer_params
                                if previous_model is not None else None),
            )
            previous_model = model
        except (RuntimeError, ValueError, FloatingPointError):
            fit_failures += 1
            if previous_model is None:
                raise RuntimeError("First GARCH fit failed; no previous fit.")
            model = rebuild_garch_state(rets, previous_model)

        total_origins += 1
        persistence.append(model.alpha + model.beta)
        log_last = float(np.log(training.iloc[-1]))

        log_sims = {
            name: log_last + np.cumsum(
                sim(model, horizon, n_simulations, drift_shrinkage,
                    drift_uncertainty, rng),
                axis=1,
            )
            for name, sim in BUILTIN_SIMULATORS.items()
        }
        for name, forecaster in external_models.items():
            samples = np.asarray(
                forecaster(training, horizon, n_simulations, rng), dtype=float
            )
            if samples.ndim == 1:
                samples = samples[None, :]
            if samples.shape[1] != horizon or not np.all(np.isfinite(samples)):
                raise ValueError(
                    f"External model '{name}' must return finite log-price "
                    f"samples of shape (m, {horizon}); got {samples.shape}."
                )
            log_sims[name] = samples

        for name in model_names:
            s = log_sims[name]
            records[name]["crps"].append(crps_ensemble_matrix(s, actual_log))
            # Absolute error of the median (point-forecast comparison).
            records[name]["ae"].append(np.abs(np.median(s, axis=0) - actual_log))
            if s.shape[0] > 1:
                q = np.quantile(s, q_levels, axis=0)
                cov = {
                    "cov99": (q[0] <= actual_log) & (actual_log <= q[5]),
                    "cov95": (q[1] <= actual_log) & (actual_log <= q[4]),
                    "cov68": (q[2] <= actual_log) & (actual_log <= q[3]),
                }
            else:
                nan = np.full(horizon, np.nan)
                cov = {"cov99": nan, "cov95": nan, "cov68": nan}
            for k, v in cov.items():
                records[name][k].append(np.asarray(v, dtype=float))

    arrays = {n: {k: np.asarray(v) for k, v in records[n].items()}
              for n in model_names}

    rows = []
    for bucket, (lower, upper) in buckets.items():
        sel = slice(lower - 1, upper)
        minimum_block = int(np.ceil(upper / step))
        row = {"bucket": bucket, "horizon_start": lower,
               "horizon_end": upper, "origins": total_origins}

        for name in model_names:
            row[f"{name}_crps"] = float(np.mean(arrays[name]["crps"][:, sel]))
            row[f"{name}_mae"] = float(np.mean(arrays[name]["ae"][:, sel]))
            for level in (68, 95, 99):
                cov = arrays[name][f"cov{level}"][:, sel]
                if np.all(np.isnan(cov)):
                    row[f"{name}_coverage_{level}"] = np.nan
                    row[f"{name}_misses_{level}"] = np.nan
                else:
                    row[f"{name}_coverage_{level}"] = float(np.mean(cov))
                    row[f"{name}_misses_{level}"] = int(cov.size - np.sum(cov))
                row[f"{name}_evaluations_{level}"] = int(cov.size)

        garch_crps = arrays["garch"]["crps"][:, sel]
        for other in model_names:
            if other == "garch":
                continue
            other_crps = arrays[other]["crps"][:, sel]
            paired_difference = np.mean(other_crps - garch_crps, axis=1)
            block_length = estimate_block_length(paired_difference,
                                                 minimum=minimum_block)
            ci_low, ci_high = block_bootstrap_ci(
                paired_difference, block_length=block_length,
                n_bootstrap=n_bootstrap, confidence=confidence, rng=rng,
            )
            mean_diff = float(np.mean(paired_difference))
            base = float(np.mean(other_crps))
            row[f"block_length_vs_{other}"] = block_length
            row[f"improvement_vs_{other}"] = mean_diff
            row[f"improvement_vs_{other}_ci_low"] = ci_low
            row[f"improvement_vs_{other}_ci_high"] = ci_high
            row[f"skill_vs_{other}"] = mean_diff / base if base > 0 else np.nan
            row[f"verdict_vs_{other}"] = _verdict(ci_low, ci_high)

            # Secondary: paired absolute error of the median forecast.
            ae_diff = np.mean(
                arrays[other]["ae"][:, sel] - arrays["garch"]["ae"][:, sel],
                axis=1,
            )
            ae_low, ae_high = block_bootstrap_ci(
                ae_diff,
                block_length=estimate_block_length(ae_diff, minimum=minimum_block),
                n_bootstrap=n_bootstrap, confidence=confidence, rng=rng,
            )
            row[f"mae_improvement_vs_{other}"] = float(np.mean(ae_diff))
            row[f"mae_improvement_vs_{other}_ci_low"] = ae_low
            row[f"mae_improvement_vs_{other}_ci_high"] = ae_high
            row[f"mae_verdict_vs_{other}"] = _verdict(ae_low, ae_high)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("bucket")
    p = np.asarray(persistence)
    result.attrs.update({
        "fit_failures": fit_failures,
        "total_origins": total_origins,
        "horizon_buckets": buckets,
        "confidence": confidence,
        "persistence_median": float(np.median(p)),
        "persistence_share_at_bound": float(np.mean(p > 0.998)),
    })
    return result


if __name__ == "__main__":
    import json

    data = pd.read_csv("btc_minutes.csv")          # adapt to your source
    prices = prepare_minute_closes(data, "timestamp", "close")

    backtest = rolling_backtest(prices)            # add external_models=...
    backtest.to_csv("garch_fhs_backtest.csv")
    with open("garch_fhs_backtest_attrs.json", "w") as f:
        json.dump(backtest.attrs, f, indent=2, default=str)

    cols = [c for c in backtest.columns
            if c.startswith(("improvement_vs_", "verdict_vs_", "skill_vs_",
                            "mae_verdict_vs_"))
            or c.endswith(("_crps", "_mae"))]
    print(backtest[cols].T)
    print(json.dumps(backtest.attrs, indent=2, default=str))
