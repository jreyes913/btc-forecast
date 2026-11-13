"""GJR-GARCH-X estimation and filtered historical simulation (FHS).

Variance equation (all inputs already deseasonalized, returns in bps):

    sigma2_t = omega + (alpha + gamma * 1[e_{t-1} < 0]) * e_{t-1}^2 + beta * sigma2_{t-1}
               + delta * x_{t-1}

``x`` is an exogenous variance proxy (here: Parkinson high/low range variance). ``arch``
can't put regressors in the variance equation, so this module estimates the model itself
by Gaussian quasi-maximum likelihood. QMLE is consistent for the variance parameters
whatever the true shock distribution; the shock *shape* comes back in at simulation time
through FHS, which resamples the model's own standardized residuals.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

MAX_PERSISTENCE = 0.9995
_PENALTY = 1e10


@dc.dataclass(frozen=True)
class GarchXFit:
    params: dict[str, float]
    sigma2: np.ndarray  # conditional variance, aligned with ``resid``
    resid: np.ndarray  # e_t = y_t - mu - phi * y_{t-1}
    z: np.ndarray  # standardized residuals e_t / sigma_t
    u: np.ndarray  # standardized range x_t / sigma2_t (zeros when there is no x)
    x: np.ndarray  # exogenous variance proxy (zeros when there is no x)
    last_y: float  # last observed return, for the AR(1) term
    converged: bool
    loglik: float

    @property
    def phi(self) -> float:
        return self.params.get("phi", 0.0)

    def next_sigma2(self) -> float:
        """One-step-ahead conditional variance after the last observation."""
        p = self.params
        e, s2, x = self.resid[-1], self.sigma2[-1], self.x[-1]
        return float(
            p["omega"]
            + (p["alpha"] + p.get("gamma", 0.0) * (e < 0)) * e**2
            + p["beta"] * s2
            + p.get("delta", 0.0) * x
        )

    def persistence(self) -> float:
        p = self.params
        kappa = float(np.mean(self.u)) if self.u.any() else 0.0  # E[x / sigma2]
        return p["alpha"] + p.get("gamma", 0.0) / 2 + p["beta"] + p.get("delta", 0.0) * kappa


def _names(ar: bool, asym: bool, has_x: bool) -> list[str]:
    names = ["mu", "phi"] if ar else []
    names += ["omega", "alpha"] + (["gamma"] if asym else []) + ["beta"]
    return names + (["delta"] if has_x else [])


def _filter(theta: np.ndarray, names: list[str], y: np.ndarray, x: np.ndarray | None):
    """Residuals and conditional variances for parameter vector ``theta``."""
    p = dict(zip(names, theta, strict=True))
    if "mu" in p:
        e = y[1:] - p["mu"] - p["phi"] * y[:-1]
        xs = x[1:] if x is not None else None
    else:
        e = y
        xs = x
    e_lag = np.concatenate([[0.0], e[:-1]])
    neg = (e_lag < 0).astype(float)
    shock = (p["alpha"] + p.get("gamma", 0.0) * neg) * e_lag**2
    if xs is not None:
        shock = shock + p.get("delta", 0.0) * np.concatenate([[0.0], xs[:-1]])
    drive = p["omega"] + shock
    s2_0 = float(np.var(e))
    drive[0] = s2_0 * (1 - p["beta"])  # start the recursion at the sample variance
    s2 = lfilter([1.0], [1.0, -p["beta"]], drive)
    return e, s2, xs


def _nll(theta, names, y, x, x_mean_ratio):
    p = dict(zip(names, theta, strict=True))
    e, s2, _ = _filter(theta, names, y, x)
    if not np.all(np.isfinite(s2)) or s2.min() <= 1e-12:
        return _PENALTY
    nll = 0.5 * float(np.sum(np.log(s2) + e**2 / s2))
    # Smooth (quadratic) stationarity penalty: a hard wall breaks the optimizer's
    # finite-difference gradients near the boundary.
    persistence = (
        p["alpha"] + p.get("gamma", 0.0) / 2 + p["beta"] + p.get("delta", 0.0) * x_mean_ratio
    )
    excess = max(0.0, persistence - MAX_PERSISTENCE)
    return nll + 1e6 * len(e) * excess**2


def fit_garch_x(
    y: np.ndarray, x: np.ndarray | None = None, ar: bool = False, asym: bool = True
) -> GarchXFit:
    """Gaussian QMLE fit of an (AR(1)-)GJR-GARCH(1,1)-X model."""
    y = np.asarray(y, dtype=float)
    x = None if x is None else np.asarray(x, dtype=float)
    var = float(np.var(y))
    x_mean_ratio = float(np.mean(x) / var) if x is not None and var > 0 else 0.0

    def solve(names: list[str], start: dict[str, float]):
        bounds = {
            "mu": (-10 * np.sqrt(var), 10 * np.sqrt(var)),
            "phi": (-0.99, 0.99),
            "omega": (1e-8 * var, 10 * var),
            "alpha": (0.0, 1.0),
            "gamma": (0.0, 1.0),  # leverage effect: drops raise variance at least as much
            "beta": (0.0, MAX_PERSISTENCE),
            "delta": (0.0, 10.0),
        }
        theta0 = np.array([start[n] for n in names])
        res = minimize(
            _nll,
            theta0,
            args=(names, y, x, x_mean_ratio),
            method="L-BFGS-B",
            bounds=[bounds[n] for n in names],
            options={"maxiter": 500},
        )
        return res

    start = {
        "mu": float(np.mean(y)),
        "phi": 0.0,
        "omega": 0.05 * var,
        "alpha": 0.05,
        "gamma": 0.02,
        "beta": 0.90,
        "delta": 0.0,
    }
    base_names = _names(ar, asym, has_x=False)
    res = solve(base_names, start)
    if x is not None:
        # Warm-start the X model from the no-X fit so delta = 0 is always reachable.
        start.update(dict(zip(base_names, res.x, strict=True)))
        start["delta"] = 0.01 if x_mean_ratio == 0 else 0.05 / max(x_mean_ratio, 1e-6)
        start["delta"] = min(start["delta"], 1.0)
        start["beta"] = min(start["beta"], 0.85)
        names = _names(ar, asym, has_x=True)
        res_x = solve(names, start)
        # Keep the no-X solution if the X fit didn't improve the likelihood.
        if res_x.fun <= res.fun + 1e-6:
            res = res_x
        else:
            names, res = names, _with_zero_delta(res, base_names, names)
    else:
        names = base_names

    theta = res.x
    e, s2, xs = _filter(theta, names, y, x)
    xs = np.zeros_like(e) if xs is None else xs
    params = {n: float(v) for n, v in zip(names, theta, strict=True)}
    z = e / np.sqrt(s2)
    # Re-standardize to exactly mean 0 / variance 1 so resampled shocks can't bias the
    # simulated variance away from the fitted model's (standard FHS practice).
    z = (z - z.mean()) / z.std()
    return GarchXFit(
        params=params,
        sigma2=s2,
        resid=e,
        z=z,
        u=xs / s2,
        x=xs,
        last_y=float(y[-1]),
        converged=bool(res.success) and res.fun < _PENALTY,
        loglik=float(-res.fun - 0.5 * len(e) * np.log(2 * np.pi)),
    )


def _with_zero_delta(res, base_names, names):
    """Re-express a no-X optimum in the X parameterization with delta = 0."""
    p = dict(zip(base_names, res.x, strict=True))
    p["delta"] = 0.0
    res.x = np.array([p[n] for n in names])
    return res


def simulate_fhs(
    fits: list[GarchXFit], horizon: int, n_sims: int, gen: np.random.Generator
) -> list[np.ndarray]:
    """Filtered historical simulation of several components with *joint* resampling.

    At every step one historical minute is drawn and every component uses its own
    standardized shock z and standardized range u from that same minute. This keeps
    the empirical shock shape (peaked centre, fat tails) and cross-component
    dependence such as co-jumps. All ``fits`` must share the same time index.

    Returns one array of simulated returns, shape (n_sims, horizon), per fit.
    """
    n = len(fits[0].z)
    if any(len(f.z) != n for f in fits):
        raise ValueError("FHS components must be aligned on the same time index")
    idx = gen.integers(0, n, size=(n_sims, horizon))

    out = []
    for f in fits:
        p = f.params
        gamma, delta = p.get("gamma", 0.0), p.get("delta", 0.0)
        s2 = np.full(n_sims, f.next_sigma2())
        y_prev = np.full(n_sims, f.last_y)
        sims = np.empty((n_sims, horizon))
        for h in range(horizon):
            z, u = f.z[idx[:, h]], f.u[idx[:, h]]
            e = np.sqrt(s2) * z
            y = f.phi * y_prev + e  # zero drift: the AR constant is not simulated
            sims[:, h] = y
            s2 = (
                p["omega"] + (p["alpha"] + gamma * (e < 0)) * e**2 + p["beta"] * s2 + delta * s2 * u
            )
            y_prev = y
        out.append(sims)
    return out
