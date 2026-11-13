"""Accumulating forecast scoring.

Every forecast is logged with its origin time, horizon and the quantiles of the
cumulative BTC log return on ``Q_GRID``. Once the target minute has closed, the
realized return is filled in. The summary therefore grows with every forecast instead of
re-scoring a sliding window.

Two sources feed the log:

* ``live``: forecasts made in real time, every ``LOG_EVERY_MINUTES``.
* ``replay``: historical forecasts made at past origins using only data available at
  the time, so the sample is large from day one.

Metrics per (model, horizon):

* ``cov68`` / ``cov95``: share of outcomes inside the 68% / 95% bands.
* ``qlike``: log(s2) + r^2/s2, a variance-forecast loss robust to noisy proxies. Lower
  is better.
* ``brier``: mean squared error of P(BTC >= K) at Kalshi-style $100 strikes between
  each forecast's 1st and 99th percentile prices. Lower is better.
* calibration: forecast probabilities binned against observed frequencies.
* Diebold-Mariano test of each model's QLIKE against the factor model, with
  Newey-West standard errors (overlapping horizons make losses autocorrelated).
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.services.candles import CandleStore
from src.services.quant import Q_GRID
from src.settings import ENSEMBLE_FORGETTING, ENSEMBLE_LOSS_ETA, MAX_FFILL_MINUTES, TARGET

REFERENCE_MODEL = "factor"
ENSEMBLE_MODEL = "ensemble"
ENSEMBLE_MEMBERS = ("factor", "gjr", "ewma")
STRIKE_STEP = 100.0
MAX_STRIKES = 200
CALIBRATION_BINS = 10
_I16, _I84 = int(np.argmin(abs(Q_GRID - 0.16))), int(np.argmin(abs(Q_GRID - 0.84)))
_I025, _I975 = int(np.argmin(abs(Q_GRID - 0.025))), int(np.argmin(abs(Q_GRID - 0.975)))
_I01, _I99 = 0, len(Q_GRID) - 1


@dc.dataclass(frozen=True)
class ForecastRecord:
    origin_time: pd.Timestamp
    model: str
    horizon: int
    source: str
    origin_price: float
    var_bps2: float
    q: np.ndarray  # cumulative log return quantiles (bps) on Q_GRID


class ForecastLog:
    """The ``forecast_log`` table, sharing the candle store's connection and lock."""

    def __init__(self, store: CandleStore):
        self.store = store
        with store.lock:
            store.con.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_log (
                    origin_time TIMESTAMPTZ NOT NULL,
                    model VARCHAR NOT NULL,
                    horizon INTEGER NOT NULL,
                    source VARCHAR NOT NULL,
                    origin_price DOUBLE NOT NULL,
                    var_bps2 DOUBLE NOT NULL,
                    q DOUBLE[] NOT NULL,
                    realized_bps DOUBLE,
                    PRIMARY KEY (origin_time, model, horizon, source)
                )
                """
            )

    def insert(self, records: list[ForecastRecord]) -> None:
        """Idempotent: re-inserting a logged forecast is a no-op."""
        if not records:
            return
        df = pd.DataFrame(
            {
                "origin_time": [r.origin_time for r in records],
                "model": [r.model for r in records],
                "horizon": [int(r.horizon) for r in records],
                "source": [r.source for r in records],
                "origin_price": [float(r.origin_price) for r in records],
                "var_bps2": [float(r.var_bps2) for r in records],
                "q": [list(map(float, r.q)) for r in records],
            }
        )
        with self.store.lock:
            self.store.con.register("new_forecasts", df)
            try:
                self.store.con.execute(
                    """
                    INSERT OR IGNORE INTO forecast_log
                    SELECT origin_time, model, horizon, source, origin_price, var_bps2,
                           q::DOUBLE[], NULL FROM new_forecasts
                    """
                )
            finally:
                self.store.con.unregister("new_forecasts")

    def origins(self, source: str, model: str = REFERENCE_MODEL) -> set[pd.Timestamp]:
        with self.store.lock:
            df = self.store.con.execute(
                "SELECT DISTINCT origin_time FROM forecast_log WHERE source = ? AND model = ?",
                [source, model],
            ).df()
        return set(pd.DatetimeIndex(df["origin_time"]))

    def resolve(self, target: str = TARGET) -> int:
        """Fill ``realized_bps`` for forecasts whose target minute has closed.

        Returns the number of rows resolved. Targets with no price within
        ``MAX_FFILL_MINUTES`` (a data hole) are marked NaN so they aren't retried.
        """
        last = self.store.last_time(target)
        if last is None:
            return 0
        with self.store.lock:
            pending = self.store.con.execute(
                """
                SELECT origin_time, model, horizon, source, origin_price
                FROM forecast_log
                WHERE realized_bps IS NULL
                  AND origin_time + horizon * INTERVAL 1 MINUTE <= ?
                """,
                [last],
            ).df()
        if pending.empty:
            return 0

        pending["target_time"] = pending["origin_time"] + pd.to_timedelta(pending["horizon"], "min")
        since = pending["target_time"].min() - pd.Timedelta(minutes=MAX_FFILL_MINUTES + 1)
        closes = self.store.read_closes([target], since=since.to_pydatetime())[target].dropna()
        grid = pd.date_range(closes.index.min(), closes.index.max(), freq="1min")
        prices = closes.reindex(grid).ffill(limit=MAX_FFILL_MINUTES)
        target_px = prices.reindex(pd.DatetimeIndex(pending["target_time"])).to_numpy()
        pending["realized_bps"] = np.log(target_px / pending["origin_price"].to_numpy()) * 1e4
        pending["realized_bps"] = pending["realized_bps"].astype(float)  # NaN marks holes

        upd = pending[["origin_time", "model", "horizon", "source", "realized_bps"]]
        with self.store.lock:
            self.store.con.register("resolved", upd)
            try:
                self.store.con.execute(
                    """
                    UPDATE forecast_log AS f SET realized_bps = r.realized_bps
                    FROM resolved AS r
                    WHERE f.origin_time = r.origin_time AND f.model = r.model
                      AND f.horizon = r.horizon AND f.source = r.source
                    """
                )
            finally:
                self.store.con.unregister("resolved")
        return len(upd)

    def read_resolved(self, source: str | None = None) -> pd.DataFrame:
        query = (
            "SELECT * FROM forecast_log WHERE realized_bps IS NOT NULL AND isfinite(realized_bps)"
        )
        params: list = []
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        with self.store.lock:
            return self.store.con.execute(query, params).df()

    def prune(self, cutoff: dt.datetime) -> None:
        with self.store.lock:
            self.store.con.execute("DELETE FROM forecast_log WHERE origin_time < ?", [cutoff])


# --- Metrics ---------------------------------------------------------------------------


def quantile_cdf(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(R <= x) from quantile rows ``q`` (n, len(Q_GRID)) at points ``x`` (n, m).

    Piecewise-linear between grid quantiles, monotone by construction, clamped to
    [Q_GRID[0], Q_GRID[-1]] outside the grid.
    """
    q = np.maximum.accumulate(q, axis=1)  # guard against tiny non-monotonicities
    k = len(Q_GRID)
    j = (x[:, :, None] > q[:, None, :]).sum(axis=-1)  # number of grid quantiles below x
    j = np.clip(j, 1, k - 1)
    rows = np.arange(len(q))[:, None]
    lo, hi = q[rows, j - 1], q[rows, j]
    t = np.clip(np.where(hi > lo, (x - lo) / np.where(hi > lo, hi - lo, 1.0), 1.0), 0.0, 1.0)
    return Q_GRID[j - 1] + t * (Q_GRID[j] - Q_GRID[j - 1])


def strike_probabilities(
    q: np.ndarray,
    origin_price: np.ndarray,
    realized: np.ndarray,
    lo_px: np.ndarray | None = None,
    hi_px: np.ndarray | None = None,
    step: float = STRIKE_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """P(BTC >= K) and outcomes at every $``step`` strike in [lo_px, hi_px] per row.

    The strike range defaults to each row's own p1..p99. To compare models fairly,
    pass a common range: otherwise a model with too-wide bands gets scored on more
    easy, far-away strikes and looks better than it is.
    """
    if lo_px is None:
        lo_px = origin_price * np.exp(q[:, _I01] / 1e4)
    if hi_px is None:
        hi_px = origin_price * np.exp(q[:, _I99] / 1e4)
    first = np.ceil(lo_px / step) * step
    m = int(min(MAX_STRIKES, np.max(np.floor((hi_px - first) / step)) + 1)) if len(q) else 0
    if m <= 0:
        return np.empty(0), np.empty(0)
    strikes = first[:, None] + step * np.arange(m)[None, :]
    valid = strikes <= hi_px[:, None]
    x = np.log(strikes / origin_price[:, None]) * 1e4
    p_above = 1.0 - quantile_cdf(q, x)
    outcome = (realized[:, None] >= x).astype(float)
    return p_above[valid], outcome[valid]


def calibration_bins(p: np.ndarray, o: np.ndarray, bins: int = CALIBRATION_BINS) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            out.append(
                {
                    "p_mean": float(p[mask].mean()),
                    "obs_freq": float(o[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return out


def diebold_mariano(d: np.ndarray, lag: int) -> tuple[float, float]:
    """Mean loss differential and two-sided p-value with Newey-West variance."""
    n = len(d)
    if n < 10:
        return float(np.mean(d)) if n else float("nan"), float("nan")
    dc_ = d - d.mean()
    lr_var = np.dot(dc_, dc_) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1)
        lr_var += 2 * w * np.dot(dc_[k:], dc_[:-k]) / n
    if lr_var <= 0:
        return float(d.mean()), float("nan")
    stat = d.mean() / math.sqrt(lr_var / n)
    return float(d.mean()), float(2 * (1 - norm.cdf(abs(stat))))


def _qlike(var: np.ndarray, r: np.ndarray) -> np.ndarray:
    var = np.maximum(var, 1e-12)
    return np.log(var) + r**2 / var


def combine_quantiles(
    q: np.ndarray, var: np.ndarray, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted quantile averaging (Vincentization) over axis 1.

    ``q`` has shape (n, members, len(Q_GRID)), ``var`` (n, members) and ``weights``
    (n, members), rows summing to 1 (equal weights if omitted). The combined variance
    is (weighted mean SD)^2, consistent with averaging location-scale quantiles.
    Averaging quantiles keeps the bands sharp; averaging CDFs (a linear pool) would
    widen them, which is the opposite of what these forecasts need.
    """
    if weights is None:
        return q.mean(axis=1), np.sqrt(var).mean(axis=1) ** 2
    q_comb = np.einsum("nk,nkg->ng", weights, q)
    sd = np.einsum("nk,nk->n", weights, np.sqrt(var))
    return q_comb, sd**2


def quantile_density(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Predictive density at ``x`` (n,) implied by quantile rows ``q`` (n, len(Q_GRID)).

    Between the 1st and 99th percentiles the density is piecewise constant, matching the
    piecewise-linear CDF in ``quantile_cdf``. Beyond them the tails are exponential,
    continuing the decay between the 95th and 99th percentiles (scale (q99 - q95) / ln 5),
    so each tail holds exactly the 1% of mass the grid leaves outside.
    """
    q = np.maximum.accumulate(q, axis=1)
    g = len(Q_GRID)
    rows = np.arange(len(q))
    j = np.clip((x[:, None] >= q).sum(axis=1) - 1, 0, g - 2)
    width = np.maximum(q[rows, j + 1] - q[rows, j], 1e-9)
    inner = (Q_GRID[j + 1] - Q_GRID[j]) / width

    i05, i95 = int(np.argmin(abs(Q_GRID - 0.05))), int(np.argmin(abs(Q_GRID - 0.95)))
    tail_lo, tail_hi = Q_GRID[0], 1.0 - Q_GRID[-1]
    s_hi = np.maximum(q[:, -1] - q[:, i95], 1e-9) / np.log((1 - Q_GRID[i95]) / tail_hi)
    s_lo = np.maximum(q[:, i05] - q[:, 0], 1e-9) / np.log(Q_GRID[i05] / tail_lo)
    upper = tail_hi / s_hi * np.exp(-np.maximum(x - q[:, -1], 0.0) / s_hi)
    lower = tail_lo / s_lo * np.exp(-np.maximum(q[:, 0] - x, 0.0) / s_lo)
    dens = np.where(x > q[:, -1], upper, np.where(x < q[:, 0], lower, inner))
    return np.maximum(dens, 1e-12)


def pinball_loss(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Mean quantile (pinball) loss across ``Q_GRID``: an approximation of CRPS."""
    diff = r[:, None] - q
    return np.mean(np.maximum(Q_GRID * diff, (Q_GRID - 1) * diff), axis=1)


def sequential_weights(
    origins: np.ndarray, horizon: int, scores: np.ndarray, forgetting: float
) -> tuple[np.ndarray, np.ndarray]:
    """Causal, discounted model weights for forecasts ordered by origin time.

    ``scores`` (n, members) is each member's score for each forecast's outcome. An
    outcome only becomes known at origin + horizon, so it updates the state only for
    origins at or after that time: no forecast's weights use information from its own
    future. State update per resolved outcome: S <- forgetting * S + score; weights are
    softmax(S), starting from equal weights. Returns the per-origin weights (n, members)
    and the latest weights after every outcome is in (members,).
    """
    n, k = scores.shape
    resolve = origins + np.timedelta64(int(horizon), "m")
    state = np.zeros(k)
    weights = np.empty((n, k))
    j = 0
    for i in range(n):
        while j < n and resolve[j] <= origins[i]:
            state = forgetting * state + scores[j]
            j += 1
        w = np.exp(state - state.max())
        weights[i] = w / w.sum()
    for jj in range(j, n):
        state = forgetting * state + scores[jj]
    latest = np.exp(state - state.max())
    return weights, latest / latest.sum()


def ensemble_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Append the three ensemble variants built from logged member forecasts.

    * ``ensemble``: equal weights.
    * ``ensemble_bma``: dynamic Bayesian model averaging; each resolved outcome
      multiplies a member's weight by its predictive density there (with forgetting).
    * ``ensemble_loss``: weights from discounted relative pinball loss, more robust to
      how the tails are interpolated from the logged quantiles.

    One row per (origin, horizon, source) where every member in ``ENSEMBLE_MEMBERS``
    was logged. Built at summary time, so historical forecasts get scored too. Also
    returns the latest adaptive weights per (variant, source, horizon).
    """
    keys = ["origin_time", "horizon", "source"]
    members = df.loc[df["model"].isin(ENSEMBLE_MEMBERS)] if not df.empty else df
    if members.empty:
        return df, []
    counts = members.groupby(keys)["model"].nunique()
    complete = counts[counts == len(ENSEMBLE_MEMBERS)].index.to_frame(index=False)
    if complete.empty:
        return df, []
    k = len(ENSEMBLE_MEMBERS)
    m = members.merge(complete, on=keys)
    m["_member"] = m["model"].map({name: i for i, name in enumerate(ENSEMBLE_MEMBERS)})
    m = m.sort_values(["source", "horizon", "origin_time", "_member"])

    frames, latest = [], []
    for (source, h), g in m.groupby(["source", "horizon"], sort=False):
        q = np.vstack(g["q"].to_numpy()).reshape(-1, k, len(Q_GRID))
        var = g["var_bps2"].to_numpy().reshape(-1, k)
        base = g.iloc[::k][[*keys, "origin_price", "realized_bps"]]
        r = base["realized_bps"].to_numpy()
        origins = base["origin_time"].to_numpy(dtype="datetime64[ns]")

        loglik = np.column_stack([np.log(quantile_density(q[:, i], r)) for i in range(k)])
        loss = np.column_stack([pinball_loss(q[:, i], r) for i in range(k)])
        # Scale by the members' average forecast SD, known *before* the outcome. Dividing
        # by the realized loss level instead would make the rule improper: it shrinks
        # penalties on big moves and quietly favors overly narrow forecasts.
        scale = np.sqrt(var).mean(axis=1, keepdims=True)
        rel_loss = loss / np.maximum(scale, 1e-12)
        variants = {
            ENSEMBLE_MODEL: None,
            "ensemble_bma": loglik,
            "ensemble_loss": -ENSEMBLE_LOSS_ETA * rel_loss,
        }
        for name, scores in variants.items():
            weights = None
            if scores is not None:
                weights, last = sequential_weights(origins, int(h), scores, ENSEMBLE_FORGETTING)
                latest.append(
                    {
                        "variant": name,
                        "source": source,
                        "horizon": int(h),
                        "updates": int(len(base)),
                        "weights": dict(zip(ENSEMBLE_MEMBERS, map(float, last), strict=True)),
                    }
                )
            q_ens, var_ens = combine_quantiles(q, var, weights)
            ens = base.copy()
            ens["model"] = name
            ens["var_bps2"] = var_ens
            ens["q"] = list(q_ens)
            frames.append(ens)

    out = pd.concat([df, *[f.reindex(columns=df.columns) for f in frames]], ignore_index=True)
    return out, latest


def add_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` plus the ensemble rows (see ``ensemble_frames``)."""
    return ensemble_frames(df)[0]


def summarize(df: pd.DataFrame) -> dict:
    """Summary rows, calibration, DM tests and a fit-window recommendation."""
    if df.empty:
        return {"rows": [], "calibration": [], "recommended_fit_window": None,
                "ensemble_weights": []}

    df, ensemble_weights = ensemble_frames(df)
    df = df.sort_values("origin_time")
    # Common strike range per (origin, horizon): the reference model's p1..p99 where it
    # exists, otherwise the row's own. Every model is then scored on the same strikes.
    ref = df.loc[df["model"] == REFERENCE_MODEL]
    ref_q = np.vstack(ref["q"].to_numpy()) if len(ref) else np.empty((0, len(Q_GRID)))
    ref_range = {
        (t, int(h)): (p * np.exp(qq[_I01] / 1e4), p * np.exp(qq[_I99] / 1e4))
        for t, h, p, qq in zip(
            ref["origin_time"], ref["horizon"], ref["origin_price"], ref_q, strict=True
        )
    }
    rows, calibration = [], []
    losses: dict[tuple[str, int], pd.Series] = {}
    for (model, h), g in df.groupby(["model", "horizon"]):
        q = np.vstack(g["q"].to_numpy())
        r = g["realized_bps"].to_numpy()
        var = g["var_bps2"].to_numpy()
        ql = _qlike(var, r)
        losses[(model, int(h))] = pd.Series(ql, index=pd.DatetimeIndex(g["origin_time"]))
        px = g["origin_price"].to_numpy()
        own_lo, own_hi = px * np.exp(q[:, _I01] / 1e4), px * np.exp(q[:, _I99] / 1e4)
        rng = [ref_range.get((t, int(h))) for t in g["origin_time"]]
        lo = np.array([x[0] if x else d for x, d in zip(rng, own_lo, strict=True)])
        hi = np.array([x[1] if x else d for x, d in zip(rng, own_hi, strict=True)])
        p, o = strike_probabilities(q, px, r, lo, hi)
        rows.append(
            {
                "model": model,
                "horizon": int(h),
                "n": int(len(g)),
                "cov68": float(np.mean((r >= q[:, _I16]) & (r <= q[:, _I84]))),
                "cov95": float(np.mean((r >= q[:, _I025]) & (r <= q[:, _I975]))),
                "qlike": float(ql.mean()),
                "mse": float(np.mean((r**2 - var) ** 2)),
                "brier": float(np.mean((p - o) ** 2)) if len(p) else None,
                "n_strikes": int(len(p)),
            }
        )
        calibration.append({"model": model, "horizon": int(h), "bins": calibration_bins(p, o)})

    # Diebold-Mariano: each model vs the reference, on common origins
    for row in rows:
        ref = losses.get((REFERENCE_MODEL, row["horizon"]))
        row["dm_mean_diff"] = row["dm_p_value"] = None
        if row["model"] == REFERENCE_MODEL or ref is None:
            continue
        mine = losses[(row["model"], row["horizon"])]
        both = pd.concat([ref, mine], axis=1, join="inner").dropna()
        if both.empty:
            continue
        step = np.median(np.diff(both.index.asi8)) / 6e10 if len(both) > 1 else 1.0
        lag = max(1, math.ceil(row["horizon"] / max(step, 1.0)))
        # Positive mean diff = this model's loss is higher = the factor model is better.
        mean_diff, p_value = diebold_mariano((both.iloc[:, 1] - both.iloc[:, 0]).to_numpy(), lag)
        row["dm_mean_diff"], row["dm_p_value"] = mean_diff, p_value

    return _json_safe(
        {
            "rows": rows,
            "calibration": calibration,
            "recommended_fit_window": recommend_fit_window(rows),
            "ensemble_weights": ensemble_weights,
        }
    )


def _json_safe(obj):
    """Replace NaN/inf (e.g. an undefined DM p-value) with None for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def recommend_fit_window(rows: list[dict], horizon: int = 30) -> str | None:
    """Lowest-QLIKE factor-model variant at ``horizon`` (needs replay variants)."""
    cands = [r for r in rows if r["model"].startswith(REFERENCE_MODEL) and r["horizon"] == horizon]
    if len(cands) < 2:
        return None
    return min(cands, key=lambda r: r["qlike"])["model"]
