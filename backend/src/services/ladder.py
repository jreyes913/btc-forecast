"""Strike-probability table for Kalshi-style hourly BTC deadlines.

For each $``step`` strike K around the current price, the chance BTC finishes at or
above K at the deadline, the chance it finishes below, and the most a trade is worth
paying under the "edge" rule (model probability minus ``edge`` percentage points).

The probabilities come from the dashboard cone's cumulative-return quantiles. When the
live (forming) bar's price is known, the table is anchored on it and uses the cone's
distribution for the minutes *remaining* until the deadline, so a move since the cone's
origin is already accounted for.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from src.services.quant import BPS, Q_GRID, ForecastResult
from src.services.scoring import quantile_cdf

BANDS = {  # cone quantile key -> label
    "p025": "95% low",
    "p16": "68% low",
    "p50": "median",
    "p84": "68% high",
    "p975": "95% high",
}
_LEVELS = {"p025": 0.025, "p16": 0.16, "p50": 0.50, "p84": 0.84, "p975": 0.975}
MIN_MINUTES_LEFT = 1


def deadline_for(now: dt.datetime, which: str) -> dt.datetime:
    """Next top of the hour (skipping one less than a minute away), or the one after."""
    top = now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    if top - now < dt.timedelta(minutes=MIN_MINUTES_LEFT):
        top += dt.timedelta(hours=1)
    return top + dt.timedelta(hours=1) if which == "following" else top


def _quantiles_at(cone: ForecastResult, minutes: int) -> np.ndarray:
    """Cumulative-return quantiles (bps) on Q_GRID after ``minutes`` steps."""
    return cone.cum_q_grid[minutes - 1]


def _band_quantiles(q_grid_row: np.ndarray) -> dict[str, float]:
    """Band levels (bps) interpolated from a Q_GRID quantile row."""
    return {k: float(np.interp(p, Q_GRID, q_grid_row)) for k, p in _LEVELS.items()}


def build_ladder(
    cone: ForecastResult,
    now: dt.datetime,
    live_price: float | None,
    which: str = "next",
    step: float = 100.0,
    edge: float = 10.0,
) -> dict:
    deadline = deadline_for(now, which)
    origin = cone.last_time.to_pydatetime()
    if live_price is not None:
        ref_price, ref_live = float(live_price), True
        minutes = math.ceil((deadline - now) / dt.timedelta(minutes=1))
    else:
        ref_price, ref_live = cone.last_close, False
        minutes = math.ceil((deadline - origin) / dt.timedelta(minutes=1))

    header = {
        "deadline": deadline,
        "which": which,
        "minutes_left": max(0, math.ceil((deadline - now) / dt.timedelta(minutes=1))),
        "reference_price": ref_price,
        "reference_is_live": ref_live,
        "cone_origin_time": origin,
        "cone_model": cone.cone_model,
        "edge_points": edge,
    }
    if minutes < 1 or minutes > cone.horizon:
        return {**header, "available": False, "rows": [], "ranges": None}

    q = _quantiles_at(cone, minutes)
    bands_bps = _band_quantiles(q)
    band_px = {k: ref_price * math.exp(v / BPS) for k, v in bands_bps.items()}

    # Strikes from one step below the 1st percentile to one step above the 99th, so
    # both 95% band edges and the tails beyond them are always in the table.
    lo_px = ref_price * math.exp(q[0] / BPS)
    hi_px = ref_price * math.exp(q[-1] / BPS)
    strikes = np.arange(math.floor(lo_px / step) * step - step,
                        math.ceil(hi_px / step) * step + step + 1e-9, step)

    x = np.log(strikes / ref_price) * BPS
    p_below = quantile_cdf(q[None, :], x[None, :])[0]
    p_above = 1.0 - p_below

    # Tag each band on its nearest strike. Near a deadline two bands can share a strike.
    band_of: dict[int, list[str]] = {}
    for key, label in BANDS.items():
        i = int(np.argmin(np.abs(strikes - band_px[key])))
        band_of.setdefault(i, []).append(label)
    current = int(np.argmin(np.abs(strikes - ref_price)))

    rows = []
    for i, (k, pa, pb, xi) in enumerate(zip(strikes, p_above, p_below, x, strict=True)):
        max_yes = pa * 100 - edge
        max_no = pb * 100 - edge
        rows.append(
            {
                "strike": float(k),
                "p_above": float(pa),
                "p_below": float(pb),
                "beyond_grid": "low" if xi < q[0] else "high" if xi > q[-1] else None,
                "max_yes_cents": round(max_yes, 1) if max_yes > 0 else None,
                "max_no_cents": round(max_no, 1) if max_no > 0 else None,
                "bands": band_of.get(i, []),
                "is_current": i == current,
            }
        )

    return {
        **header,
        "available": True,
        "horizon_minutes": minutes,
        "ranges": {
            "p68": [band_px["p16"], band_px["p84"]],
            "p95": [band_px["p025"], band_px["p975"]],
            "median": band_px["p50"],
        },
        "rows": rows,
    }

