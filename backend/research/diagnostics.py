"""Post-hoc, read-only diagnostic for the brief's section-3 check.

GARCH's 60-minute CRPS edge over the i.i.d./Gaussian baselines should be checked
against the gap between its long-run variance omega/(1-alpha-beta) and the window's
sample variance: a level difference is a different effect from better dynamics.
Refits the frozen ``fit_garch`` on the same 7-day windows at every 24th origin (daily)
and saves summary ratios. Does not change any backtest result.

Usage (from backend/):  python -m research.diagnostics
"""

from __future__ import annotations

import json

import numpy as np

from research.data_io import close_time_frame, load_candles
from research.garch_fhs import fit_garch, prepare_minute_closes
from research.run_backtest import PRICES_END, PRICES_START, RESULTS
from src.settings import TARGET

WINDOW, STEP, HORIZON = 10_080, 60, 60


def main() -> None:
    btc = close_time_frame(load_candles(TARGET))
    btc = btc.loc[(btc["timestamp"] >= PRICES_START) & (btc["timestamp"] <= PRICES_END)]
    prices = prepare_minute_closes(btc, "timestamp", "close")
    logp = np.log(prices.to_numpy())

    rows = []
    origins = list(range(WINDOW - 1, len(prices) - HORIZON, STEP))[::24]
    previous = None
    for origin in origins:
        rets = np.diff(logp[origin + 1 - WINDOW: origin + 1])
        fit = fit_garch(rets, previous.optimizer_params if previous is not None else None)
        previous = fit
        sample_var = float(np.var(fit.residuals))
        persistence = fit.alpha + fit.beta
        long_run = fit.omega / (1 - persistence)
        next_var = (
            fit.omega + fit.alpha * fit.residuals[-1] ** 2
            + fit.beta * fit.conditional_variance[-1]
        )
        # Average one-step variance over the next 60 minutes implied by GARCH
        k = np.arange(HORIZON)
        path = long_run + (next_var - long_run) * persistence**k
        rows.append({
            "persistence": persistence,
            "long_run_over_sample": long_run / sample_var,
            "next_over_sample": next_var / sample_var,
            "mean60_over_sample": float(path.mean() / sample_var),
            "half_life_minutes": float(np.log(0.5) / np.log(persistence)),
        })

    def q(key):
        v = np.array([r[key] for r in rows])
        return {"median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)),
                "p90": float(np.quantile(v, 0.9))}

    out = {
        "origins_checked": len(rows),
        "every_nth_origin": 24,
        **{key: q(key) for key in rows[0]},
    }
    with open(RESULTS / "garch_longrun_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
