"""Run the pre-registered GARCH-FHS vs current-method backtest.

Usage (from backend/):  python -m research.run_backtest
Settings are frozen in research/PREREGISTRATION.md; do not change them after a run.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

from research.current_method_adapter import make_current_method
from research.data_io import close_time_frame, load_candles, load_panel
from research.garch_fhs import prepare_minute_closes, rolling_backtest
from src.settings import TARGET

RESULTS = Path(__file__).resolve().parent / "results"

# Pre-registered window (close-time index): 7-day warm-up from Jul 25 00:01, so the
# first origin is Aug 1 00:00; through Sep 15 01:00 so the last origin is Sep 15 00:00.
PRICES_START = pd.Timestamp("2026-07-25 00:01", tz="UTC")
PRICES_END = pd.Timestamp("2026-09-15 01:00", tz="UTC")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    btc = close_time_frame(load_candles(TARGET))
    btc = btc.loc[(btc["timestamp"] >= PRICES_START) & (btc["timestamp"] <= PRICES_END)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prices = prepare_minute_closes(btc, "timestamp", "close")
    data_info = {
        "source": "Coinbase Exchange public API, BTC-USD 1-minute candles",
        "timestamp_convention": "candle open time + 1 min = close time (value at t known at t)",
        "requested_range": [str(PRICES_START), str(PRICES_END)],
        "kept_range": [str(prices.index[0]), str(prices.index[-1])],
        "minutes_kept": int(len(prices)),
        "candles_in_range": int(len(btc)),
        "missing_minutes_before_fill": int(prices.attrs["missing_minutes_before_fill"]),
        "warnings": [str(w.message) for w in caught],
    }
    print(json.dumps(data_info, indent=2))

    current = make_current_method(load_panel())
    t0 = time.time()
    bt = rolling_backtest(
        prices,
        eval_horizons=(1, 15, 30, 60),
        min_train=10_080, rolling_window=10_080, step=60,
        n_simulations=2_000, drift_shrinkage=0.0, drift_uncertainty=True,
        external_models={"current": current},
        n_bootstrap=2_000, confidence=0.99, seed=42,
    )
    runtime = time.time() - t0

    bt.to_csv(RESULTS / "garch_fhs_backtest.csv")
    attrs = {**bt.attrs, "data": data_info, "runtime_seconds": round(runtime, 1)}
    with open(RESULTS / "garch_fhs_backtest_attrs.json", "w") as f:
        json.dump(attrs, f, indent=2, default=str)

    cols = [c for c in bt.columns
            if c.startswith(("improvement_vs_", "verdict_vs_", "skill_vs_", "mae_verdict_vs_"))
            or c.endswith(("_crps", "_mae"))]
    print(bt[cols].T)
    print(json.dumps(attrs, indent=2, default=str))


if __name__ == "__main__":
    main()
