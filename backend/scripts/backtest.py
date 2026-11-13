"""Print the accumulated forecast-scoring summary, or run a full offline replay demo.

Usage (from backend/):
    python -m scripts.backtest                    # summary from data/candles.duckdb
    python -m scripts.backtest --source replay    # only historical replay forecasts
    python -m scripts.backtest --synthetic        # offline: simulate 10 days and replay

The live app holds a write lock on the DuckDB file, so stop it first or copy the file.
"""

import argparse
import time

import duckdb as dd
import numpy as np
import pandas as pd

from src.services.candles import CandleStore
from src.services.evaluation import Replayer
from src.services.scoring import ForecastLog, summarize
from src.settings import DUCKDB_PATH, FACTOR_PRODUCTS, PRODUCTS, TARGET


def synthetic_panel(days: int = 10, seed: int = 0) -> pd.DataFrame:
    """Close/high/low panel with the features the model is built for: volatility
    clustering, a time-of-day pattern, peaked fat-tailed shocks, and a common factor."""
    g = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    hours = idx.hour + idx.minute / 60
    tod = 1 + 0.9 * np.exp(-0.5 * ((hours - 14.5) / 1.2) ** 2)  # US-open bump

    def shocks(size):
        # 85% calm, 15% turbulent: more tiny moves *and* more big ones than a normal
        calm = g.random(size) < 0.85
        z = np.where(calm, g.standard_normal(size) * 0.6, g.standard_t(3, size) * 1.6)
        return z / z.std()

    def garch_path(size, omega, alpha, beta):
        z = shocks(size)
        s2 = np.empty(size)
        e = np.empty(size)
        v = omega / (1 - alpha - beta)
        for t in range(size):
            s2[t] = v
            e[t] = np.sqrt(v) * z[t]
            v = omega + alpha * e[t] ** 2 + beta * v
        return e, np.sqrt(s2)

    f, sf = garch_path(n, 0.4, 0.06, 0.92)
    rets, sig = {}, {}
    e, se = garch_path(n, 0.3, 0.05, 0.93)
    rets[TARGET], sig[TARGET] = (0.9 * f + e) * tod, np.hypot(0.9 * sf, se) * tod
    for i, p in enumerate(FACTOR_PRODUCTS):
        e, se = garch_path(n, 0.6, 0.05, 0.9)
        b = 1.0 + 0.2 * i
        rets[p], sig[p] = (b * f + e) * tod, np.hypot(b * sf, se) * tod
    r = pd.DataFrame(rets, index=idx) / 1e4
    s = pd.DataFrame(sig, index=idx) / 1e4
    close = 70_000 * np.exp(r.cumsum())
    open_ = close.shift(1).fillna(close)
    # Intraminute extremes: a Brownian-bridge-sized excursion beyond open/close
    exc = np.abs(g.standard_normal(close.shape)) * s.to_numpy() * 0.6
    high = np.maximum(open_, close) * np.exp(exc)
    low = np.minimum(open_, close) * np.exp(-exc)
    return pd.concat({"open": open_, "high": high, "low": low, "close": close}, axis=1)


def load_synthetic(store: CandleStore, panel: pd.DataFrame) -> None:
    for p in PRODUCTS:
        store.upsert(
            pd.DataFrame(
                {
                    "product": p,
                    "time": panel.index,
                    "open": panel["open"][p].to_numpy(),
                    "high": panel["high"][p].to_numpy(),
                    "low": panel["low"][p].to_numpy(),
                    "close": panel["close"][p].to_numpy(),
                    "volume": 1.0,
                }
            )
        )


def format_table(rows: list[dict]) -> str:
    header = (
        f"{'model':<12}{'h':>4}{'n':>6}{'cov68':>7}{'cov95':>7}{'qlike':>8}"
        f"{'brier':>8}{'DM p':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: (r["horizon"], r["model"])):
        dm = "" if r["dm_p_value"] is None else f"{r['dm_p_value']:.3f}"
        brier = "" if r["brier"] is None else f"{r['brier']:.4f}"
        lines.append(
            f"{r['model']:<12}{r['horizon']:>4}{r['n']:>6}{r['cov68']:>7.2f}{r['cov95']:>7.2f}"
            f"{r['qlike']:>8.3f}{brier:>8}{dm:>7}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--db", default=DUCKDB_PATH)
    parser.add_argument("--source", choices=["live", "replay"], default=None)
    args = parser.parse_args()

    if args.synthetic:
        store = CandleStore(dd.connect(":memory:"))
        load_synthetic(store, synthetic_panel(args.days))
        flog = ForecastLog(store)
        replayer = Replayer(store, flog)
        t0 = time.time()
        pending = replayer.pending_origins()
        print(f"Replaying {len(pending)} origins...")
        for origin in pending:
            replayer.process(origin)
        print(f"done in {time.time() - t0:.0f}s")
        flog.resolve()
        df = flog.read_resolved(args.source)
    else:
        con = dd.connect(args.db, read_only=True)
        df = con.execute(
            "SELECT * FROM forecast_log WHERE realized_bps IS NOT NULL AND isfinite(realized_bps)"
            + (f" AND source = '{args.source}'" if args.source else "")
        ).df()

    summary = summarize(df)
    print(format_table(summary["rows"]))
    if summary["recommended_fit_window"]:
        print(f"\nLowest 30-minute QLIKE: {summary['recommended_fit_window']}")


if __name__ == "__main__":
    main()
