"""Run the rolling-origin backtest and print a comparison table.

Usage (from backend/):
    python -m scripts.backtest                    # uses data/candles.duckdb
    python -m scripts.backtest --synthetic        # offline demo on simulated data
"""

import argparse

import duckdb as dd
import numpy as np
import pandas as pd

from src.services.backtest import format_table, run_backtest
from src.services.candles import CandleStore
from src.settings import DUCKDB_PATH, FACTOR_PRODUCTS, PRODUCTS, TARGET


def synthetic_closes(n_minutes: int = 72 * 60, seed: int = 0) -> pd.DataFrame:
    """Factor-driven prices with GARCH-like volatility clustering, for offline runs."""
    g = np.random.default_rng(seed)
    sig = np.empty(n_minutes)
    f = np.empty(n_minutes)
    s2 = 25.0
    for t in range(n_minutes):
        f[t] = np.sqrt(s2) * g.standard_t(6) / np.sqrt(1.5)
        s2 = 1.0 + 0.08 * f[t] ** 2 + 0.88 * s2
        sig[t] = np.sqrt(s2)
    idx = pd.date_range("2026-01-01", periods=n_minutes, freq="1min", tz="UTC")
    rets = {TARGET: 0.9 * f + 3 * g.standard_normal(n_minutes)}
    for i, p in enumerate(FACTOR_PRODUCTS):
        rets[p] = (1.0 + 0.2 * i) * f + 4 * g.standard_normal(n_minutes)
    return 100 * np.exp(pd.DataFrame(rets, index=idx).cumsum() / 1e4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--db", default=DUCKDB_PATH)
    args = parser.parse_args()

    if args.synthetic:
        closes = synthetic_closes()
    else:
        closes = CandleStore(dd.connect(args.db, read_only=True)).read_closes(PRODUCTS)
    print(f"{len(closes)} bars, {closes.index.min()} -> {closes.index.max()}")
    print(format_table(run_backtest(closes, TARGET, FACTOR_PRODUCTS)))


if __name__ == "__main__":
    main()
