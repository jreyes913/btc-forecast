"""Tests for the GARCH-FHS research harness (synthetic data only, no network)."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.current_method_adapter import LEVELS, make_current_method
from research.data_io import close_time_frame
from research.garch_fhs import prepare_minute_closes, rolling_backtest
from src.settings import FACTOR_PRODUCTS, PRODUCTS, TARGET

FROZEN_SHA256 = "31a665de14c07cf3876ad8f19aa072944aa63beed84216e86da00c02ab6394ab"


def test_frozen_script_is_unchanged():
    src = (Path(__file__).resolve().parents[1] / "research" / "garch_fhs.py").read_bytes()
    assert hashlib.sha256(src).hexdigest() == FROZEN_SHA256


def synthetic_candles(days=4, seed=0):
    """Open-time-indexed (field, product) panel plus BTC candle rows."""
    g = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2026-08-01", periods=n, freq="1min", tz="UTC")
    f = g.standard_normal(n) * 5
    rets = {TARGET: 0.9 * f + g.standard_normal(n) * 3}
    for i, p in enumerate(FACTOR_PRODUCTS):
        rets[p] = (1 + 0.2 * i) * f + g.standard_normal(n) * 4
    close = 60_000 * np.exp(pd.DataFrame(rets, index=idx).cumsum() / 1e4)
    r = np.log(close).diff().abs().fillna(0)
    panel = pd.concat({"close": close, "high": close * np.exp(r * 0.5 + 1e-5),
                       "low": close * np.exp(-r * 0.5 - 1e-5)}, axis=1)
    btc = pd.DataFrame({"time": idx, "close": close[TARGET].to_numpy()})
    return panel, btc


@pytest.fixture(scope="module")
def data():
    return synthetic_candles()


def test_close_time_convention(data):
    _, btc = data
    prices = prepare_minute_closes(close_time_frame(btc), "timestamp", "close")
    # The candle opened at t-1 min closes at t: the value at index t is that close.
    t = pd.Timestamp("2026-08-02 12:00", tz="UTC")
    opened = btc.set_index("time").loc[t - pd.Timedelta(minutes=1), "close"]
    assert prices.loc[t] == opened


def origin_train(btc, origin):
    prices = prepare_minute_closes(close_time_frame(btc), "timestamp", "close")
    return prices.loc[:origin]


def test_adapter_shape_order_and_anchor(data):
    panel, btc = data
    origin = pd.Timestamp("2026-08-04 06:00", tz="UTC")
    train = origin_train(btc, origin)
    out = make_current_method(panel)(train, 60, 300, np.random.default_rng(0))
    assert out.shape == (len(LEVELS), 60) and np.all(np.isfinite(out))
    assert np.all(np.diff(out, axis=0) >= -1e-12)  # quantile rows non-decreasing
    median_h1 = out[len(LEVELS) // 2, 0]
    assert abs(median_h1 - np.log(train.iloc[-1])) < 5e-4  # zero drift, 1 minute


def test_adapter_has_no_look_ahead(data):
    panel, btc = data
    origin = pd.Timestamp("2026-08-04 06:00", tz="UTC")
    train = origin_train(btc, origin)
    clean = make_current_method(panel)(train, 60, 300, np.random.default_rng(1))

    poisoned = panel.copy()
    future = poisoned.index > origin - pd.Timedelta(minutes=1)  # candles not closed by origin
    poisoned.loc[future, :] = 1e9
    poisoned.loc[poisoned.index[future][::7], :] = np.nan
    dirty = make_current_method(poisoned)(train, 60, 300, np.random.default_rng(1))
    np.testing.assert_array_equal(clean, dirty)


def test_backtest_smoke_with_adapter(data):
    panel, btc = data
    prices = prepare_minute_closes(close_time_frame(btc), "timestamp", "close")
    # 3 days of training; a handful of origins; small simulation/bootstrap counts.
    bt = rolling_backtest(
        prices.iloc[: 3 * 1440 + 6 * 60 + 61],
        eval_horizons=(1, 15, 30, 60),
        min_train=3 * 1440, rolling_window=3 * 1440, step=60,
        n_simulations=200, external_models={"current": make_current_method(panel)},
        n_bootstrap=200, seed=1,
    )
    assert list(bt.index) == ["h1", "h15", "h30", "h60"]
    for col in ("current_crps", "verdict_vs_current", "mae_verdict_vs_current",
                "improvement_vs_iid_ci_low", "skill_vs_current"):
        assert col in bt.columns
    assert bt.attrs["total_origins"] >= 5
    assert set(bt.attrs) >= {"fit_failures", "persistence_median", "persistence_share_at_bound"}
    assert set(PRODUCTS) <= set(panel["close"].columns)
