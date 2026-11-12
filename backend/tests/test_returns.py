import numpy as np
import pandas as pd

from src.services.quant import BPS, build_return_panel


def _closes(times, prices):
    idx = pd.DatetimeIndex(times, tz="UTC")
    return pd.DataFrame({"A": prices}, index=idx)


def test_short_gap_is_zero_returns_not_one_big_return():
    t0 = pd.Timestamp("2026-11-01 00:00", tz="UTC")
    times = [t0 + pd.Timedelta(minutes=m) for m in (0, 1, 5)]  # minutes 2-4 untraded
    ret = build_return_panel(_closes(times, [100.0, 101.0, 102.0]), max_ffill=5)["A"]

    assert len(ret) == 5  # every minute 1..5 is on the grid
    assert ret.iloc[1:4].eq(0).all()
    np.testing.assert_allclose(ret.iloc[-1], np.log(102 / 101) * BPS)


def test_long_gap_is_dropped():
    t0 = pd.Timestamp("2026-11-01 00:00", tz="UTC")
    times = [t0, t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=20)]
    ret = build_return_panel(_closes(times, [100.0, 101.0, 150.0]), max_ffill=5)["A"]

    # The jump across the 19-minute hole never appears as a 1-minute return.
    assert ret.abs().max() < np.log(101 / 100) * BPS + 1e-9


def test_columns_are_aligned_on_common_minutes():
    t0 = pd.Timestamp("2026-11-01 00:00", tz="UTC")
    idx = pd.date_range(t0, periods=4, freq="1min")
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [1.0, np.nan, np.nan, 2.0]}, index=idx)
    ret = build_return_panel(closes, max_ffill=1)
    # B's 2-minute hole exceeds the limit at minute 2, so minutes 2 and 3 are dropped.
    assert list(ret.index) == [idx[1]]
