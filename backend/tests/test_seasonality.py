import numpy as np
import pandas as pd

from src.services.seasonality import SeasonalProfile, estimate_profile


def seasonal_returns(days=14, seed=0):
    g = np.random.default_rng(seed)
    idx = pd.date_range("2026-11-01", periods=days * 1440, freq="1min", tz="UTC")
    hours = idx.hour + idx.minute / 60
    true = 1 + 1.0 * np.exp(-0.5 * ((hours - 14.5) / 1.0) ** 2)  # US-open bump
    true = np.where(idx.day_of_week >= 5, true * 0.6, true)  # calmer weekends
    r = pd.DataFrame({"A": g.standard_normal(len(idx)) * 5 * true}, index=idx)
    return r, pd.Series(true, index=idx)


def test_recovers_injected_profile():
    r, true = seasonal_returns()
    prof = estimate_profile(r)
    s = pd.Series(prof.scale_at("A", r.index), index=r.index)
    ratio = s / true
    assert ratio.std() / ratio.mean() < 0.08  # same shape up to a constant
    assert abs(prof.weekend["A"] - 0.6) < 0.05


def test_normalized_to_unit_mean_variance():
    r, _ = seasonal_returns()
    prof = estimate_profile(r)
    s = prof.scale_at("A", r.index)
    assert abs(np.mean(s**2) - 1) < 1e-6


def test_deseasonalize_round_trip():
    r, _ = seasonal_returns()
    prof = estimate_profile(r)
    s = prof.scales(r)
    pd.testing.assert_frame_equal((r / s) * s, r)
    # Deseasonalized |returns| no longer depend on time of day.
    adj = (r / s)["A"].abs()
    by_hour = adj.groupby(adj.index.hour).mean()
    assert by_hour.max() / by_hour.min() < 1.15


def test_flat_below_minimum_history():
    r, _ = seasonal_returns(days=3)
    prof = estimate_profile(r)
    assert not prof.active
    assert np.all(prof.scale_at("A", r.index) == 1.0)
    assert not SeasonalProfile.flat(["A"]).active
