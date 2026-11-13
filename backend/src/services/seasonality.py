"""Intraday (time-of-day) volatility profile.

Minute-level crypto volatility follows a daily rhythm: spikes around the US open and
scheduled data releases, quiet stretches overnight, and calmer weekends. GARCH has no
notion of clock time, so it would misread every predictable spike as a "shock". The
standard fix is to divide returns by a deterministic profile s(t) before fitting and to
multiply forecasts back by s(t) for the future minutes they cover.
"""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pandas as pd

MIN_DAYS = 7
MINUTES_PER_DAY = 1440


@dc.dataclass(frozen=True)
class SeasonalProfile:
    """Per-series multipliers: ``tod[col]`` per time-of-day bucket, ``weekend[col]``."""

    bucket_minutes: int
    tod: dict[str, np.ndarray]
    weekend: dict[str, float]
    active: bool

    @classmethod
    def flat(cls, columns, bucket_minutes: int = 15) -> SeasonalProfile:
        n = MINUTES_PER_DAY // bucket_minutes
        return cls(
            bucket_minutes=bucket_minutes,
            tod={c: np.ones(n) for c in columns},
            weekend={c: 1.0 for c in columns},
            active=False,
        )

    def scale_at(self, column: str, times: pd.DatetimeIndex) -> np.ndarray:
        """s(t) for ``column`` at each UTC timestamp in ``times``."""
        times = pd.DatetimeIndex(times)
        bucket = (times.hour * 60 + times.minute) // self.bucket_minutes
        s = self.tod[column][np.asarray(bucket)]
        weekend = np.asarray(times.day_of_week >= 5)
        return np.where(weekend, s * self.weekend[column], s)

    def scales(self, frame: pd.DataFrame) -> pd.DataFrame:
        """s(t) for every column of ``frame`` on its own index."""
        return pd.DataFrame(
            {c: self.scale_at(c, frame.index) for c in frame.columns}, index=frame.index
        )


def _circular_smooth(values: np.ndarray, width: float) -> np.ndarray:
    n = len(values)
    offsets = np.arange(-3 * int(np.ceil(width)), 3 * int(np.ceil(width)) + 1)
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[-len(offsets) :], values, values[: len(offsets)]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[len(offsets) : len(offsets) + n]


def estimate_profile(
    returns: pd.DataFrame, bucket_minutes: int = 15, smooth_buckets: float = 1.0
) -> SeasonalProfile:
    """Estimate s(t) per column from (bps) returns indexed by UTC minute.

    Scale per bucket is the mean absolute return after winsorizing at the 99.5th
    percentile, so a few jumps can't dominate a bucket. The profile is normalized so
    the average of s(t)^2 over the sample is 1, leaving the overall volatility level
    to GARCH.
    """
    span_days = (returns.index.max() - returns.index.min()) / pd.Timedelta(days=1)
    if len(returns) == 0 or span_days < MIN_DAYS:
        return SeasonalProfile.flat(returns.columns, bucket_minutes)

    idx = pd.DatetimeIndex(returns.index)
    bucket = np.asarray((idx.hour * 60 + idx.minute) // bucket_minutes)
    weekend = np.asarray(idx.day_of_week >= 5)
    n_buckets = MINUTES_PER_DAY // bucket_minutes

    tod, wk = {}, {}
    for col in returns.columns:
        a = returns[col].abs().to_numpy()
        a = np.minimum(a, np.nanquantile(a, 0.995))
        weekday_mask = ~weekend
        by_bucket = np.array(
            [
                a[weekday_mask & (bucket == b)].mean()
                if (weekday_mask & (bucket == b)).any()
                else np.nan
                for b in range(n_buckets)
            ]
        )
        by_bucket = pd.Series(by_bucket).interpolate(limit_direction="both").to_numpy()
        prof = _circular_smooth(by_bucket, smooth_buckets)
        prof = np.maximum(prof, 1e-6)

        # Weekend multiplier: weekend scale relative to the weekday profile at the same times.
        if weekend.any():
            ratio = a[weekend].mean() / prof[bucket[weekend]].mean()
        else:
            ratio = 1.0

        s = np.where(weekend, prof[bucket] * ratio, prof[bucket])
        norm = np.sqrt(np.mean(s**2))
        tod[col] = prof / norm
        wk[col] = float(ratio)

    return SeasonalProfile(bucket_minutes=bucket_minutes, tod=tod, weekend=wk, active=True)
