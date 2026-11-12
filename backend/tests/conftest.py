import numpy as np
import pandas as pd
import pytest

from src.settings import FACTOR_PRODUCTS, TARGET

BETA = 0.9
SIGMA_F = 5.0  # bps, factor shock
SIGMA_E = 3.0  # bps, idiosyncratic shock


@pytest.fixture
def synthetic_closes() -> pd.DataFrame:
    """72h of i.i.d. factor-driven prices with known beta and variances."""
    g = np.random.default_rng(42)
    n = 72 * 60
    f = g.standard_normal(n) * SIGMA_F
    rets = {TARGET: BETA * f + g.standard_normal(n) * SIGMA_E}
    for i, p in enumerate(FACTOR_PRODUCTS):
        rets[p] = (1.0 + 0.2 * i) * f + g.standard_normal(n) * 4.0
    idx = pd.date_range("2026-11-01", periods=n, freq="1min", tz="UTC")
    return 60_000 * np.exp(pd.DataFrame(rets, index=idx).cumsum() / 1e4)
