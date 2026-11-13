import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.services.ladder import BANDS, build_ladder, deadline_for
from src.services.quant import fit_and_forecast
from src.settings import FACTOR_PRODUCTS, TARGET

UTC = dt.UTC


@pytest.fixture(scope="module")
def cone():
    # Module-scoped copy of the conftest data so the fit runs once.
    g = np.random.default_rng(42)
    n = 72 * 60
    f = g.standard_normal(n) * 5.0
    rets = {TARGET: 0.9 * f + g.standard_normal(n) * 3.0}
    for i, p in enumerate(FACTOR_PRODUCTS):
        rets[p] = (1.0 + 0.2 * i) * f + g.standard_normal(n) * 4.0
    idx = pd.date_range("2026-11-01", periods=n, freq="1min", tz="UTC")
    closes = 70_000 * np.exp(pd.DataFrame(rets, index=idx).cumsum() / 1e4)
    return fit_and_forecast(closes, TARGET, FACTOR_PRODUCTS, 120, n_sims=2000, seed=1)


def now_after(cone, minutes, seconds=0):
    return cone.last_time.to_pydatetime() + dt.timedelta(minutes=minutes, seconds=seconds)


def test_deadline_rolls_forward_when_under_a_minute_left():
    assert deadline_for(dt.datetime(2026, 11, 13, 17, 20, tzinfo=UTC), "next") == dt.datetime(
        2026, 11, 13, 18, 0, tzinfo=UTC
    )
    late = dt.datetime(2026, 11, 13, 17, 59, 40, tzinfo=UTC)
    assert deadline_for(late, "next") == dt.datetime(2026, 11, 13, 19, 0, tzinfo=UTC)
    assert deadline_for(late, "following") == dt.datetime(2026, 11, 13, 20, 0, tzinfo=UTC)


def test_probabilities_are_monotone_and_complementary(cone):
    lad = build_ladder(cone, now_after(cone, 1, 30), None)
    assert lad["available"]
    pa = np.array([r["p_above"] for r in lad["rows"]])
    pb = np.array([r["p_below"] for r in lad["rows"]])
    assert np.all(np.diff(pa) <= 1e-12)
    np.testing.assert_allclose(pa + pb, 1.0)


def test_strikes_span_both_95_percent_bands_and_tags_each_band_once(cone):
    lad = build_ladder(cone, now_after(cone, 1, 30), None)
    strikes = [r["strike"] for r in lad["rows"]]
    lo95, hi95 = lad["ranges"]["p95"]
    assert strikes[0] < lo95 and strikes[-1] > hi95
    assert all(np.diff(strikes) == 100.0)
    tags = [t for r in lad["rows"] for t in r["bands"]]
    assert sorted(tags) == sorted(BANDS.values())
    # Tails beyond the model's 1-99% range are present and flagged
    assert lad["rows"][0]["beyond_grid"] == "low" and lad["rows"][-1]["beyond_grid"] == "high"


def test_max_prices_follow_the_edge_rule(cone):
    lad = build_ladder(cone, now_after(cone, 1, 30), None, edge=10.0)
    for r in lad["rows"]:
        yes, no = r["p_above"] * 100 - 10, r["p_below"] * 100 - 10
        assert r["max_yes_cents"] == (round(yes, 1) if yes > 0 else None)
        assert r["max_no_cents"] == (round(no, 1) if no > 0 else None)


def test_live_price_anchors_the_table_on_remaining_minutes(cone):
    now = now_after(cone, 1, 30)
    base = build_ladder(cone, now, None)
    up = build_ladder(cone, now, cone.last_close * 1.002)  # price rose 20 bps since origin
    assert up["reference_is_live"] and not base["reference_is_live"]
    p_base = {r["strike"]: r["p_above"] for r in base["rows"]}
    shared = [r for r in up["rows"] if r["strike"] in p_base and 0.05 < r["p_above"] < 0.95]
    assert shared and all(r["p_above"] > p_base[r["strike"]] for r in shared)
    # Live anchoring uses the minutes left from *now*, not from the cone's origin
    assert up["horizon_minutes"] == up["minutes_left"]
    assert base["horizon_minutes"] >= up["horizon_minutes"]


def test_unavailable_beyond_cone_horizon(cone):
    # Origin is 23:59. At 00:30 the "following" deadline (02:00) is 121 minutes past the
    # origin, one more than the 120-minute cone, so it can't be priced without a live bar.
    assert cone.last_time == pd.Timestamp("2026-11-03 23:59", tz="UTC")
    now = now_after(cone, 31)
    lad = build_ladder(cone, now, None, which="following")
    assert not lad["available"] and lad["rows"] == [] and lad["ranges"] is None
    assert build_ladder(cone, now, None, which="next")["available"]
    # With a live price, only the 90 minutes left from now are needed: available again
    assert build_ladder(cone, now, cone.last_close, which="following")["available"]
