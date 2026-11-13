import duckdb as dd
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.services.candles import CandleStore
from src.services.quant import Q_GRID
from src.services.scoring import (
    ForecastLog,
    ForecastRecord,
    add_ensemble,
    diebold_mariano,
    quantile_cdf,
    strike_probabilities,
    summarize,
)

T0 = pd.Timestamp("2026-11-12 10:00", tz="UTC")


def test_quantile_cdf_is_monotone_and_exact_on_grid():
    q = np.tile(norm.ppf(Q_GRID) * 20, (1, 1))
    x = np.linspace(-80, 80, 400)[None, :]
    cdf = quantile_cdf(q, x)[0]
    assert np.all(np.diff(cdf) >= 0)
    np.testing.assert_allclose(quantile_cdf(q, q)[0], Q_GRID, atol=1e-12)


def gaussian_frame(n=3000, sd_true=20.0, sd_model=20.0, seed=0, model="factor", horizon=15):
    """Forecasts with Gaussian quantiles of ``sd_model`` scored on N(0, sd_true) outcomes."""
    g = np.random.default_rng(seed)
    q = norm.ppf(Q_GRID) * sd_model
    return pd.DataFrame(
        {
            "origin_time": T0 + pd.to_timedelta(np.arange(n) * 30, "min"),
            "model": model,
            "horizon": horizon,
            "source": "replay",
            "origin_price": 70_000.0,
            "var_bps2": sd_model**2,
            "q": [q] * n,
            "realized_bps": g.standard_normal(n) * sd_true,
        }
    )


def test_calibrated_forecasts_score_as_calibrated():
    s = summarize(gaussian_frame())
    row = s["rows"][0]
    assert abs(row["cov68"] - 0.68) < 0.03 and abs(row["cov95"] - 0.95) < 0.015
    assert row["n_strikes"] > 0
    bins = s["calibration"][0]["bins"]
    for b in bins:
        if b["n"] > 200:
            assert abs(b["obs_freq"] - b["p_mean"]) < 0.06


def test_too_wide_bands_over_cover():
    row = summarize(gaussian_frame(sd_model=30.0))["rows"][0]
    assert row["cov68"] > 0.8


def test_dm_prefers_the_right_model():
    good = gaussian_frame(model="factor")
    bad = gaussian_frame(model="ewma", sd_model=35.0)
    rows = {r["model"]: r for r in summarize(pd.concat([good, bad]))["rows"]}
    assert rows["ewma"]["dm_mean_diff"] > 0  # baseline loss is higher
    assert rows["ewma"]["dm_p_value"] < 0.01
    assert rows["factor"]["brier"] < rows["ewma"]["brier"]


def test_dm_identical_losses_not_significant():
    d = np.random.default_rng(1).standard_normal(2000)
    _, p = diebold_mariano(d, lag=2)
    assert p > 0.01


def test_strikes_lie_within_forecast_range():
    q = (norm.ppf(Q_GRID) * 20)[None, :]
    p, o = strike_probabilities(q, np.array([70_050.0]), np.array([0.0]))
    assert len(p) > 0 and np.all((p > 0) & (p < 1))


@pytest.fixture
def store_with_prices():
    store = CandleStore(dd.connect(":memory:"))
    times = pd.date_range(T0, periods=120, freq="1min")
    store.upsert(pd.DataFrame({"product": "BTC-USD", "time": times, "open": 1.0,
                               "high": 1.0, "low": 1.0, "close": 70_000.0 + np.arange(120),
                               "volume": 1.0}))
    return store


def record(horizon, origin=T0, model="factor"):
    return ForecastRecord(origin, model, horizon, "live", 70_000.0, 100.0,
                          norm.ppf(Q_GRID) * 10)


def test_resolver_fills_only_matured_rows(store_with_prices):
    flog = ForecastLog(store_with_prices)
    flog.insert([record(15), record(60), record(200)])
    assert flog.resolve() == 2
    resolved = flog.read_resolved().set_index("horizon")
    assert set(resolved.index) == {15, 60}
    np.testing.assert_allclose(resolved.loc[15, "realized_bps"], np.log(70_015 / 70_000) * 1e4)


def test_insert_is_idempotent(store_with_prices):
    flog = ForecastLog(store_with_prices)
    flog.insert([record(15)])
    flog.resolve()
    flog.insert([record(15)])  # must not wipe the realized value
    assert len(flog.read_resolved()) == 1
    assert flog.origins("live") == {T0}


def test_ensemble_averages_quantiles_and_sds():
    members = pd.concat(
        [gaussian_frame(n=50, sd_model=sd, model=m)
         for m, sd in (("factor", 10.0), ("gjr", 20.0), ("ewma", 30.0))]
    )
    out = add_ensemble(members)
    ens = out.loc[out["model"] == "ensemble"]
    assert len(ens) == 50
    np.testing.assert_allclose(ens["var_bps2"], 20.0**2)  # (mean of 10, 20, 30)^2
    np.testing.assert_allclose(np.vstack(ens["q"].to_numpy())[0], norm.ppf(Q_GRID) * 20.0)
    # Outcomes and prices carry over from the members at the same origin
    first = members.loc[members["model"] == "factor"].iloc[0]
    assert ens.iloc[0]["realized_bps"] == first["realized_bps"]


def test_ensemble_skips_origins_missing_a_member():
    factor = gaussian_frame(n=20, model="factor")
    gjr = gaussian_frame(n=20, model="gjr")
    ewma = gaussian_frame(n=20, model="ewma").iloc[:15]  # 5 origins lack EWMA
    out = add_ensemble(pd.concat([factor, gjr, ewma]))
    assert (out["model"] == "ensemble").sum() == 15
    assert add_ensemble(factor).equals(factor)  # nothing to combine


def test_ensemble_of_wide_and_narrow_members_is_better_calibrated():
    # Same outcomes for every member: only the forecast widths differ.
    frames = [gaussian_frame(n=4000, sd_true=20.0, sd_model=sd, model=m, seed=7)
              for m, sd in (("factor", 12.0), ("gjr", 20.0), ("ewma", 28.0))]
    rows = {r["model"]: r for r in summarize(pd.concat(frames))["rows"]}
    ens, narrow, wide = rows["ensemble"], rows["factor"], rows["ewma"]
    assert abs(ens["cov68"] - 0.68) < abs(narrow["cov68"] - 0.68)
    assert abs(ens["cov68"] - 0.68) < abs(wide["cov68"] - 0.68)
    assert ens["brier"] < narrow["brier"] and ens["brier"] < wide["brier"]
    assert ens["dm_p_value"] is not None and ens["dm_mean_diff"] < 0  # beats the factor model
