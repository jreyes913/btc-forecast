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
    ensemble_frames,
    pinball_loss,
    quantile_cdf,
    quantile_density,
    sequential_weights,
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


# --- Adaptive (Bayesian / loss-weighted) ensembles ------------------------------------


def test_quantile_density_matches_gaussian_and_integrates_to_one():
    sd = 20.0
    q = (norm.ppf(Q_GRID) * sd)[None, :]
    x = np.linspace(-200, 200, 20_001)
    dens = quantile_density(np.repeat(q, len(x), axis=0), x)
    assert np.all(dens > 0)
    assert abs(np.trapezoid(dens, x) - 1) < 0.01
    inside = np.abs(x) < norm.ppf(0.9) * sd
    rel = dens[inside] / norm.pdf(x[inside], scale=sd)
    # Piecewise-constant on a 13-point grid: the p25-p75 segments are wide, so the
    # approximation is off by up to ~20% inside them. Good enough to rank forecasts.
    assert np.all(np.abs(rel - 1) < 0.25)


def test_log_density_prefers_the_right_scale():
    r = np.random.default_rng(1).standard_normal(5000) * 20
    def mean_log_score(sd):
        q = np.tile(norm.ppf(Q_GRID) * sd, (len(r), 1))
        return np.log(quantile_density(q, r)).mean()
    assert mean_log_score(20) > max(mean_log_score(12), mean_log_score(35))


def test_pinball_loss_is_lower_for_the_right_scale():
    r = np.random.default_rng(0).standard_normal(5000) * 20
    right = np.tile(norm.ppf(Q_GRID) * 20, (len(r), 1))
    wrong = np.tile(norm.ppf(Q_GRID) * 40, (len(r), 1))
    assert pinball_loss(right, r).mean() < pinball_loss(wrong, r).mean()


def test_sequential_weights_are_causal():
    origins = (T0 + pd.to_timedelta(np.arange(100) * 5, "min")).to_numpy(dtype="datetime64[ns]")
    scores = np.zeros((100, 3))
    scores[50:, 0] = 5.0  # member 0 becomes far better from origin 50 onwards
    w, latest = sequential_weights(origins, horizon=60, scores=scores, forgetting=0.99)
    np.testing.assert_allclose(w[0], 1 / 3)
    # Origin 50's outcome resolves 60 minutes later = origin 62. Before that, no change.
    np.testing.assert_allclose(w[:62], 1 / 3)
    assert w[63, 0] > 0.9 and latest[0] > 0.99


def test_sequential_weights_forget_after_regime_switch():
    n = 600
    origins = (T0 + pd.to_timedelta(np.arange(n), "min")).to_numpy(dtype="datetime64[ns]")
    scores = np.zeros((n, 3))
    scores[:300, 0] = 0.1  # member 0 better first...
    scores[300:, 1] = 0.1  # ...then member 1
    w, _ = sequential_weights(origins, horizon=1, scores=scores, forgetting=0.99)
    assert w[299].argmax() == 0
    assert w[300 + 200].argmax() == 1  # recovers within ~1/(1-forgetting) updates


def adaptive_frame(sds, n=3000, seed=11, horizon=15):
    """Members factor/gjr/ewma with Gaussian bands of the given widths, same outcomes."""
    return pd.concat(
        [gaussian_frame(n=n, sd_true=20.0, sd_model=sd, model=m, seed=seed, horizon=horizon)
         for m, sd in zip(("factor", "gjr", "ewma"), sds, strict=True)]
    )


def test_adaptive_weights_favor_the_better_member():
    _, latest = ensemble_frames(adaptive_frame((12.0, 20.0, 35.0)))
    for entry in latest:
        w = entry["weights"]
        assert max(w, key=w.get) == "gjr", entry  # the correctly scaled member


def test_adaptive_weights_stay_near_equal_for_identical_members():
    _, latest = ensemble_frames(adaptive_frame((20.0, 20.0, 20.0)))
    for entry in latest:
        np.testing.assert_allclose(list(entry["weights"].values()), 1 / 3, atol=1e-9)


def test_adaptive_ensembles_beat_equal_weights_when_one_member_is_right():
    rows = {r["model"]: r for r in summarize(adaptive_frame((12.0, 20.0, 35.0)))["rows"]}
    for name in ("ensemble_bma", "ensemble_loss"):
        assert rows[name]["brier"] <= rows["ensemble"]["brier"]
        assert abs(rows[name]["cov68"] - 0.68) < 0.04
        assert rows[name]["dm_p_value"] is not None


def test_summary_reports_latest_weights():
    s = summarize(adaptive_frame((15.0, 20.0, 25.0), n=200))
    variants = {(e["variant"], e["horizon"]) for e in s["ensemble_weights"]}
    assert variants == {("ensemble_bma", 15), ("ensemble_loss", 15)}
    for e in s["ensemble_weights"]:
        assert abs(sum(e["weights"].values()) - 1) < 1e-9 and e["updates"] == 200
