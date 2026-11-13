import numpy as np

from src.services.quant import build_return_panel, fit_and_forecast, fit_factor
from src.settings import FACTOR_PRODUCTS, TARGET

from .conftest import BETA, SIGMA_E, SIGMA_F

H = 60


def test_bands_include_factor_risk(synthetic_closes):
    res = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, H, n_sims=2000, seed=1)

    total_sd = np.sqrt(BETA**2 * SIGMA_F**2 + SIGMA_E**2) * np.sqrt(H)
    idio_sd = SIGMA_E * np.sqrt(H)
    half_width = (res.cum_return_quantiles["p84"][-1] - res.cum_return_quantiles["p16"][-1]) / 2

    assert abs(half_width / total_sd - 1) < 0.15
    assert half_width > 1.5 * idio_sd  # the old residual-only cone would be ~idio_sd
    assert 0.5 < res.factor_variance_share < 0.8


def test_capm_recovers_known_structure(synthetic_closes):
    res = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 5, n_sims=200, seed=1)
    assert res.resid_converged and res.factor_converged
    assert res.capm_r2 > 0.5
    assert abs(res.resid_std_bps - SIGMA_E) < 0.5


def test_pca_sign_is_fixed(synthetic_closes):
    returns = build_return_panel(synthetic_closes)[FACTOR_PRODUCTS]
    fit_pos = fit_factor(returns)
    fit_neg = fit_factor(-returns)
    assert fit_pos.loadings[FACTOR_PRODUCTS[0]] > 0
    assert fit_neg.loadings[FACTOR_PRODUCTS[0]] > 0
    assert np.corrcoef(fit_pos.pc1, returns.mean(axis=1))[0, 1] > 0.9


def test_forecast_is_deterministic_per_snapshot(synthetic_closes):
    a = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 10, n_sims=200)
    b = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 10, n_sims=200)
    np.testing.assert_array_equal(a.price_quantiles["p50"], b.price_quantiles["p50"])


def test_price_bands_are_ordered(synthetic_closes):
    res = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 30, n_sims=500, seed=3)
    q = res.price_quantiles
    assert np.all(q["p025"] <= q["p16"]) and np.all(q["p16"] <= q["p50"])
    assert np.all(q["p50"] <= q["p84"]) and np.all(q["p84"] <= q["p975"])


def test_zero_drift_even_with_strong_in_sample_alpha(synthetic_closes):
    trend = np.exp(np.arange(len(synthetic_closes)) * 2e-5)  # +0.2 bps/min BTC-only drift
    drifted = synthetic_closes.copy()
    drifted[TARGET] = drifted[TARGET] * trend
    res = fit_and_forecast(drifted, TARGET, FACTOR_PRODUCTS, H, n_sims=4000, seed=1)
    assert res.alpha_bps > 0.1  # the drift is estimated and reported...
    half_width = (res.cum_return_quantiles["p84"][-1] - res.cum_return_quantiles["p16"][-1]) / 2
    # ...but not simulated: the median stays near the origin (0.2 * 60 = 12 bps if it were)
    assert abs(res.cum_return_quantiles["p50"][-1]) < 0.1 * half_width


def test_range_term_is_used_when_high_low_given(synthetic_closes):
    import pandas as pd

    r = np.log(synthetic_closes).diff().fillna(0).abs()
    panel = pd.concat(
        {"close": synthetic_closes, "high": synthetic_closes * np.exp(r * 0.5 + 1e-5),
         "low": synthetic_closes * np.exp(-r * 0.5 - 1e-5)}, axis=1)
    res = fit_and_forecast(panel, TARGET, FACTOR_PRODUCTS, 15, n_sims=500, seed=1)
    assert "delta" in res.resid_garch_params and "delta" in res.factor_garch_params
    assert res.resid_converged and res.factor_converged


def test_seasonal_profile_widens_forecasts_at_busy_times(synthetic_closes):
    from src.services.seasonality import SeasonalProfile

    busy_hour = (synthetic_closes.index[-1].hour + 1) % 24
    tod = np.ones(96)
    tod[busy_hour * 4 : busy_hour * 4 + 4] = 3.0
    cols = [TARGET, *FACTOR_PRODUCTS]
    prof = SeasonalProfile(15, {c: tod for c in cols}, {c: 1.0 for c in cols}, active=True)
    flat = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 120, n_sims=2000, seed=1)
    seas = fit_and_forecast(synthetic_closes, TARGET, FACTOR_PRODUCTS, 120, n_sims=2000,
                            seed=1, profile=prof)
    # Future minutes 0-59 fall in the busy hour (variance x9 per minute), 70-110 don't.
    def growth_per_min(res, a, b):
        return (res.cum_return_var[b] - res.cum_return_var[a]) / (b - a)

    assert seas.seasonality_active
    busy_vs_quiet = growth_per_min(seas, 0, 50) / growth_per_min(seas, 70, 110)
    flat_ratio = growth_per_min(flat, 0, 50) / growth_per_min(flat, 70, 110)
    assert busy_vs_quiet > 5 * flat_ratio
