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
