import numpy as np
import pytest
from arch import arch_model

from src.services.garch import fit_garch_x, simulate_fhs

TRUE = {"omega": 0.5, "alpha": 0.03, "gamma": 0.04, "beta": 0.80, "delta": 0.10}


def simulate_garch_x(n=12_000, seed=1, shock=None):
    g = np.random.default_rng(seed)
    shock = shock or (lambda: g.standard_t(5) / np.sqrt(5 / 3))
    p = TRUE
    s2, e_prev, x_prev = 10.0, 0.0, 10.0
    ys, xs = np.empty(n), np.empty(n)
    for t in range(n):
        s2 = (p["omega"] + (p["alpha"] + p["gamma"] * (e_prev < 0)) * e_prev**2
              + p["beta"] * s2 + p["delta"] * x_prev)
        e = np.sqrt(s2) * shock()
        x = s2 * g.chisquare(4) / 4  # noisy, unbiased proxy of the current variance
        ys[t], xs[t], e_prev, x_prev = e, x, e, x
    return ys, xs


@pytest.fixture(scope="module")
def garch_x_data():
    return simulate_garch_x()


def test_recovers_known_parameters(garch_x_data):
    y, x = garch_x_data
    fit = fit_garch_x(y, x)
    assert fit.converged
    for name, tol in [("alpha", 0.02), ("gamma", 0.03), ("delta", 0.05)]:
        assert abs(fit.params[name] - TRUE[name]) < tol, name
    total_true = TRUE["alpha"] + TRUE["gamma"] / 2 + TRUE["beta"] + TRUE["delta"]
    assert abs(fit.persistence() - total_true) < 0.05


def test_matches_arch_without_exogenous_term(garch_x_data):
    y, _ = garch_x_data
    ours = fit_garch_x(y, None, ar=True)
    ref = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="normal",
                     rescale=False).fit(disp="off")
    assert abs(ours.params["alpha"] - ref.params["alpha[1]"]) < 0.01
    assert abs(ours.params["gamma"] - ref.params["gamma[1]"]) < 0.01
    assert abs(ours.params["beta"] - ref.params["beta[1]"]) < 0.02
    assert abs(ours.loglik - ref.loglikelihood) / abs(ref.loglikelihood) < 1e-3


def test_noise_regressor_gets_near_zero_weight(garch_x_data):
    y, _ = garch_x_data
    noise = np.random.default_rng(3).chisquare(4, len(y)) * np.var(y) / 4
    assert fit_garch_x(y, noise).params["delta"] < 0.02


def test_fhs_is_deterministic_and_matches_model_variance(garch_x_data):
    y, x = garch_x_data
    fit = fit_garch_x(y, x)
    a = simulate_fhs([fit], 30, 4000, np.random.default_rng(0))[0]
    b = simulate_fhs([fit], 30, 4000, np.random.default_rng(0))[0]
    np.testing.assert_array_equal(a, b)
    assert abs(a[:, 0].var() / fit.next_sigma2() - 1) < 0.1


def test_fhs_reproduces_peaked_fat_tailed_shocks():
    """Calm/turbulent mixture: too many tiny moves and too many big ones for a t."""
    g = np.random.default_rng(5)

    def shock():
        z = g.standard_normal() * 0.6 if g.random() < 0.85 else g.standard_t(3) * 1.6
        return z / 1.03  # roughly unit variance

    y, x = simulate_garch_x(n=15_000, seed=5, shock=shock)
    fit = fit_garch_x(y, x)
    sims = simulate_fhs([fit], 1, 20_000, np.random.default_rng(1))[0][:, 0]
    z_sim = sims / np.sqrt(fit.next_sigma2())
    inside_1sd = np.mean(np.abs(z_sim) < 1)
    # A unit-variance normal puts 68% inside +/-1; this mixture puts far more there.
    assert inside_1sd > 0.75
    assert abs(inside_1sd - np.mean(np.abs(fit.z) < 1)) < 0.02


def test_joint_resampling_keeps_cross_component_dependence():
    g = np.random.default_rng(7)
    common = g.standard_t(3, 8000)
    a = common + g.standard_normal(8000) * 0.5
    b = common + g.standard_normal(8000) * 0.5
    fa, fb = fit_garch_x(a, None), fit_garch_x(b, None)
    sa, sb = simulate_fhs([fa, fb], 1, 20_000, np.random.default_rng(2))
    assert np.corrcoef(sa[:, 0], sb[:, 0])[0, 1] > 0.6
