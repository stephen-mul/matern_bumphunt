"""Tests for GP utilities and ScanResult."""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from bumphunt.config import BumpHuntConfig
from bumphunt.models import (
    ScanResult,
    _build_kernel,
    _fit_gp,
    _gp_predict,
    _log_to_counts,
)


# ── _log_to_counts ────────────────────────────────────────────────────────────

def test_log_to_counts_zero_sigma():
    """With zero GP uncertainty, mu_counts = exp(mu_log) - 1."""
    mu_log = np.array([0.0, 1.0, 2.0])
    sigma_log = np.zeros(3)
    mu_c, sigma_c = _log_to_counts(mu_log, sigma_log)
    np.testing.assert_allclose(mu_c, np.exp(mu_log) - 1.0, rtol=1e-10)
    np.testing.assert_array_equal(sigma_c, np.zeros(3))


def test_log_to_counts_clipped_to_zero():
    """Negative back-transformed means should be clipped to 0."""
    # Very large negative mu → exp(mu+…) ≈ 0, raw value is exp(…)-1 which can be negative
    mu_log = np.array([-100.0])
    sigma_log = np.array([0.0])
    mu_c, _ = _log_to_counts(mu_log, sigma_log)
    assert mu_c[0] >= 0.0


def test_log_to_counts_lognormal_mean():
    """E[N] = exp(mu + sigma²/2) - 1."""
    mu_log = np.array([1.5])
    sigma_log = np.array([0.5])
    mu_c, _ = _log_to_counts(mu_log, sigma_log)
    expected = np.exp(1.5 + 0.5 * 0.5**2) - 1.0
    np.testing.assert_allclose(mu_c[0], expected, rtol=1e-10)


def test_log_to_counts_delta_method_std():
    """sigma_N ≈ exp(mu) * sigma."""
    mu_log = np.array([1.0])
    sigma_log = np.array([0.3])
    _, sigma_c = _log_to_counts(mu_log, sigma_log)
    expected = np.exp(1.0) * 0.3
    np.testing.assert_allclose(sigma_c[0], expected, rtol=1e-10)


# ── _build_kernel ─────────────────────────────────────────────────────────────

def test_build_kernel_returns_kernel():
    from sklearn.gaussian_process.kernels import Sum, Product
    cfg = BumpHuntConfig()
    k = _build_kernel(cfg)
    # Should be a Sum kernel (Product + WhiteKernel)
    assert isinstance(k, Sum)


# ── _fit_gp and _gp_predict ───────────────────────────────────────────────────

@pytest.fixture
def simple_gp():
    """Fit a GP on a short, simple log-counts sequence."""
    cfg = BumpHuntConfig(length_scale_bounds=(1.0, 50.0))
    x = np.linspace(0, 20, 15)
    y = np.log(100.0 * np.exp(-0.02 * x) + 1.0)
    gp = _fit_gp(x, y, cfg)
    return gp, x, y


def test_fit_gp_returns_regressor(simple_gp):
    gp, _, _ = simple_gp
    assert isinstance(gp, GaussianProcessRegressor)


def test_gp_predict_shape(simple_gp):
    gp, x, _ = simple_gp
    x_pred = np.linspace(0, 20, 30)
    mu, sigma = _gp_predict(gp, x_pred)
    assert mu.shape == (30,)
    assert sigma.shape == (30,)


def test_gp_predict_sigma_positive(simple_gp):
    gp, x, _ = simple_gp
    x_pred = np.linspace(0, 20, 30)
    _, sigma = _gp_predict(gp, x_pred)
    assert np.all(sigma >= 0)


def test_gp_interpolates_training_data(simple_gp):
    """GP predictions at training points should be close to observed values."""
    gp, x_train, y_train = simple_gp
    mu, _ = _gp_predict(gp, x_train)
    np.testing.assert_allclose(mu, y_train, atol=0.5)


# ── ScanResult ────────────────────────────────────────────────────────────────

def test_scan_result_fields():
    n = 10
    res = ScanResult(
        mask_half_width=2,
        positions=np.arange(n, dtype=float),
        local_sigma=np.zeros(n),
        excess_counts=np.zeros(n),
        predicted_bkg=np.zeros(n),
        predicted_unc=np.zeros(n),
        stitched_bkg=np.zeros(n),
    )
    assert res.mask_half_width == 2
    assert len(res.positions) == n
