"""Tests for the look-elsewhere effect correction."""

import numpy as np
import pytest
from scipy.stats import norm

from bumphunt.config import BumpHuntConfig
from bumphunt.data import generate_synthetic_spectrum
from bumphunt.lee import count_upcrossings, gross_vitells_correction, lee_toy_mc


# ── count_upcrossings ─────────────────────────────────────────────────────────

def test_upcrossings_simple():
    curve = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
    assert count_upcrossings(curve, 1.0) == 2


def test_upcrossings_single_peak():
    curve = np.array([0.0, 0.5, 2.0, 0.5, 0.0])
    assert count_upcrossings(curve, 1.0) == 1


def test_upcrossings_none():
    curve = np.array([0.0, 0.5, 0.8])
    assert count_upcrossings(curve, 1.0) == 0


def test_upcrossings_all_above():
    """If the curve never dips below the level there are no upcrossings."""
    curve = np.array([2.0, 3.0, 4.0])
    assert count_upcrossings(curve, 1.0) == 0


def test_upcrossings_ignores_nans():
    curve = np.array([np.nan, 0.0, 2.0, 0.0, np.nan])
    # NaNs stripped → [0, 2, 0] → one upcrossing
    assert count_upcrossings(curve, 1.0) == 1


def test_upcrossings_short_curve():
    assert count_upcrossings(np.array([1.5]), 1.0) == 0
    assert count_upcrossings(np.array([]), 1.0) == 0
    assert count_upcrossings(np.array([np.nan]), 1.0) == 0


def test_upcrossings_level_at_boundary():
    """Values exactly equal to the level are NOT above it (strict >)."""
    curve = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    # 1.0 is not > 1.0, so the transition 0→1 is not an upcrossing,
    # but 1→2 is (0<2, 1 not above).  Let's trace carefully:
    # above = [F, F, T, F, F] → upcrossings where ~above[:-1] & above[1:]
    # pairs: (F,F),(F,T),(T,F),(F,F) → one upcrossing at position (1→2)
    assert count_upcrossings(curve, 1.0) == 1


# ── gross_vitells_correction ──────────────────────────────────────────────────

def test_gv_zero_upcrossings_equals_local():
    """With no upcrossings, the global p-value equals the local p-value."""
    # Flat curve below reference level → 0 upcrossings
    curve = np.zeros(50)
    t0 = 5.0
    p_global, sigma_global, n_up = gross_vitells_correction(t0, curve, reference_level=1.0)
    p_local = norm.sf(t0)
    assert n_up == 0
    assert abs(p_global - p_local) < 1e-12


def test_gv_correction_increases_p_value():
    """The global p-value should be >= the local p-value."""
    # Curve with several upcrossings
    curve = np.tile([0.0, 2.0], 25).astype(float)
    t0 = 5.0
    p_global, sigma_global, n_up = gross_vitells_correction(t0, curve, reference_level=1.0)
    p_local = norm.sf(t0)
    assert p_global >= p_local
    assert n_up > 0


def test_gv_correction_reduces_sigma():
    """Global significance should be ≤ local significance."""
    curve = np.tile([0.0, 2.0], 25).astype(float)
    t0 = 5.0
    p_global, sigma_global, n_up = gross_vitells_correction(t0, curve, reference_level=1.0)
    assert sigma_global <= t0


def test_gv_p_global_capped_at_one():
    """p_global must never exceed 1.0."""
    # Absurdly many upcrossings
    curve = np.tile([0.0, 2.0], 5000).astype(float)
    t0 = 0.1  # very low significance → huge correction possible
    p_global, _, _ = gross_vitells_correction(t0, curve, reference_level=0.05)
    assert p_global <= 1.0


def test_gv_p_global_non_negative():
    curve = np.tile([0.0, 2.0], 10).astype(float)
    t0 = 3.0
    p_global, sigma_global, _ = gross_vitells_correction(t0, curve)
    assert p_global >= 0.0
    assert sigma_global >= 0.0


def test_gv_sigma_equiv_matches_p_global():
    """sigma_global should satisfy Φ(-sigma_global) ≈ p_global."""
    curve = np.tile([0.0, 2.0], 15).astype(float)
    t0 = 4.0
    p_global, sigma_global, _ = gross_vitells_correction(t0, curve)
    # Allow small discrepancy when p_global = 1 (sigma = 0 by floor)
    if p_global < 1.0:
        assert abs(norm.sf(sigma_global) - p_global) < 1e-6


def test_gv_more_upcrossings_means_lower_sigma():
    """More upcrossings → stronger correction → lower global sigma."""
    t0 = 5.0
    few_crossings = np.array([0.0, 2.0] + [0.0] * 20, dtype=float)
    many_crossings = np.tile([0.0, 2.0], 15).astype(float)
    _, sigma_few, _ = gross_vitells_correction(t0, few_crossings)
    _, sigma_many, _ = gross_vitells_correction(t0, many_crossings)
    assert sigma_many < sigma_few


def test_gv_handles_all_nan_curve():
    """All-NaN curve → 0 upcrossings → global p-value = local p-value."""
    curve = np.full(20, np.nan)
    t0 = 4.0
    p_global, _, n_up = gross_vitells_correction(t0, curve)
    assert n_up == 0
    assert abs(p_global - norm.sf(t0)) < 1e-12


# ── lee_toy_mc ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_cfg_lee():
    """Fast config for LEE toy MC tests."""
    return BumpHuntConfig(
        window_half_width=12,
        mask_half_widths=[1, 2],
        length_scale_bounds=(3.0, 20.0),
        min_sideband_bins=8,
        local_sigma_threshold=3.0,
    )


@pytest.fixture
def small_spectrum_lee():
    return generate_synthetic_spectrum(
        n_bins=60,
        signal_pos=30.0,
        signal_width=2.0,
        signal_events=100.0,
        seed=99,
    )


def test_toy_mc_returns_correct_types(small_cfg_lee, small_spectrum_lee):
    x, counts = small_spectrum_lee
    p_global, sigma_global, toy_sigmas = lee_toy_mc(
        x, counts, small_cfg_lee, n_toys=3, seed=0
    )
    assert isinstance(p_global, float)
    assert isinstance(sigma_global, float)
    assert toy_sigmas.shape == (3,)


def test_toy_mc_p_in_unit_interval(small_cfg_lee, small_spectrum_lee):
    x, counts = small_spectrum_lee
    p_global, _, _ = lee_toy_mc(x, counts, small_cfg_lee, n_toys=5, seed=1)
    assert 0.0 < p_global <= 1.0


def test_toy_mc_sigma_non_negative(small_cfg_lee, small_spectrum_lee):
    x, counts = small_spectrum_lee
    _, sigma_global, _ = lee_toy_mc(x, counts, small_cfg_lee, n_toys=5, seed=2)
    assert sigma_global >= 0.0


def test_toy_mc_global_sigma_le_local(small_cfg_lee, small_spectrum_lee):
    """Global significance should be ≤ local significance."""
    from bumphunt.scan import max_local_significance, run_bumphunt
    x, counts = small_spectrum_lee
    results = run_bumphunt(x, counts, cfg=small_cfg_lee)
    _, max_sig, _ = max_local_significance(results)
    t0 = float(np.nanmax(max_sig))
    _, sigma_global, _ = lee_toy_mc(
        x, counts, small_cfg_lee, n_toys=5, observed_max_sigma=t0, seed=3
    )
    assert sigma_global <= t0 + 0.01  # small tolerance for stochastic floor


def test_toy_mc_reproducible(small_cfg_lee, small_spectrum_lee):
    x, counts = small_spectrum_lee
    _, _, sigmas1 = lee_toy_mc(x, counts, small_cfg_lee, n_toys=3, seed=42)
    _, _, sigmas2 = lee_toy_mc(x, counts, small_cfg_lee, n_toys=3, seed=42)
    # Same seed → same Poisson draws → same toy spectra → identical or near-identical
    # max-significances (tiny differences can arise from GP optimizer non-determinism)
    np.testing.assert_allclose(sigmas1, sigmas2, rtol=1e-4)


def test_toy_mc_different_seeds_differ(small_cfg_lee, small_spectrum_lee):
    x, counts = small_spectrum_lee
    _, _, sigmas1 = lee_toy_mc(x, counts, small_cfg_lee, n_toys=3, seed=10)
    _, _, sigmas2 = lee_toy_mc(x, counts, small_cfg_lee, n_toys=3, seed=11)
    # With different seeds, max sigmas should differ (extremely unlikely to be identical)
    assert not np.array_equal(sigmas1, sigmas2)


def test_toy_mc_precomputed_observed_sigma(small_cfg_lee, small_spectrum_lee):
    """Passing observed_max_sigma explicitly should give same result as computing internally."""
    from bumphunt.scan import max_local_significance, run_bumphunt
    x, counts = small_spectrum_lee
    results = run_bumphunt(x, counts, cfg=small_cfg_lee)
    _, max_sig, _ = max_local_significance(results)
    t0 = float(np.nanmax(max_sig))

    p1, s1, _ = lee_toy_mc(x, counts, small_cfg_lee, n_toys=4, observed_max_sigma=t0, seed=7)
    p2, s2, _ = lee_toy_mc(x, counts, small_cfg_lee, n_toys=4, observed_max_sigma=None, seed=7)
    # Results should be identical (same seed, same t0)
    assert abs(p1 - p2) < 1e-9
