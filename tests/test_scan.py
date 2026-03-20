"""Tests for the core scan engine."""

import numpy as np
import pytest

from bumphunt.config import BumpHuntConfig
from bumphunt.data import generate_synthetic_spectrum
from bumphunt.models import ScanResult
from bumphunt.scan import max_local_significance, run_bumphunt, run_scan


# Small, fast config to keep tests quick
@pytest.fixture
def small_cfg():
    return BumpHuntConfig(
        window_half_width=15,
        mask_half_widths=[1, 2],
        length_scale_bounds=(3.0, 30.0),
        min_sideband_bins=8,
        local_sigma_threshold=3.0,
    )


@pytest.fixture
def small_spectrum():
    return generate_synthetic_spectrum(
        n_bins=80,
        signal_pos=40.0,
        signal_width=3.0,
        signal_events=200.0,
        seed=0,
    )


# ── run_scan ──────────────────────────────────────────────────────────────────

def test_run_scan_returns_scan_result(small_cfg, small_spectrum):
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=1)
    assert isinstance(res, ScanResult)


def test_run_scan_correct_mask_half_width(small_cfg, small_spectrum):
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=2)
    assert res.mask_half_width == 2


def test_run_scan_output_length(small_cfg, small_spectrum):
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=1)
    W = small_cfg.window_half_width
    expected_len = len(x) - 2 * W
    assert len(res.positions) == expected_len
    assert len(res.local_sigma) == expected_len
    assert len(res.stitched_bkg) == expected_len


def test_run_scan_positions_range(small_cfg, small_spectrum):
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=1)
    W = small_cfg.window_half_width
    assert res.positions[0] == W
    assert res.positions[-1] == len(x) - W - 1


def test_run_scan_no_nan_free_positions(small_cfg, small_spectrum):
    """At least some positions should have valid (non-nan) significance values."""
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=1)
    assert np.any(~np.isnan(res.local_sigma))


def test_run_scan_stitched_bkg_positive_where_valid(small_cfg, small_spectrum):
    x, counts = small_spectrum
    res = run_scan(x, counts, small_cfg, mask_half_width=1)
    valid = ~np.isnan(res.stitched_bkg)
    assert np.all(res.stitched_bkg[valid] >= 0)


# ── run_bumphunt ──────────────────────────────────────────────────────────────

def test_run_bumphunt_returns_list(small_cfg, small_spectrum):
    x, counts = small_spectrum
    results = run_bumphunt(x, counts, cfg=small_cfg)
    assert isinstance(results, list)


def test_run_bumphunt_one_result_per_mask_width(small_cfg, small_spectrum):
    x, counts = small_spectrum
    results = run_bumphunt(x, counts, cfg=small_cfg)
    assert len(results) == len(small_cfg.mask_half_widths)


def test_run_bumphunt_mask_widths_match(small_cfg, small_spectrum):
    x, counts = small_spectrum
    results = run_bumphunt(x, counts, cfg=small_cfg)
    for res, w in zip(results, small_cfg.mask_half_widths):
        assert res.mask_half_width == w


def test_run_bumphunt_default_cfg(small_spectrum):
    """run_bumphunt with cfg=None should use default BumpHuntConfig."""
    x, counts = small_spectrum
    results = run_bumphunt(x, counts, cfg=None)
    assert len(results) == len(BumpHuntConfig().mask_half_widths)


# ── max_local_significance ────────────────────────────────────────────────────

def _make_scan_result(hw, sigmas):
    n = len(sigmas)
    return ScanResult(
        mask_half_width=hw,
        positions=np.arange(n, dtype=float),
        local_sigma=sigmas,
        excess_counts=np.zeros(n),
        predicted_bkg=np.zeros(n),
        predicted_unc=np.zeros(n),
        stitched_bkg=np.zeros(n),
    )


def test_max_local_significance_shape():
    r1 = _make_scan_result(1, np.array([1.0, 2.0, 3.0]))
    r2 = _make_scan_result(2, np.array([3.0, 1.0, 2.0]))
    pos, max_sig, best_hw = max_local_significance([r1, r2])
    assert len(pos) == 3
    assert len(max_sig) == 3
    assert len(best_hw) == 3


def test_max_local_significance_values():
    r1 = _make_scan_result(1, np.array([1.0, 5.0, 2.0]))
    r2 = _make_scan_result(2, np.array([3.0, 2.0, 4.0]))
    _, max_sig, best_hw = max_local_significance([r1, r2])
    np.testing.assert_array_equal(max_sig, [3.0, 5.0, 4.0])
    np.testing.assert_array_equal(best_hw, [2, 1, 2])


def test_max_local_significance_handles_nans():
    r1 = _make_scan_result(1, np.array([np.nan, 5.0, 2.0]))
    r2 = _make_scan_result(2, np.array([3.0, np.nan, 4.0]))
    _, max_sig, _ = max_local_significance([r1, r2])
    assert max_sig[0] == 3.0
    assert max_sig[1] == 5.0
    assert max_sig[2] == 4.0
