"""Tests for print_candidates."""

import numpy as np
import pytest

from bumphunt.config import BumpHuntConfig
from bumphunt.models import ScanResult
from bumphunt.reporting import print_candidates


def _make_scan_result(hw, sigmas, positions=None):
    n = len(sigmas)
    if positions is None:
        positions = np.arange(n, dtype=float)
    return ScanResult(
        mask_half_width=hw,
        positions=positions,
        local_sigma=sigmas,
        excess_counts=np.zeros(n),
        predicted_bkg=np.zeros(n),
        predicted_unc=np.zeros(n),
        stitched_bkg=np.zeros(n),
    )


def test_no_candidates(capsys):
    """When no bin exceeds threshold, should log 'No significant excesses'."""
    cfg = BumpHuntConfig(local_sigma_threshold=10.0)
    sigmas = np.array([1.0, 2.0, 3.0, 1.0, 0.5])
    results = [_make_scan_result(1, sigmas), _make_scan_result(2, sigmas)]
    # Should complete without error
    print_candidates(results, cfg)


def test_candidates_printed(capsys):
    """When bins exceed threshold, a table should be printed to stdout."""
    cfg = BumpHuntConfig(local_sigma_threshold=3.0)
    # Only position 2 is above threshold
    sigmas = np.array([1.0, 2.0, 8.0, 2.0, 1.0])
    results = [_make_scan_result(1, sigmas), _make_scan_result(2, sigmas * 0.5)]
    print_candidates(results, cfg)
    captured = capsys.readouterr()
    # The table header and at least one data row should be printed
    assert "Peak pos" in captured.out
    assert "Peak sigma" in captured.out


def test_multiple_clusters(capsys):
    """Separated peaks should appear as distinct clusters."""
    cfg = BumpHuntConfig(local_sigma_threshold=3.0)
    # Two peaks well separated (positions 2 and 18 — gap > 3)
    n = 25
    sigmas = np.ones(n)
    sigmas[2] = 7.0
    sigmas[18] = 8.0
    results = [_make_scan_result(1, sigmas)]
    print_candidates(results, cfg)
    captured = capsys.readouterr()
    # Clusters 1 and 2 should appear
    assert "1" in captured.out
    assert "2" in captured.out


def test_all_nans_raises_value_error():
    """All-NaN significance array raises ValueError from np.nanargmax."""
    cfg = BumpHuntConfig(local_sigma_threshold=3.0)
    sigmas = np.full(10, np.nan)
    results = [_make_scan_result(1, sigmas)]
    with pytest.raises(ValueError, match="All-NaN"):
        print_candidates(results, cfg)
