"""Tests for synthetic data generation."""

import numpy as np
import pytest
from bumphunt.data import generate_synthetic_spectrum


def test_output_shapes():
    x, counts = generate_synthetic_spectrum(n_bins=100)
    assert x.shape == (100,)
    assert counts.shape == (100,)


def test_bin_centers_are_sequential():
    x, _ = generate_synthetic_spectrum(n_bins=50)
    np.testing.assert_array_equal(x, np.arange(50, dtype=float))


def test_counts_non_negative():
    _, counts = generate_synthetic_spectrum(n_bins=200, seed=0)
    assert np.all(counts >= 0)


def test_reproducibility():
    x1, c1 = generate_synthetic_spectrum(seed=42)
    x2, c2 = generate_synthetic_spectrum(seed=42)
    np.testing.assert_array_equal(c1, c2)


def test_different_seeds_differ():
    _, c1 = generate_synthetic_spectrum(seed=1)
    _, c2 = generate_synthetic_spectrum(seed=2)
    assert not np.array_equal(c1, c2)


def test_no_signal_bump_lower_counts():
    """Setting signal_events=0 should give fewer total counts than with a bump."""
    _, c_no_sig = generate_synthetic_spectrum(signal_events=0.0, seed=0)
    _, c_sig = generate_synthetic_spectrum(signal_events=500.0, seed=0)
    assert c_sig.sum() > c_no_sig.sum()


def test_signal_at_expected_position():
    """Counts near the signal position should be elevated vs. a no-signal run."""
    signal_pos = 60
    _, c_sig = generate_synthetic_spectrum(n_bins=120, signal_pos=signal_pos,
                                           signal_events=1000.0, seed=7)
    _, c_bkg = generate_synthetic_spectrum(n_bins=120, signal_pos=signal_pos,
                                           signal_events=0.0, seed=7)
    window = slice(signal_pos - 5, signal_pos + 6)
    assert c_sig[window].sum() > c_bkg[window].sum()
