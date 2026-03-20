"""Tests for BumpHuntConfig."""

import pytest
from bumphunt.config import BumpHuntConfig


def test_defaults():
    cfg = BumpHuntConfig()
    assert cfg.window_half_width == 25
    assert cfg.mask_half_widths == [1, 2, 3, 4, 5]
    assert cfg.nu == 2.5
    assert cfg.length_scale_bounds == (6.0, 60.0)
    assert cfg.noise_level_bounds == (1e-4, 1e0)
    assert cfg.min_sideband_bins == 10
    assert cfg.local_sigma_threshold == 6.0
    assert cfg.figsize == (14, 10)
    assert cfg.dpi == 130


def test_custom_values():
    cfg = BumpHuntConfig(
        window_half_width=30,
        mask_half_widths=[2, 4],
        local_sigma_threshold=3.0,
    )
    assert cfg.window_half_width == 30
    assert cfg.mask_half_widths == [2, 4]
    assert cfg.local_sigma_threshold == 3.0


def test_mask_half_widths_are_independent_instances():
    """Each instance should have its own list, not share a default."""
    cfg1 = BumpHuntConfig()
    cfg2 = BumpHuntConfig()
    cfg1.mask_half_widths.append(99)
    assert 99 not in cfg2.mask_half_widths
