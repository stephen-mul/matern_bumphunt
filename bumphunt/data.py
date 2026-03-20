"""Synthetic spectrum generator."""

from __future__ import annotations

import numpy as np


def generate_synthetic_spectrum(
    n_bins: int = 200,
    bkg_norm: float = 5000.0,
    bkg_slope: float = -0.025,
    bkg_curve: float = -0.00005,
    signal_pos: float = 120.0,
    signal_width: float = 3.0,
    signal_events: float = 250.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a falling background + Gaussian bump + Poisson noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(n_bins, dtype=float)

    # smooth falling background
    bkg = bkg_norm * np.exp(bkg_slope * x + bkg_curve * x**2)

    # Gaussian signal bump
    sig = signal_events * np.exp(-0.5 * ((x - signal_pos) / signal_width) ** 2)

    # Poisson-fluctuated observed counts
    lam = np.clip(bkg + sig, 0.1, None)
    counts = rng.poisson(lam).astype(float)

    return x, counts
