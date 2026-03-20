"""Configuration dataclass for the GP bump hunt."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BumpHuntConfig:
    """All tuneable parameters in one place."""

    # -- Sliding window --
    window_half_width: int = 25  # half-width of sideband window (in bins)
    mask_half_widths: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5]  # signal-region half-widths to scan
    )

    # -- GP kernel --
    nu: float = 5 / 2  # Matérn smoothness (2.5 = twice differentiable)
    length_scale_bounds: tuple = (6.0, 60.0)  # ℓ bounds — lower > max signal width
    noise_level_bounds: tuple = (1e-4, 1e0)  # white-noise kernel bounds (in log-space)

    # -- Significance --
    min_sideband_bins: int = 10  # require at least this many sideband bins
    local_sigma_threshold: float = 6.0  # flag peaks above this local significance

    # -- Look-elsewhere effect (Gross-Vitells correction) --
    lee_reference_level: float = 1.0  # reference level c₀ for counting upcrossings

    # -- Visualisation --
    figsize: tuple = (14, 10)
    dpi: int = 130
