"""Core sliding-window GP scan engine."""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from bumphunt.config import BumpHuntConfig
from bumphunt.models import ScanResult, _fit_gp, _gp_predict, _log_to_counts


def run_scan(
    x: np.ndarray,
    counts: np.ndarray,
    cfg: BumpHuntConfig,
    mask_half_width: int,
    verbose: bool = False,
) -> ScanResult:
    """
    Run the sliding-window GP scan for a single mask width.

    The GP works in log(counts+1) space to naturally handle the falling
    spectrum and stabilise the variance.
    """
    n = len(x)
    W = cfg.window_half_width
    w = mask_half_width

    # Transform to log-space
    log_counts = np.log(counts + 1.0)

    # Positions we can test
    lo = W
    hi = n - W
    test_positions = np.arange(lo, hi)

    sigmas = np.full(len(test_positions), np.nan)
    excesses = np.full(len(test_positions), np.nan)
    pred_bkg = np.full(len(test_positions), np.nan)
    pred_unc = np.full(len(test_positions), np.nan)
    stitched = np.full(len(test_positions), np.nan)

    for i, m0 in enumerate(test_positions):
        if verbose and i % 20 == 0:
            logger.debug("mask_hw={}  scanning bin {} / {}", w, m0, test_positions[-1])

        # window indices
        win_idx = np.arange(max(m0 - W, 0), min(m0 + W + 1, n))

        # mask indices (signal region)
        mask_idx = np.arange(max(m0 - w, 0), min(m0 + w + 1, n))
        mask_set = set(mask_idx)

        # sideband = window minus mask
        sb_idx = np.array([j for j in win_idx if j not in mask_set])

        if len(sb_idx) < cfg.min_sideband_bins:
            continue

        x_sb = x[sb_idx]
        y_sb = log_counts[sb_idx]
        x_mask = x[mask_idx]

        # Fit GP on sideband, predict into mask (log-space)
        try:
            gp = _fit_gp(x_sb, y_sb, cfg)
            mu_log, sigma_log = _gp_predict(gp, x_mask)
        except Exception:
            continue

        # Convert to counts space
        mu_counts, sigma_counts = _log_to_counts(mu_log, sigma_log)

        # Observed counts in mask
        obs_mask = counts[mask_idx]

        # Test statistic: sum of (obs - pred) / sqrt(pred + sigma_gp²)
        # Denominator combines Poisson variance (≈ predicted counts)
        # with the GP's own prediction uncertainty
        obs_total = obs_mask.sum()
        pred_total = mu_counts.sum()
        total_var = (mu_counts + sigma_counts**2).sum()

        if total_var <= 0:
            continue

        t_stat = (obs_total - pred_total) / np.sqrt(total_var)

        sigmas[i] = t_stat
        pred_bkg[i] = pred_total
        excesses[i] = obs_total - pred_total
        pred_unc[i] = np.sqrt(total_var)

        # Stitched background: GP prediction at the center bin m₀ only
        center_offset = m0 - mask_idx[0]
        stitched[i] = mu_counts[center_offset]

    return ScanResult(
        mask_half_width=mask_half_width,
        positions=test_positions.astype(float),
        local_sigma=sigmas,
        excess_counts=excesses,
        predicted_bkg=pred_bkg,
        predicted_unc=pred_unc,
        stitched_bkg=stitched,
    )


def run_bumphunt(
    x: np.ndarray,
    counts: np.ndarray,
    cfg: Optional[BumpHuntConfig] = None,
    verbose: bool = False,
) -> list[ScanResult]:
    """Run the full multi-width sliding-window GP bump hunt."""
    if cfg is None:
        cfg = BumpHuntConfig()

    results = []
    for w in cfg.mask_half_widths:
        if verbose:
            logger.info("Scanning with mask half-width = {} bins …", w)
        res = run_scan(x, counts, cfg, mask_half_width=w, verbose=verbose)
        results.append(res)

    return results


def fit_full_background(
    x: np.ndarray,
    counts: np.ndarray,
    cfg: BumpHuntConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a GP to the ENTIRE spectrum (no mask) to produce a smooth
    background overlay for the diagnostic plot.

    This is NOT used in the scan — it's purely for visualisation.
    Because it sees all the data (including any signal), it will
    partially absorb a bump, but that's fine for display purposes.

    Returns (x_fine, mu_counts, lower_counts, upper_counts).
    """
    log_counts = np.log(counts + 1.0)
    gp = _fit_gp(x, log_counts, cfg)

    x_fine = np.linspace(x[0], x[-1], len(x) * 3)
    mu_log, sigma_log = _gp_predict(gp, x_fine)

    mu_counts, _ = _log_to_counts(mu_log, sigma_log)
    upper = np.exp(mu_log + sigma_log) - 1.0
    lower = np.exp(mu_log - sigma_log) - 1.0
    lower = np.clip(lower, 0, None)

    return x_fine, mu_counts, lower, upper


def max_local_significance(
    results: list[ScanResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    At each position, take the maximum local significance across all mask widths.
    Returns (positions, max_sigma, best_mask_hw).
    """
    pos = results[0].positions
    stack = np.column_stack([r.local_sigma for r in results])
    best_idx = np.nanargmax(stack, axis=1)
    max_sig = np.nanmax(stack, axis=1)
    best_hw = np.array(
        [
            results[0].mask_half_width
            if np.isnan(max_sig[i])
            else results[best_idx[i]].mask_half_width
            for i in range(len(pos))
        ]
    )
    return pos, max_sig, best_hw
