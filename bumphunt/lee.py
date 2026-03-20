"""
Look-elsewhere effect (LEE) correction
=======================================

Two methods are provided:

Gross-Vitells (fast)
    Estimates the global p-value analytically from the number of upcrossings
    of a reference level in the scan envelope (Gross & Vitells 2010).
    Formula:
        p_global ≈ Φ(−t₀) + N(c₀) × exp(−(t₀² − c₀²) / 2)
    where t₀ is the observed max local significance, c₀ is a reference level
    at which upcrossings N(c₀) are counted from the scan curve.

Toy MC (slow, opt-in)
    Generates Poisson pseudo-experiments from the full-spectrum GP background
    fit, runs the full bump-hunt on each, and computes the fraction of toys
    whose max significance exceeds the observed value.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from scipy.stats import norm

from bumphunt.config import BumpHuntConfig
from bumphunt.models import ScanResult
from bumphunt.scan import fit_full_background, max_local_significance, run_bumphunt


# ── Upcrossing counter ────────────────────────────────────────────────────────


def count_upcrossings(curve: np.ndarray, level: float) -> int:
    """
    Count the number of times *curve* crosses *level* from below (upcrossings).

    NaN values are ignored; only contiguous valid stretches contribute.
    """
    valid = ~np.isnan(curve)
    c = curve[valid]
    if len(c) < 2:
        return 0
    above = c > level
    return int(np.sum(~above[:-1] & above[1:]))


# ── Gross-Vitells approximation ───────────────────────────────────────────────


def gross_vitells_correction(
    max_significance: float,
    scan_envelope: np.ndarray,
    reference_level: float = 1.0,
) -> tuple[float, float, int]:
    """
    Estimate the global p-value with the Gross-Vitells (2010) approximation.

    The expected number of upcrossings of level t₀ is estimated from the
    observed count at a lower reference level c₀:

        E[N(t₀)] ≈ N_obs(c₀) × exp(−(t₀² − c₀²) / 2)

    The global p-value is then:

        p_global ≈ Φ(−t₀) + E[N(t₀)]

    Parameters
    ----------
    max_significance : float
        Observed maximum local significance t₀.
    scan_envelope : np.ndarray
        Significance envelope (max over mask widths at each position).
    reference_level : float
        Reference level c₀ for counting upcrossings.  Should be low enough
        that several crossings are expected (default 1.0 σ).

    Returns
    -------
    p_global : float
    global_sigma_equiv : float   — Φ⁻¹(1 − p_global), floored at 0
    n_upcrossings : int          — N_obs(c₀)
    """
    n_up = count_upcrossings(scan_envelope, reference_level)
    p_local = norm.sf(max_significance)
    # Expected number of upcrossings at t₀, extrapolated from reference level
    expected_crossings = n_up * np.exp(-0.5 * (max_significance**2 - reference_level**2))
    p_global = float(np.clip(p_local + expected_crossings, 0.0, 1.0))
    global_sigma = float(max(norm.isf(p_global), 0.0))
    return p_global, global_sigma, n_up


# ── Toy-MC correction ─────────────────────────────────────────────────────────


def lee_toy_mc(
    x: np.ndarray,
    counts: np.ndarray,
    cfg: BumpHuntConfig,
    n_toys: int = 500,
    observed_max_sigma: Optional[float] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    Estimate the global p-value via toy pseudo-experiments.

    Procedure:
      1. Fit a full-spectrum GP background to the data.
      2. Draw *n_toys* Poisson spectra from that background.
      3. Run the full bump-hunt scan on each toy.
      4. p_global = fraction of toys whose max significance ≥ observed max.

    The GP background partially absorbs any real signal, so the toys are
    slightly signal-contaminated — this makes the correction conservative
    (p_global is slightly over-estimated).

    Each toy requires a full scan, so total runtime ≈ n_toys × single-scan
    time.  Use n_toys ≥ 1 000 for reliable p-values below 10⁻³.

    Parameters
    ----------
    x, counts : np.ndarray
        Observed spectrum.
    cfg : BumpHuntConfig
    n_toys : int
        Number of pseudo-experiments.
    observed_max_sigma : float, optional
        Observed maximum local significance.  If None, computed internally
        from a fresh scan on *counts*.
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool

    Returns
    -------
    p_global : float
    global_sigma_equiv : float
    toy_max_sigmas : np.ndarray, shape (n_toys,)  — NaN for failed toys
    """
    logger.info("LEE toy MC: fitting background model for pseudo-experiments …")
    x_fine, bkg_mu, _, _ = fit_full_background(x, counts, cfg)
    bkg_at_x = np.clip(np.interp(x, x_fine, bkg_mu), 0.1, None)

    if observed_max_sigma is None:
        results = run_bumphunt(x, counts, cfg=cfg, verbose=False)
        _, max_sig, _ = max_local_significance(results)
        observed_max_sigma = float(np.nanmax(max_sig))

    rng = np.random.default_rng(seed)
    toy_max_sigmas = np.full(n_toys, np.nan)

    for i in range(n_toys):
        if verbose and i % 50 == 0:
            logger.debug("LEE toy {}/{}", i + 1, n_toys)
        toy_counts = rng.poisson(bkg_at_x).astype(float)
        try:
            toy_results = run_bumphunt(x, toy_counts, cfg=cfg, verbose=False)
            _, toy_sig, _ = max_local_significance(toy_results)
            toy_max_sigmas[i] = float(np.nanmax(toy_sig))
        except Exception:
            continue

    valid = toy_max_sigmas[~np.isnan(toy_max_sigmas)]
    if len(valid) == 0:
        raise RuntimeError("All toy scans failed — cannot estimate global p-value.")

    n_exceed = int(np.sum(valid >= observed_max_sigma))
    p_global = float(n_exceed / len(valid))

    if p_global == 0.0:
        # No toy exceeded threshold: set upper limit p < 1/n_valid
        p_global = 1.0 / len(valid)
        logger.warning(
            "No toy exceeded the observed significance; "
            "global p-value set to upper limit {:.2e} (1/{}).",
            p_global,
            len(valid),
        )

    global_sigma = float(max(norm.isf(p_global), 0.0))
    logger.info(
        "LEE toy MC: {}/{} valid toys; {} exceeded t₀={:.2f} → p_global={:.3e} ({:.2f}σ)",
        len(valid),
        n_toys,
        n_exceed,
        observed_max_sigma,
        p_global,
        global_sigma,
    )
    return p_global, global_sigma, toy_max_sigmas
