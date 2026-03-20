"""Candidate reporting / printing."""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger

from bumphunt.config import BumpHuntConfig
from bumphunt.models import ScanResult
from bumphunt.scan import max_local_significance


def print_candidates(results: list[ScanResult], cfg: BumpHuntConfig) -> None:
    """Print a table of bump candidates above threshold."""
    pos, max_sig, best_hw = max_local_significance(results)
    above = max_sig > cfg.local_sigma_threshold

    logger.info(
        "BUMP CANDIDATES  (local significance > {:.1f}sigma)", cfg.local_sigma_threshold
    )

    if not np.any(above):
        logger.info("No significant excesses found.")
        return

    # cluster adjacent bins into bump regions
    indices = np.where(above)[0]
    clusters = []
    start = indices[0]
    for k in range(1, len(indices)):
        if indices[k] - indices[k - 1] > 3:
            clusters.append((start, indices[k - 1]))
            start = indices[k]
    clusters.append((start, indices[-1]))

    print(f"  {'Region':>10s}  {'Peak pos':>10s}  {'Peak sigma':>8s}  {'Best hw':>8s}")
    print("  " + "-" * 42)
    for ci, (s, e) in enumerate(clusters):
        region_sig = max_sig[s : e + 1]
        peak_local = np.nanargmax(region_sig)
        peak_idx = s + peak_local
        print(
            f"  {ci + 1:>10d}  {pos[peak_idx]:>10.1f}  "
            f"{max_sig[peak_idx]:>8.2f}  {best_hw[peak_idx]:>8d}"
        )

    logger.warning(
        "These are LOCAL significances (pre-trials). "
        "Apply look-elsewhere correction for global p-values."
    )


def print_lee_summary(
    max_significance: float,
    gv_p: float,
    gv_sigma: float,
    gv_n_up: int,
    toy_p: Optional[float] = None,
    toy_sigma: Optional[float] = None,
    n_toys: Optional[int] = None,
) -> None:
    """Print a summary table of local vs. global significance."""
    from scipy.stats import norm

    local_p = norm.sf(max_significance)

    print("\n  Look-elsewhere effect correction")
    print("  " + "─" * 50)
    print(f"  {'Method':<26s}  {'p-value':>12s}  {'Significance':>12s}")
    print("  " + "─" * 50)
    print(f"  {'Local (no correction)':<26s}  {local_p:>12.3e}  {max_significance:>11.2f}σ")
    print(
        f"  {'Gross-Vitells':<26s}  {gv_p:>12.3e}  {gv_sigma:>11.2f}σ"
        f"  (N_up @ c₀ = {gv_n_up})"
    )
    if toy_p is not None and toy_sigma is not None:
        label = f"Toy MC (n={n_toys})" if n_toys else "Toy MC"
        print(f"  {label:<26s}  {toy_p:>12.3e}  {toy_sigma:>11.2f}σ")
    print("  " + "─" * 50)
    print()
