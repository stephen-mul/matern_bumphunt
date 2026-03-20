"""
Sliding-Window Gaussian Process Bump Hunt
==========================================

Performs a model-independent search for localised excesses (bumps) in a
binned 1-D invariant-mass spectrum using Gaussian Process Regression with
a Matérn kernel as the background model.

Algorithm
---------
For each candidate bump position m₀ the code:
  1. Opens a window of width W centered on m₀.
  2. Masks a signal region of width w (scan parameter) around m₀.
  3. Fits a GP (Matérn-5/2 kernel) to the sideband data inside the window
     but outside the mask.  The GP is trained in log-counts space.
  4. Predicts the background expectation and uncertainty inside the mask.
  5. Computes a local test statistic (excess significance in sigma).
  6. Slides m₀ across the full spectrum and records the scan.

The result is a significance curve t(m₀) whose peaks flag bump candidates.

Usage
-----
    python main.py                            # synthetic demo data
    python main.py --input data.csv           # real data
    python main.py --lee-toys 1000            # toy-MC LEE correction
"""

from __future__ import annotations

import argparse

import numpy as np
from loguru import logger

from bumphunt import (
    BumpHuntConfig,
    generate_synthetic_spectrum,
    gross_vitells_correction,
    lee_toy_mc,
    max_local_significance,
    plot_results,
    print_candidates,
    print_lee_summary,
    run_bumphunt,
)


def main():
    parser = argparse.ArgumentParser(description="GP Sliding-Window Bump Hunt")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="CSV file with columns: bin_center, counts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bumphunt_results.png",
        help="Output plot path (default: bumphunt_results.png)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level (DEBUG, INFO, WARNING, ERROR). Default: DEBUG if --verbose, else INFO.",
    )
    parser.add_argument(
        "--lee-toys",
        type=int,
        default=0,
        metavar="N",
        help="Run toy-MC LEE correction with N pseudo-experiments (default: off). "
        "Use N ≥ 1000 for reliable sub-per-mille p-values.",
    )
    parser.add_argument(
        "--lee-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="RNG seed for the toy-MC LEE correction.",
    )
    args = parser.parse_args()

    log_level = args.log_level or ("DEBUG" if args.verbose else "INFO")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=log_level, colorize=True)

    cfg = BumpHuntConfig()

    if args.input:
        data = np.loadtxt(args.input, delimiter=",", skiprows=1)
        x, counts = data[:, 0], data[:, 1]
        true_pos = None
    else:
        logger.info(
            "No input file — generating synthetic spectrum with injected bump …"
        )
        x, counts = generate_synthetic_spectrum(
            n_bins=200, signal_pos=120.0, signal_width=3.0, signal_events=80.0
        )
        true_pos = 120.0

    logger.info("Spectrum: {} bins,  total counts = {:.0f}", len(x), counts.sum())
    logger.info("Mask half-widths to scan: {}", cfg.mask_half_widths)
    logger.info("GP kernel: Matérn ν={}, ℓ bounds={}", cfg.nu, cfg.length_scale_bounds)

    results = run_bumphunt(x, counts, cfg=cfg, verbose=args.verbose)

    print_candidates(results, cfg)

    # ── Look-elsewhere effect correction ──────────────────────────────────────
    pos, max_sig, _ = max_local_significance(results)
    t0 = float(np.nanmax(max_sig))

    logger.info(
        "LEE (Gross-Vitells): reference level c₀ = {:.1f}σ", cfg.lee_reference_level
    )
    gv_p, gv_sigma, gv_n_up = gross_vitells_correction(
        t0, max_sig, reference_level=cfg.lee_reference_level
    )

    toy_p: float | None = None
    toy_sigma: float | None = None
    if args.lee_toys > 0:
        logger.info("LEE toy MC: running {} pseudo-experiments …", args.lee_toys)
        toy_p, toy_sigma, _ = lee_toy_mc(
            x,
            counts,
            cfg,
            n_toys=args.lee_toys,
            observed_max_sigma=t0,
            seed=args.lee_seed,
            verbose=args.verbose,
        )

    print_lee_summary(
        t0,
        gv_p,
        gv_sigma,
        gv_n_up,
        toy_p=toy_p,
        toy_sigma=toy_sigma,
        n_toys=args.lee_toys if args.lee_toys > 0 else None,
    )

    plot_results(
        x, counts, results, cfg, true_signal_pos=true_pos, save_path=args.output
    )


if __name__ == "__main__":
    main()
