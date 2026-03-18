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
    python gp_bumphunt.py                   # run on synthetic demo data
    python gp_bumphunt.py --input data.csv  # run on real data (two columns: bin_center, counts)
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.gridspec import GridSpec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore", category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


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

    # -- Visualisation --
    figsize: tuple = (14, 10)
    dpi: int = 130


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generator
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# GP fitting utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _build_kernel(cfg: BumpHuntConfig):
    """Construct the GP kernel: ConstantKernel × Matérn + WhiteKernel."""
    kern = ConstantKernel(
        constant_value=1.0, constant_value_bounds=(1e-2, 1e4)
    ) * Matern(
        nu=cfg.nu,
        length_scale=10.0,
        length_scale_bounds=cfg.length_scale_bounds,
    ) + WhiteKernel(
        noise_level=0.01,
        noise_level_bounds=cfg.noise_level_bounds,
    )
    return kern


def _fit_gp(x_train, y_train, cfg):
    """Fit a GP on log-counts and return the fitted regressor."""
    kernel = _build_kernel(cfg)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        alpha=0.0,
    )
    gp.fit(x_train.reshape(-1, 1), y_train)
    return gp


def _gp_predict(gp, x_pred):
    """Predict mean and std from a fitted GP."""
    mu, sigma = gp.predict(x_pred.reshape(-1, 1), return_std=True)
    return mu, sigma


def _log_to_counts(mu_log, sigma_log):
    """
    Convert GP predictions in log(counts+1) space back to counts space.

    Uses the log-normal mean: E[N] = exp(mu + sigma²/2) - 1
    and delta-method std:     sigma_N  ≈ exp(mu) * sigma
    """
    mu_counts = np.exp(mu_log + 0.5 * sigma_log**2) - 1.0
    mu_counts = np.clip(mu_counts, 0.0, None)
    sigma_counts = np.exp(mu_log) * sigma_log
    return mu_counts, sigma_counts


# ═══════════════════════════════════════════════════════════════════════════════
# Core: GP-based sliding-window bump hunt
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScanResult:
    """Holds the full scan output for one mask width."""

    mask_half_width: int
    positions: np.ndarray  # tested m₀ positions
    local_sigma: np.ndarray  # local significance at each m₀
    excess_counts: np.ndarray  # observed − predicted in mask
    predicted_bkg: np.ndarray  # GP predicted background in mask (counts, summed)
    predicted_unc: np.ndarray  # GP uncertainty (1sigma) in mask (counts)
    stitched_bkg: np.ndarray   # GP predicted count at the center bin m₀ (one per position)


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
            # log any GP fitting/prediction errors and skip this position

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


# ═══════════════════════════════════════════════════════════════════════════════
# Full-spectrum background fit (for plotting only)
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# Envelope: combine mask widths
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════


def plot_results(
    x: np.ndarray,
    counts: np.ndarray,
    results: list[ScanResult],
    cfg: BumpHuntConfig,
    true_signal_pos: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Produce the diagnostic plot: global fit, one stitched panel per mask width, significance panels."""
    pos, max_sig, best_hw = max_local_significance(results)
    n_widths = len(results)

    # Fit a full-spectrum GP background for the overlay
    logger.info("Fitting full-spectrum GP background for display …")
    x_fine, bkg_mu, bkg_lo, bkg_hi = fit_full_background(x, counts, cfg)

    # Layout: [global fit] + [stitched × n_widths] + [per-width sig] + [envelope]
    n_panels = 1 + n_widths + 2
    height_ratios = [2.5] + [1.5] * n_widths + [1.2, 1.2]
    fig_height = cfg.figsize[1] + 1.5 * n_widths
    fig = plt.figure(figsize=(cfg.figsize[0], fig_height), dpi=cfg.dpi, facecolor="white")
    gs = GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.08)

    c_data = "#2c3e50"
    c_bkg = "#e74c3c"
    c_stitch = "#8e44ad"
    c_sig = "#3498db"
    c_thresh = "#e67e22"
    c_widths = plt.cm.viridis(np.linspace(0.2, 0.9, n_widths))

    def _vline(ax):
        if true_signal_pos is not None:
            ax.axvline(true_signal_pos, color=c_sig, ls="--", lw=1, alpha=0.5)

    def _data_bars(ax):
        ax.bar(x, counts, width=1.0, color=c_data, alpha=0.45, label="Data")
        ax.errorbar(
            x, counts, yerr=np.sqrt(counts + 0.5),
            fmt="none", ecolor=c_data, alpha=0.35, linewidth=0.7,
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Counts / bin")

    # ── Panel 0: spectrum + global GP background ──────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _data_bars(ax0)
    ax0.plot(x_fine, bkg_mu, color=c_bkg, lw=2.0, label="GP bkg (global fit)")
    ax0.fill_between(x_fine, bkg_lo, bkg_hi, color=c_bkg, alpha=0.15, label="GP ±1σ")
    if true_signal_pos is not None:
        ax0.axvline(true_signal_pos, color=c_sig, ls="--", lw=1, alpha=0.7,
                    label=f"True signal @ {true_signal_pos:.0f}")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.set_title("Sliding-Window GP Bump Hunt", fontsize=13, fontweight="bold")
    ax0.tick_params(labelbottom=False)

    # ── Panels 1..n_widths: stitched background per mask width ────────────
    ax_prev = ax0
    stitch_axes = []
    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[1 + j], sharex=ax_prev)
        _data_bars(ax)
        v = ~np.isnan(res.stitched_bkg)
        ax.plot(
            res.positions[v], res.stitched_bkg[v],
            color=c_stitch, lw=1.5,
            label=f"Stitched GP bkg  (mask hw={res.mask_half_width})",
        )
        _vline(ax)
        ax.legend(loc="upper right", fontsize=9)
        ax.tick_params(labelbottom=False)
        stitch_axes.append(ax)
        ax_prev = ax

    # ── Panel n_widths+1: per-width significance ───────────────────────────
    ax_sig = fig.add_subplot(gs[1 + n_widths], sharex=ax0)
    for j, res in enumerate(results):
        v = ~np.isnan(res.local_sigma)
        ax_sig.plot(
            res.positions[v], res.local_sigma[v],
            color=c_widths[j], lw=1.0, alpha=0.8,
            label=f"hw={res.mask_half_width}",
        )
    ax_sig.axhline(cfg.local_sigma_threshold, color=c_thresh, ls=":", lw=1.2,
                   label=f"{cfg.local_sigma_threshold:.0f}σ threshold")
    ax_sig.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
    ax_sig.set_ylabel("Local signif. (σ)")
    ax_sig.legend(loc="upper right", fontsize=8, ncol=3)
    ax_sig.tick_params(labelbottom=False)
    _vline(ax_sig)

    # ── Panel n_widths+2: max significance envelope ────────────────────────
    ax_env = fig.add_subplot(gs[2 + n_widths], sharex=ax0)
    valid_env = ~np.isnan(max_sig)
    ax_env.fill_between(pos[valid_env], 0, max_sig[valid_env], color=c_sig, alpha=0.25)
    ax_env.plot(pos[valid_env], max_sig[valid_env], color=c_sig, lw=1.5,
                label="Max over mask widths")
    ax_env.axhline(cfg.local_sigma_threshold, color=c_thresh, ls=":", lw=1.2)
    ax_env.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
    above = max_sig > cfg.local_sigma_threshold
    if np.any(above & valid_env):
        ax_env.scatter(
            pos[above & valid_env], max_sig[above & valid_env],
            color=c_thresh, s=30, zorder=5,
            label=f"Candidates (>{cfg.local_sigma_threshold:.0f}σ)",
        )
    _vline(ax_env)
    ax_env.set_xlabel("Bin index  (invariant mass proxy)")
    ax_env.set_ylabel("Max local σ")
    ax_env.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.success("Saved → {}", save_path)
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_candidates(results: list[ScanResult], cfg: BumpHuntConfig):
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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════


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
        logger.info("No input file — generating synthetic spectrum with injected bump …")
        x, counts = generate_synthetic_spectrum(
            n_bins=200, signal_pos=120.0, signal_width=3.0, signal_events=250.0
        )
        true_pos = 120.0

    logger.info("Spectrum: {} bins,  total counts = {:.0f}", len(x), counts.sum())
    logger.info("Mask half-widths to scan: {}", cfg.mask_half_widths)
    logger.info("GP kernel: Matérn ν={}, ℓ bounds={}", cfg.nu, cfg.length_scale_bounds)

    results = run_bumphunt(x, counts, cfg=cfg, verbose=args.verbose)

    print_candidates(results, cfg)

    plot_results(
        x, counts, results, cfg, true_signal_pos=true_pos, save_path=args.output
    )


if __name__ == "__main__":
    main()
