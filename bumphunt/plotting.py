"""Diagnostic plot for the GP bump hunt."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.gridspec import GridSpec

from bumphunt.config import BumpHuntConfig
from bumphunt.models import ScanResult
from bumphunt.scan import fit_full_background, max_local_significance


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
