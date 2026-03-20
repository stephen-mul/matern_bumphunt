"""ScanResult dataclass and GP fitting utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bumphunt.config import BumpHuntConfig

warnings.filterwarnings("ignore", category=UserWarning)


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


def _fit_gp(x_train: np.ndarray, y_train: np.ndarray, cfg: BumpHuntConfig) -> GaussianProcessRegressor:
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


def _gp_predict(gp: GaussianProcessRegressor, x_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and std from a fitted GP."""
    mu, sigma = gp.predict(x_pred.reshape(-1, 1), return_std=True)
    return mu, sigma


def _log_to_counts(mu_log: np.ndarray, sigma_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert GP predictions in log(counts+1) space back to counts space.

    Uses the log-normal mean: E[N] = exp(mu + sigma²/2) - 1
    and delta-method std:     sigma_N  ≈ exp(mu) * sigma
    """
    mu_counts = np.exp(mu_log + 0.5 * sigma_log**2) - 1.0
    mu_counts = np.clip(mu_counts, 0.0, None)
    sigma_counts = np.exp(mu_log) * sigma_log
    return mu_counts, sigma_counts
