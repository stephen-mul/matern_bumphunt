"""bumphunt — sliding-window GP bump hunt package."""

from bumphunt.config import BumpHuntConfig
from bumphunt.data import generate_synthetic_spectrum
from bumphunt.models import ScanResult
from bumphunt.scan import (
    fit_full_background,
    max_local_significance,
    run_bumphunt,
    run_scan,
)
from bumphunt.reporting import print_candidates, print_lee_summary
from bumphunt.plotting import plot_results
from bumphunt.lee import count_upcrossings, gross_vitells_correction, lee_toy_mc

__all__ = [
    "BumpHuntConfig",
    "generate_synthetic_spectrum",
    "ScanResult",
    "run_scan",
    "run_bumphunt",
    "fit_full_background",
    "max_local_significance",
    "print_candidates",
    "print_lee_summary",
    "plot_results",
    "count_upcrossings",
    "gross_vitells_correction",
    "lee_toy_mc",
]
