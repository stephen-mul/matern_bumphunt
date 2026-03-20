# Matérn GP Bump Hunt

A model-independent search for localised excesses ("bumps") in a binned 1-D invariant-mass spectrum, using Gaussian Process Regression with a Matérn-5/2 kernel as the background model.

## Algorithm

For each candidate bump position m₀, the sliding-window procedure:

1. Opens a window of half-width `W` bins centered on m₀
2. Masks a signal region of half-width `w` around m₀
3. Fits a GP (Matérn-5/2 kernel) to the sideband data (window minus mask)
4. Predicts the background expectation and uncertainty inside the mask
5. Computes a local test statistic (excess significance in σ) in counts space
6. Slides m₀ across the full spectrum and records the scan

The scan is repeated for multiple mask half-widths to be sensitive to bumps of different widths. The final significance curve is the per-position envelope (maximum σ over all mask widths), whose peaks flag bump candidates.

## Usage

```bash
# Run on synthetic demo data (200-bin falling spectrum with injected Gaussian bump at bin 120)
uv run python main.py

# Run on real data — CSV with columns: bin_center, counts
uv run python main.py --input data.csv

# Specify output plot path and log level
uv run python main.py --input data.csv --output results.png --log-level DEBUG

# Shorthand for DEBUG logging
uv run python main.py --verbose

# Add toy-MC look-elsewhere correction (slow: each toy = one full scan)
uv run python main.py --lee-toys 1000 --lee-seed 42
```

## Output

- A multi-panel diagnostic PNG plot:
  - **Global fit**: observed spectrum with full-spectrum GP background overlay and ±1σ band
  - **Stitched fit × N**: one panel per mask half-width showing the stitched sideband-GP background overlaid on the data
  - **Significance**: local significance curves for each mask half-width
  - **Envelope**: max-significance curve across all mask widths, with candidates marked above threshold
- A printed table of bump candidates (clustered regions above the local σ threshold)

> **Note**: Gross-Vitells LEE correction runs automatically after every scan.  Use `--lee-toys N` for a toy-MC cross-check (N ≥ 1 000 recommended for reliable sub-per-mille global p-values).

## Configuration

All tunable parameters are in `BumpHuntConfig` (`bumphunt/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `window_half_width` | 25 bins | Sideband window half-width |
| `mask_half_widths` | [1,2,3,4,5] | Signal region half-widths to scan |
| `nu` | 2.5 | Matérn smoothness parameter |
| `length_scale_bounds` | (6, 60) | GP length-scale bounds (lower bound > max signal width) |
| `noise_level_bounds` | (1e-4, 1e0) | White-noise kernel bounds (in log-space) |
| `min_sideband_bins` | 10 | Minimum sideband bins required to fit |
| `local_sigma_threshold` | 3.0 | σ threshold for flagging candidates |
| `lee_reference_level` | 1.0 | Reference level c₀ for Gross-Vitells upcrossing count |

## Installation

Requires Python ≥ 3.12. [uv](https://github.com/astral-sh/uv) is the recommended way to install and run the project:

```bash
uv sync
uv run python main.py
```

Or with pip:

```bash
pip install numpy matplotlib scikit-learn loguru
python main.py
```

## Tests

```bash
uv run pytest
```

## Project layout

```
bumphunt/          # core package
  config.py        # BumpHuntConfig
  data.py          # generate_synthetic_spectrum
  models.py        # ScanResult + GP utilities
  scan.py          # run_scan, run_bumphunt, fit_full_background, max_local_significance
  reporting.py     # print_candidates, print_lee_summary
  plotting.py      # plot_results
  lee.py           # gross_vitells_correction, lee_toy_mc, count_upcrossings
tests/             # pytest test suite
main.py            # CLI entry point
```

## Dependencies

- `numpy` ≥ 2.4
- `matplotlib` ≥ 3.10
- `scikit-learn` ≥ 1.8
- `loguru` ≥ 0.7
