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
```

## Output

- A multi-panel diagnostic PNG plot:
  - **Global fit**: observed spectrum with full-spectrum GP background overlay and ±1σ band
  - **Stitched fit × N**: one panel per mask half-width showing the stitched sideband-GP background overlaid on the data
  - **Significance**: local significance curves for each mask half-width
  - **Envelope**: max-significance curve across all mask widths, with candidates marked above threshold
- A printed table of bump candidates (clustered regions above the local σ threshold)

> **Note**: reported significances are *local* (pre-trials). Apply a look-elsewhere correction for global p-values.

## Configuration

All tunable parameters are in `BumpHuntConfig` (`main.py:51`):

| Parameter | Default | Description |
|---|---|---|
| `window_half_width` | 25 bins | Sideband window half-width |
| `mask_half_widths` | [1,2,3,4,5] | Signal region half-widths to scan |
| `nu` | 2.5 | Matérn smoothness parameter |
| `length_scale_bounds` | (6, 60) | GP length-scale bounds (lower bound > max signal width) |
| `noise_level_bounds` | (1e-4, 1e0) | White-noise kernel bounds (in log-space) |
| `min_sideband_bins` | 10 | Minimum sideband bins required to fit |
| `local_sigma_threshold` | 3.0 | σ threshold for flagging candidates |

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

## Dependencies

- `numpy` ≥ 2.4
- `matplotlib` ≥ 3.10
- `scikit-learn` ≥ 1.8
- `loguru` ≥ 0.7
