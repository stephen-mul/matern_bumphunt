# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Maintenance

Keep `README.md` up to date as the project evolves. Any significant change — new CLI flags, changed defaults, replaced transforms, new dependencies, or architectural shifts — should be reflected in the README before the work is considered complete.

## Running the code

```bash
uv run python main.py                        # synthetic demo (200-bin spectrum, bump at bin 120)
uv run python main.py --input data.csv       # real data: CSV with columns bin_center, counts
uv run python main.py --verbose              # print per-bin progress
```

Output plot is saved to `bumphunt_results.png` by default (PNG files are gitignored).

## Architecture

Everything lives in `main.py`. The pipeline has four layers:

1. **Config** — `BumpHuntConfig` dataclass holds all tunable parameters (window width, mask widths, GP kernel bounds, significance threshold).

2. **Data** — `generate_synthetic_spectrum()` produces a falling exponential background plus a Gaussian bump with Poisson noise, used when no `--input` is given.

3. **Scan engine** — `run_bumphunt()` → `run_scan()` implements the sliding-window loop. Counts are transformed to **log(counts + 1)** space before GP fitting. For each candidate position the GP is fit on sideband data and predicts into the masked signal region; predictions are converted back to counts space via `_log_to_counts()` (log-normal mean: `exp(μ + σ²/2) - 1`, delta-method std: `exp(μ) * σ`). The local significance is `(obs - pred) / sqrt(pred + sigma_gp²)` summed over the mask. Results are stored in `ScanResult` dataclasses.

4. **Post-processing** — `max_local_significance()` takes the per-position envelope over all mask widths. `print_candidates()` clusters adjacent above-threshold bins and reports them. `plot_results()` produces a multi-panel diagnostic figure: one global-fit panel, one stitched-background panel per mask width, then the per-width significance curves and the max-significance envelope. The stitched background for a given mask width is the GP's prediction at each center bin m₀ (stored as `ScanResult.stitched_bkg`), assembled from the per-position sideband fits. `fit_full_background()` fits the entire spectrum at once for the global overlay (display only — it partially absorbs any signal).

## Key design choices

- The GP kernel is `ConstantKernel × Matérn(ν=2.5) + WhiteKernel`. The length-scale lower bound (default 6 bins) is intentionally set above the maximum expected signal width so the GP cannot absorb a real bump into the background model.
- The GP fits in log(counts + 1) space, which suits the multiplicative structure of a falling spectrum. Back-conversion uses the exact log-normal mean rather than a linear approximation.
- The test statistic variance combines Poisson variance (≈ predicted counts) with GP uncertainty in quadrature, all in counts space.
- The background overlay in the plot (`fit_full_background`) sees the full spectrum including any signal, so it will partially absorb a bump — this is intentional and noted in the code.
- Reported significances are **local** (pre-trials correction). A look-elsewhere correction is needed for global p-values.
