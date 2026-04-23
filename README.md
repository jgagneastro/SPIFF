# SPIFF

SPIFF provides the SPHEREx level-2 batch workflow, single-image fitting tools, result compilation, and local autotyping against bundled templates.

## Citation

If you use SPIFF in published work, please cite:

J. Gagné et al. (2026, submitted to ApJ)

## Install

### From a clone

```bash
git clone https://github.com/jgagneastro/SPIFF.git
cd SPIFF
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Directly from GitHub

Base install:

```bash
pip install "spiff @ git+https://github.com/jgagneastro/SPIFF.git"
```

With UltraNest support:

```bash
pip install "spiff[ultranest] @ git+https://github.com/jgagneastro/SPIFF.git"
```

If you do not install the UltraNest extra, use `--scipy-only` for fitting commands.

## Command Summary

- `spiff-lv2`
  - Discover nearby SPHEREx level-2 observations, download FITS files or cutouts, and run the per-image fitter.
- `spiff-fit`
  - Run the single-image fitter on one local FITS file.
- `spiff-compile-results`
  - Combine `figs/*/result.csv` or `figs/*/results.csv` into one `compiled_results.csv`.
- `spiff-autotype`
  - Compare a local spectrum against the bundled SPHEREx template library.

## Required Target Inputs

For SPIFF runs, you must supply:

- `ra`
- `dec`
- `reference-crd-epoch-yr`
- `reference-pmra-masyr`
- `reference-pmdec-masyr`

If your target has negligible proper motion, pass `0.0` for both proper-motion values.

## Quick Start

### 1. Run the `lv2` workflow

This example uses a stationary target and writes into `./runs`.

```bash
spiff-lv2 \
  --ra 10.684708 \
  --dec 41.268750 \
  --reference-crd-epoch-yr 2016.0 \
  --reference-pmra-masyr 0.0 \
  --reference-pmdec-masyr 0.0 \
  --target-name M31_demo \
  --outdir ./runs \
  --save-figs
```

That creates an output directory like:

```text
./runs/M31_demo_RA10.684708_DEC41.268750/
```

Inside it you will typically see:

- `results.csv`
- `results.jsonl`
- `download_manifest.csv`
- `fits/`
- `figs/`

### 2. Compile per-image result files

```bash
spiff-compile-results \
  --outdir ./runs/M31_demo_RA10.684708_DEC41.268750
```

This writes:

```text
./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv
```

### 3. Run autotype on the compiled SPIFF output

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv \
  --no-plot
```

If you want the diagnostic plot saved next to the CSV, omit `--no-save-plot`.

## SIMP J0136 Smoke Test

SPIFF includes a reproducible smoke-test target: `SIMP J013656.5+093347.3`.

Use these target parameters:

- `ra = 24.24124978791965`
- `dec = 9.563070454755835`
- `reference-crd-epoch-yr = 2016.0`
- `reference-pmra-masyr = 1238.239990234375`
- `reference-pmdec-masyr = -16.15559959411621`

If you installed the UltraNest extra, you can run the full end-to-end SPIFF flow with:

```bash
spiff-lv2 \
  --ra 24.24124978791965 \
  --dec 9.563070454755835 \
  --reference-crd-epoch-yr 2016.0 \
  --reference-pmra-masyr 1238.239990234375 \
  --reference-pmdec-masyr -16.15559959411621 \
  --target-name "SIMP J013656.5+093347.3" \
  --outdir ./runs \
  --save-figs

spiff-compile-results \
  --outdir ./runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070

spiff-autotype \
  --csv ./runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070/compiled_results.csv \
  --no-plot
```

If you prefer Python over shell commands, the same flow is available as a helper function:

```python
from spiff.examples import run_simp_j0136_smoke_test

result = run_simp_j0136_smoke_test("./runs")
print(result["compiled_csv"])
```

If you only want the copy-paste shell commands, you can generate them programmatically:

```python
from spiff.examples import build_simp_j0136_test_commands

for command in build_simp_j0136_test_commands("./runs"):
    print(command)
```

## Autotype Inputs

`spiff-autotype` supports these local CSV shapes.

### Raw SPIFF-style CSV

Required columns:

- `psf_un_wv_um`
- `psf_un_flux_uJy`
- `psf_un_flux_uJy_err`

Optional columns that help filtering/plotting:

- `psf_un_snr`
- `n_pix_used_in_fit`
- `ignored`

This works directly on SPIFF `results.csv` or `compiled_results.csv` files because those already contain the `psf_un_*` columns.

### Reduced-spectrum CSV

Required columns:

- `wavelength_angstrom`
- `flux_flambda`
- `flux_flambda_unc`

Optional:

- `ignored`

## Useful Autotype Options

### Overplot local binning without fitting the binned spectrum

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv \
  --overplotbin \
  --no-plot
```

### Fit the locally binned spectrum

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv \
  --bin \
  --no-plot
```

### Overlay specific template types

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv \
  --overlay-spectral-types T8,L0 \
  --no-plot
```

## Single-Image Fitter

If you already have one local SPHEREx FITS file, use `spiff-fit`.

```bash
spiff-fit \
  --fits /path/to/level2_file.fits \
  --ra 10.684708 \
  --dec 41.268750 \
  --fit-radius 4.0 \
  --save-figs \
  --figs-dir ./single_fit_output \
  --scipy-only
```

To save the JSON result dictionary:

```bash
spiff-fit \
  --fits /path/to/level2_file.fits \
  --ra 10.684708 \
  --dec 41.268750 \
  --fit-radius 4.0 \
  --save-results \
  --results-path ./single_fit_output/result.json \
  --scipy-only
```

## Output Notes

`results.csv` from `spiff-lv2` is the main machine-readable output. It includes:

- input target coordinates
- projected coordinates used for each exposure
- FITS metadata
- aperture metrics
- SciPy PSF-fit metrics
- UltraNest PSF-fit metrics when available

For autotype, the most important columns are:

- `psf_un_wv_um`
- `psf_un_flux_uJy`
- `psf_un_flux_uJy_err`

If you ran with `--scipy-only`, the `psf_un_*` columns may be empty or less useful for downstream autotyping. In that case, inspect the output carefully before treating it as a final spectrum.

## Bundled Local Assets

SPIFF ships two small local assets so autotype works without any database access:

- `src/spiff/data/spherex_templates.joblib`
  - compact SPHEREx template bundle
- `src/spiff/data/spherex_bins.csv`
  - exported SPHEREx bin-center table

No external template or bin lookup is required at runtime.

## Troubleshooting

### `ultranest` import error

Install the optional extra or use SciPy-only mode:

```bash
pip install ".[ultranest]"
```

or

```bash
spiff-fit ... --scipy-only
spiff-lv2 ... --scipy-only
```

### `spiff-autotype --bin` fails

`--bin` only works for SPIFF-style CSV inputs that contain:

- `wavelength_um`, `flux_ujy`, `flux_err_ujy`, or
- `psf_un_wv_um`, `psf_un_flux_uJy`, `psf_un_flux_uJy_err`

It is not meant for already-reduced `wavelength_angstrom / flux_flambda / flux_flambda_unc` inputs.

### No valid autotype result

Common causes:

- too few finite points after cleaning
- zero or negative uncertainties
- too many rows marked `ignored`
- a CSV that is not in one of the supported formats

### Network or discovery failures in `spiff-lv2`

`spiff-lv2` depends on live IRSA services and SPHEREx data files. Transient failures can happen during:

- TAP / SIA2 queries
- DataLink resolution
- FITS download
- S3 cutout access

Re-running the same command is often enough for transient failures.

## Development

Install with test dependencies:

```bash
pip install -e ".[dev]"
pytest -q
```
