# SPIFF

SPIFF provides the SPHEREx level-2 batch workflow, single-image fitting tools, result compilation, and local autotyping against bundled templates.

## Citation

If you use SPIFF in published work, please cite:

J. Gagné et al. (2026, submitted to ApJ) https://arxiv.org/pdf/2604.22012

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
  - Discover nearby SPHEREx level-2 observations, download FITS files or cutouts, run the per-image fitter, and write `binned_spectrum.csv` by default.
- `spiff-fit`
  - Run the single-image fitter on one local FITS file.
- `spiff-compile-results`
  - Combine `figs/*/result.csv` or `figs/*/results.csv` into one `compiled_results.csv` and optionally regenerate `binned_spectrum.csv`.
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

By default, `spiff-lv2` uses anonymous S3 byte-range reads and saves a
20×20-pixel cutout around the epoch-adjusted target position. Use
`--no-s3-cutout` only when a full Level 2 FITS file is explicitly required.

Inside it you will typically see:

- `results.csv`
- `binned_spectrum.csv`
- `results.jsonl`
- `download_manifest.csv`
- `fits/`
- `figs/`

### 2. Run autotype on the default binned SPIFF spectrum

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/binned_spectrum.csv \
  --no-plot
```

If you want the diagnostic plot saved next to the CSV, omit `--no-save-plot`.

### 3. Optional: compile the raw per-image result files

```bash
spiff-compile-results \
  --outdir ./runs/M31_demo_RA10.684708_DEC41.268750
```

This writes:

```text
./runs/M31_demo_RA10.684708_DEC41.268750/compiled_results.csv
```

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

spiff-autotype \
  --csv ./runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070/binned_spectrum.csv \
  --no-plot
```

If you prefer Python over shell commands, the same flow is available as a helper function:

```python
from spiff.examples import run_simp_j0136_smoke_test

result = run_simp_j0136_smoke_test("./runs")
print(result["binned_csv"])
```

If you only want the copy-paste shell commands, you can generate them programmatically:

```python
from spiff.examples import build_simp_j0136_test_commands

for command in build_simp_j0136_test_commands("./runs"):
    print(command)
```

## Committed Example Products

The repo now includes a small committed SIMP J0136 comparison set under:

```text
example_products/simp_j0136/
```

It contains:

- `simp_j0136_unbinned.csv`
  - combined non-binned SPIFF CSV
- `simp_j0136_binned.csv`
  - SPHEREx-bin-weighted reduced spectrum
- `simp_j0136_autotype_binned.png`
  - autotype comparison plot produced from the binned CSV

That gives public users a stable three-way comparison target without having to rerun the full `lv2` workflow first.

## Autotype Inputs

`spiff-autotype` supports these local CSV shapes.

### Raw SPIFF-style CSV

The preferred UltraNest columns are:

- `psf_un_wv_um`
- `psf_un_flux_uJy`
- `psf_un_flux_uJy_err`

For a `--scipy-only` run, SPIFF automatically uses:

- `psf_scipy_wv_um`
- `psf_scipy_flux_uJy`
- `psf_scipy_flux_uJy_err`

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

SPIFF writes `binned_spectrum.csv` in this reduced format by default at the end of `spiff-lv2`.

## Useful Autotype Options

### Overplot local binning without fitting the binned spectrum

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/results.csv \
  --overplotbin \
  --no-plot
```

### Fit the locally binned spectrum

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/results.csv \
  --bin \
  --no-plot
```

### Overlay specific template types

```bash
spiff-autotype \
  --csv ./runs/M31_demo_RA10.684708_DEC41.268750/binned_spectrum.csv \
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

`results.csv` from `spiff-lv2` is the main raw machine-readable output. It includes:

- input target coordinates
- projected coordinates used for each exposure
- FITS metadata
- aperture metrics
- SciPy PSF-fit metrics
- UltraNest PSF-fit metrics when available

`binned_spectrum.csv` is the default reduced-spectrum product. It uses the same nearest-SPHEREx-bin weighted mean and weighted standard-error method as the SQL workflow, then converts the bin-averaged `uJy` fluxes to `F_lambda`.

For autotype, the most important columns are:

- `psf_un_wv_um`
- `psf_un_flux_uJy`
- `psf_un_flux_uJy_err`

If you ran with `--scipy-only`, SPIFF uses the corresponding populated
`psf_scipy_*` wavelength, flux, uncertainty, and S/N columns when it creates
`binned_spectrum.csv`. UltraNest results remain preferred when they are populated.

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
- `psf_un_wv_um`, `psf_un_flux_uJy`, `psf_un_flux_uJy_err`, or
- `psf_scipy_wv_um`, `psf_scipy_flux_uJy`, `psf_scipy_flux_uJy_err`

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
