# SIMP J0136 Comparison Products

These files are committed as a small reference set for SPIFF comparison work.

Files:

- `simp_j0136_unbinned.csv`
  - Combined non-binned SPIFF output from the per-image `figs/*/result.csv` files.
- `simp_j0136_binned.csv`
  - The same spectrum binned onto the bundled SPHEREx wavelength centers with the pipeline's nearest-bin inverse-variance weighted method.
- `simp_j0136_autotype_binned.png`
  - `spiff-autotype` comparison plot produced from `simp_j0136_binned.csv`.

Source run:

- `runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070`

Commands used:

```bash
python -m spiff.results \
  --outdir runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070

python -m spiff.autotype \
  --csv runs/SIMP-J013656.5093347.3_RA24.241250_DEC9.563070/binned_spectrum.csv \
  --plot-file example_products/simp_j0136/simp_j0136_autotype_binned.png \
  --no-plot \
  --overwrite
```
