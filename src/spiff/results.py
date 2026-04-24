#!/usr/bin/env python3
"""
Result compilation utilities for SPIFF.

Currently includes:
  - compile_results_csvs: combine per-image figs/*/results.csv into one compiled CSV.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .spherex_binning import write_binned_spherex_spectrum_csv


def compile_results_csvs(
    outdir: str,
    output_csv: Optional[str] = None,
    *,
    write_binned_spectrum: bool = False,
    binned_output_csv: Optional[str] = None,
) -> str:
    """Combine per-image result.csv/results.csv files under outdir/figs/*/ into one CSV."""
    base = Path(outdir).resolve()
    figs_dir = base / "figs"
    if not figs_dir.exists() or not figs_dir.is_dir():
        raise FileNotFoundError(f"figs/ not found under {base}")

    patterns = [
        str(figs_dir / "*" / "result.csv"),
        str(figs_dir / "*" / "results.csv"),
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No per-image results.csv found under {figs_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            raise RuntimeError(f"Failed reading {f}: {exc}") from exc

    if not frames:
        raise ValueError(f"All results.csv files under {figs_dir} were empty.")

    out = pd.concat(frames, ignore_index=True)
    if "psf_un_wv_um" in out.columns:
        out = out.sort_values("psf_un_wv_um").reset_index(drop=True)

    if output_csv is None:
        output_csv = str(base / "compiled_results.csv")

    out.to_csv(output_csv, index=False)
    if write_binned_spectrum:
        if binned_output_csv is None:
            binned_output_csv = str(base / "binned_spectrum.csv")
        write_binned_spherex_spectrum_csv(out, binned_output_csv)
    return output_csv


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Combine per-image results.csv files into one CSV")
    ap.add_argument("--outdir", required=True, help="Parent directory that contains figs/*/results.csv")
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Output compiled CSV path (default: outdir/compiled_results.csv)",
    )
    ap.add_argument(
        "--write-binned-spectrum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write OUTDIR/binned_spectrum.csv using the SQL-equivalent SPHEREx binning method (default: on).",
    )
    ap.add_argument(
        "--binned-output-csv",
        default=None,
        help="Output path for --write-binned-spectrum (default: outdir/binned_spectrum.csv).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    out = compile_results_csvs(
        args.outdir,
        args.output_csv,
        write_binned_spectrum=bool(args.write_binned_spectrum),
        binned_output_csv=args.binned_output_csv,
    )
    print(f"Wrote {out}")
    if args.write_binned_spectrum:
        binned_out = args.binned_output_csv or str(Path(args.outdir).resolve() / "binned_spectrum.csv")
        print(f"Wrote {Path(binned_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
