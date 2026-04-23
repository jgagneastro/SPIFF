"""Runnable examples for SPIFF."""

from __future__ import annotations

import shlex
from pathlib import Path


SIMP_J0136_TARGET = {
    "target_name": "SIMP J013656.5+093347.3",
    "safe_target_name": "SIMP-J013656.5093347.3",
    "ra_deg": 24.24124978791965,
    "dec_deg": 9.563070454755835,
    "reference_crd_epoch_yr": 2016.0,
    "reference_pmra_masyr": 1238.239990234375,
    "reference_pmdec_masyr": -16.15559959411621,
    "spectral_type": "T2.5",
}


def simp_j0136_target_dir(base_outdir: str | Path = "spiff_examples") -> Path:
    base = Path(base_outdir).expanduser().resolve()
    return base / (
        f"{SIMP_J0136_TARGET['safe_target_name']}"
        f"_RA{SIMP_J0136_TARGET['ra_deg']:.6f}"
        f"_DEC{SIMP_J0136_TARGET['dec_deg']:.6f}"
    )


def build_simp_j0136_test_commands(base_outdir: str | Path = "spiff_examples") -> list[str]:
    base = Path(base_outdir).expanduser()
    target_dir = simp_j0136_target_dir(base)
    compiled_csv = target_dir / "compiled_results.csv"
    return [
        " ".join(
            [
                "spiff-lv2",
                f"--ra {SIMP_J0136_TARGET['ra_deg']}",
                f"--dec {SIMP_J0136_TARGET['dec_deg']}",
                f"--reference-crd-epoch-yr {SIMP_J0136_TARGET['reference_crd_epoch_yr']}",
                f"--reference-pmra-masyr {SIMP_J0136_TARGET['reference_pmra_masyr']}",
                f"--reference-pmdec-masyr {SIMP_J0136_TARGET['reference_pmdec_masyr']}",
                f"--target-name {shlex.quote(SIMP_J0136_TARGET['target_name'])}",
                f"--outdir {shlex.quote(str(base))}",
                "--save-figs",
            ]
        ),
        f"spiff-compile-results --outdir {shlex.quote(str(target_dir))}",
        f"spiff-autotype --csv {shlex.quote(str(compiled_csv))} --no-plot",
    ]


def run_simp_j0136_smoke_test(
    base_outdir: str | Path = "spiff_examples",
    *,
    save_figs: bool = True,
    show_plot: bool = False,
    save_plot: bool = True,
    scipy_only: bool = False,
) -> dict[str, str]:
    from .results import compile_results_csvs
    from .lv2 import cli_main as lv2_main
    from .autotype import main as autotype_main

    base = Path(base_outdir).expanduser().resolve()
    target_dir = simp_j0136_target_dir(base)
    lv2_args = [
        "--ra",
        str(SIMP_J0136_TARGET["ra_deg"]),
        "--dec",
        str(SIMP_J0136_TARGET["dec_deg"]),
        "--reference-crd-epoch-yr",
        str(SIMP_J0136_TARGET["reference_crd_epoch_yr"]),
        "--reference-pmra-masyr",
        str(SIMP_J0136_TARGET["reference_pmra_masyr"]),
        "--reference-pmdec-masyr",
        str(SIMP_J0136_TARGET["reference_pmdec_masyr"]),
        "--target-name",
        str(SIMP_J0136_TARGET["target_name"]),
        "--outdir",
        str(base),
    ]
    if save_figs:
        lv2_args.append("--save-figs")
    if scipy_only:
        lv2_args.append("--scipy-only")

    lv2_rc = lv2_main(lv2_args)
    if lv2_rc != 0:
        raise RuntimeError(f"spiff-lv2 failed for SIMP J0136 with exit code {lv2_rc}")

    compiled_csv = compile_results_csvs(str(target_dir))
    autotype_args = ["--csv", compiled_csv]
    if not show_plot:
        autotype_args.append("--no-plot")
    if not save_plot:
        autotype_args.append("--no-save-plot")

    autotype_rc = autotype_main(autotype_args)
    if autotype_rc != 0:
        raise RuntimeError(f"spiff-autotype failed for SIMP J0136 with exit code {autotype_rc}")

    return {
        "target_dir": str(target_dir),
        "compiled_csv": str(compiled_csv),
    }
