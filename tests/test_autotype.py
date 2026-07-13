from pathlib import Path

import pandas as pd

from spiff.autotype import _load_spherex_bins, _load_templates, main


def test_local_assets_load() -> None:
    bins = _load_spherex_bins()
    templates = _load_templates()

    assert not bins.empty
    assert templates


def test_autotype_example_runs_without_plot(tmp_path: Path, capsys) -> None:
    spectrum = pd.DataFrame(
        {
            "wavelength_angstrom": [
                7500,
                9000,
                10500,
                12000,
                13500,
                15000,
                17000,
                19000,
                22000,
                26000,
                30000,
                36000,
                42000,
                48000,
            ],
            "flux_flambda": [
                1.00e-17,
                1.08e-17,
                1.15e-17,
                1.11e-17,
                1.05e-17,
                9.90e-18,
                9.40e-18,
                8.95e-18,
                8.30e-18,
                7.75e-18,
                7.20e-18,
                6.70e-18,
                6.30e-18,
                5.95e-18,
            ],
            "flux_flambda_unc": [
                1.50e-18,
                1.45e-18,
                1.40e-18,
                1.35e-18,
                1.30e-18,
                1.28e-18,
                1.25e-18,
                1.22e-18,
                1.20e-18,
                1.18e-18,
                1.16e-18,
                1.15e-18,
                1.14e-18,
                1.13e-18,
            ],
            "ignored": [0] * 14,
        }
    )
    work_csv = tmp_path / "example.csv"
    spectrum.to_csv(work_csv, index=False)

    rc = main(["--csv", str(work_csv), "--no-plot", "--no-save-plot"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Best match:" in captured.out


def test_autotype_supports_raw_spiff_style_csv(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "psf_un_wv_um": [0.744, 0.766, 0.788, 0.810, 0.832],
            "psf_un_flux_uJy": [20.0, 21.0, 22.0, 21.5, 20.5],
            "psf_un_flux_uJy_err": [2.0, 2.0, 2.0, 2.0, 2.0],
            "psf_un_snr": [10.0, 10.5, 11.0, 10.75, 10.25],
            "n_pix_used_in_fit": [10, 10, 10, 10, 10],
        }
    )
    csv_path = tmp_path / "raw_spiff.csv"
    raw.to_csv(csv_path, index=False)

    rc = main(["--csv", str(csv_path), "--bin", "--no-plot", "--no-save-plot"])
    assert rc == 0


def test_autotype_supports_scipy_only_spiff_csv(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "psf_un_wv_um": [float("nan")] * 5,
            "psf_un_flux_uJy": [float("nan")] * 5,
            "psf_un_flux_uJy_err": [float("nan")] * 5,
            "psf_scipy_wv_um": [0.744, 0.766, 0.788, 0.810, 0.832],
            "psf_scipy_flux_uJy": [20.0, 21.0, 22.0, 21.5, 20.5],
            "psf_scipy_flux_uJy_err": [2.0, 2.0, 2.0, 2.0, 2.0],
            "psf_scipy_snr": [10.0, 10.5, 11.0, 10.75, 10.25],
            "n_pix_used_in_fit": [10, 10, 10, 10, 10],
        }
    )
    csv_path = tmp_path / "scipy_only_spiff.csv"
    raw.to_csv(csv_path, index=False)

    rc = main(["--csv", str(csv_path), "--bin", "--no-plot", "--no-save-plot"])
    assert rc == 0
