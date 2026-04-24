from pathlib import Path

import pandas as pd


def test_public_simp_j0136_example_products_exist_and_have_expected_shape() -> None:
    base = Path(__file__).resolve().parents[1] / "example_products" / "simp_j0136"

    unbinned = base / "simp_j0136_unbinned.csv"
    binned = base / "simp_j0136_binned.csv"
    autotype_png = base / "simp_j0136_autotype_binned.png"

    assert unbinned.exists()
    assert binned.exists()
    assert autotype_png.exists()
    assert autotype_png.stat().st_size > 0

    unbinned_df = pd.read_csv(unbinned)
    binned_df = pd.read_csv(binned)

    assert {"psf_un_wv_um", "psf_un_flux_uJy", "psf_un_flux_uJy_err"}.issubset(unbinned_df.columns)
    assert {
        "bin_id",
        "bin_center_um",
        "wavelength_angstrom",
        "flux_ujy_weighted_mean",
        "flux_ujy_weighted_sem",
        "flux_flambda",
        "flux_flambda_unc",
    }.issubset(binned_df.columns)
    assert not unbinned_df.empty
    assert not binned_df.empty
