import numpy as np
import pandas as pd

from spiff.spherex_binning import build_binned_spherex_spectrum


def _to_flambda_from_ujy(wavelength_um: float, flux_ujy: float) -> float:
    wavelength_m = wavelength_um * 1e-6
    flux_jy = flux_ujy * 1e-6
    return flux_jy * 1e-32 * 299792458.0 / (wavelength_m * wavelength_m) * 1e-10


def test_build_binned_spherex_spectrum_matches_sql_like_weighting() -> None:
    raw = pd.DataFrame(
        {
            "psf_un_wv_um": [0.743, 0.747, 0.745, 0.765, 0.790, 0.810],
            "psf_un_flux_uJy": [10.0, 14.0, 99.0, -6.0, 5.0, 8.0],
            "psf_un_flux_uJy_err": [2.0, 1.0, 0.0, 3.0, 0.5, 2.0],
            "psf_un_snr": [2.0, 14.0, 99.0, 0.2, 0.05, 0.5],
            "n_pix_used_in_fit": [4, 5, 6, 3, 10, 2],
        }
    )

    binned = build_binned_spherex_spectrum(raw)

    assert list(np.round(binned["bin_center_um"], 3)) == [0.744, 0.766]

    first = binned.iloc[0]
    expected_mean_1 = (10.0 / (2.0**2) + 14.0 / (1.0**2)) / (1.0 / (2.0**2) + 1.0 / (1.0**2))
    expected_sem_1 = np.sqrt(1.0 / (1.0 / (2.0**2) + 1.0 / (1.0**2)))
    assert np.isclose(first["flux_ujy_weighted_mean"], expected_mean_1)
    assert np.isclose(first["flux_ujy_weighted_sem"], expected_sem_1)
    assert np.isclose(first["flux_flambda"], _to_flambda_from_ujy(0.744, expected_mean_1))
    assert np.isclose(first["flux_flambda_unc"], _to_flambda_from_ujy(0.744, expected_sem_1))

    second = binned.iloc[1]
    assert np.isclose(second["flux_ujy_weighted_mean"], -6.0)
    assert np.isclose(second["flux_ujy_weighted_sem"], 3.0)
    assert np.isclose(second["flux_flambda"], _to_flambda_from_ujy(0.766, -6.0))
    assert int(second["n_points"]) == 1
