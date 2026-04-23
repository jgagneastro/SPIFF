from pathlib import Path

import pandas as pd

from spiff.results import compile_results_csvs


def test_compile_results_csvs_supports_result_and_results_names(tmp_path: Path) -> None:
    figs = tmp_path / "figs"
    first = figs / "a"
    second = figs / "b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    pd.DataFrame(
        [{"psf_un_wv_um": 2.0, "psf_un_flux_uJy": 20.0, "psf_un_flux_uJy_err": 2.0}]
    ).to_csv(first / "result.csv", index=False)
    pd.DataFrame(
        [{"psf_un_wv_um": 1.0, "psf_un_flux_uJy": 10.0, "psf_un_flux_uJy_err": 1.0}]
    ).to_csv(second / "results.csv", index=False)

    output = compile_results_csvs(str(tmp_path))
    frame = pd.read_csv(output)

    assert list(frame["psf_un_wv_um"]) == [1.0, 2.0]
