from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from spiff.spherex_psf_selection import select_spherex_psf


def _numbered_cube(nplanes: int) -> np.ndarray:
    return np.stack(
        [np.full((4, 4), float(index), dtype=float) for index in range(nplanes)]
    )


def _grid_header(xs: list[float], ys: list[float], *, incorrect_pairing: bool) -> fits.Header:
    header = fits.Header()
    header["OVERSAMP"] = 2
    if incorrect_pairing:
        pairs = [(x, y) for x in xs for y in ys]
    else:
        pairs = [(x, y) for y in ys for x in xs]
    for index, (x, y) in enumerate(pairs, start=1):
        header[f"XCTR_{index}"] = x
        header[f"YCTR_{index}"] = y
    return header


@pytest.mark.parametrize("incorrect_pairing", [False, True])
def test_selection_uses_actual_x_fast_cube_order(incorrect_pairing: bool) -> None:
    cube = _numbered_cube(6)
    header = _grid_header(
        [10.0, 20.0, 30.0],
        [100.0, 200.0],
        incorrect_pairing=incorrect_pairing,
    )

    selection = select_spherex_psf(cube, header, 29.0, 101.0)

    assert selection.plane_index == 2
    assert selection.zone_x_index == 2
    assert selection.zone_y_index == 0
    assert selection.ncols == 3
    assert selection.nrows == 2
    assert np.all(selection.image == 2.0)


def test_selection_adds_custom_cutout_detector_origin() -> None:
    cube = _numbered_cube(4)
    header = _grid_header([10.0, 30.0], [100.0, 200.0], incorrect_pairing=False)
    image_header = fits.Header({"SPXORX0": 28, "SPXORY0": 197})

    selection = select_spherex_psf(
        cube,
        header,
        2.0,
        3.0,
        image_header=image_header,
    )

    assert selection.detector_x_px == 30.0
    assert selection.detector_y_px == 200.0
    assert selection.coordinate_transform == "SPXORX0/SPXORY0"
    assert selection.plane_index == 3


def test_selection_supports_irsa_cutout_alternate_wcs_origin() -> None:
    cube = _numbered_cube(4)
    header = _grid_header([10.0, 30.0], [100.0, 200.0], incorrect_pairing=False)
    image_header = fits.Header({"CRPIX1A": -25.0, "CRPIX2A": -194.0})

    selection = select_spherex_psf(
        cube,
        header,
        4.0,
        5.0,
        image_header=image_header,
    )

    assert selection.detector_x_px == 30.0
    assert selection.detector_y_px == 200.0
    assert selection.coordinate_transform == "CRPIX1A/CRPIX2A"
    assert selection.plane_index == 3


def test_selection_falls_back_to_central_plane_without_center_cards() -> None:
    selection = select_spherex_psf(
        _numbered_cube(5),
        fits.Header({"OVERSAMP": 2}),
        1.0,
        2.0,
    )

    assert selection.plane_index == 2
    assert selection.selection_method == "central-plane-fallback"


def test_selection_rejects_center_grid_that_does_not_match_cube() -> None:
    header = _grid_header([10.0, 20.0], [100.0, 200.0], incorrect_pairing=False)

    with pytest.raises(ValueError, match="grid does not match cube"):
        select_spherex_psf(_numbered_cube(5), header, 10.0, 100.0)
