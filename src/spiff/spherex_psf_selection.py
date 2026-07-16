"""Select the detector-position-dependent SPHEREx PSF plane safely."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_CENTER_KEY = re.compile(r"^(?P<axis>[XY])CTR_(?P<plane>\d+)$")


@dataclass(frozen=True)
class SpherexPsfSelection:
    """A selected PSF plane and the coordinates used to select it."""

    image: np.ndarray
    plane_index: int
    oversamp: int
    detector_x_px: float
    detector_y_px: float
    zone_x_index: int | None
    zone_y_index: int | None
    zone_x_center_px: float | None
    zone_y_center_px: float | None
    ncols: int | None
    nrows: int | None
    coordinate_transform: str
    selection_method: str


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def detector_pixel_coordinates(
    xpix: float,
    ypix: float,
    image_header: Mapping[str, Any] | None = None,
) -> tuple[float, float, str]:
    """Convert zero-based image pixels to zero-based parent-detector pixels."""
    x_local = _finite_float(xpix)
    y_local = _finite_float(ypix)
    if x_local is None or y_local is None:
        raise ValueError("PSF selection coordinates must be finite")

    if image_header is None:
        return x_local, y_local, "identity"

    spx_x = _finite_float(image_header.get("SPXORX0"))
    spx_y = _finite_float(image_header.get("SPXORY0"))
    if (spx_x is None) != (spx_y is None):
        raise ValueError("Incomplete SPXORX0/SPXORY0 detector-origin metadata")
    if spx_x is not None and spx_y is not None:
        return x_local + spx_x, y_local + spx_y, "SPXORX0/SPXORY0"

    crpix1a = _finite_float(image_header.get("CRPIX1A"))
    crpix2a = _finite_float(image_header.get("CRPIX2A"))
    if (crpix1a is None) != (crpix2a is None):
        raise ValueError("Incomplete CRPIX1A/CRPIX2A detector-origin metadata")
    if crpix1a is not None and crpix2a is not None:
        return (
            1.0 + x_local - crpix1a,
            1.0 + y_local - crpix2a,
            "CRPIX1A/CRPIX2A",
        )

    return x_local, y_local, "identity"


def _axis_centers(header: Mapping[str, Any], axis: str) -> np.ndarray:
    values: list[float] = []
    for key in header:
        match = _CENTER_KEY.match(str(key).upper())
        if match is None or match.group("axis") != axis:
            continue
        value = _finite_float(header[key])
        if value is not None:
            values.append(value)

    centers: list[float] = []
    for value in sorted(values):
        if not centers or not np.isclose(value, centers[-1], rtol=0.0, atol=1.0e-6):
            centers.append(value)
    return np.asarray(centers, dtype=float)


def select_spherex_psf(
    psf_cube: np.ndarray,
    psf_header: Mapping[str, Any],
    xpix: float,
    ypix: float,
    *,
    image_header: Mapping[str, Any] | None = None,
) -> SpherexPsfSelection:
    """Select the nearest SPHEREx PSF using the cube's true x-fast ordering.

    Historical SPHEREx products paired the per-plane ``XCTR_n``/``YCTR_n``
    cards in y-fast order even though the cube is x-fast. Reconstructing the
    independent detector axes makes this work with both old and fixed headers.
    """
    cube = np.asarray(psf_cube)
    if cube.ndim != 3 or cube.shape[0] <= 0:
        raise ValueError(f"Expected a non-empty 3-D PSF cube, got shape {cube.shape}")

    oversamp = int(psf_header.get("OVERSAMP", 10))
    if oversamp <= 0:
        raise ValueError(f"Invalid PSF oversampling factor: {oversamp}")

    detector_x, detector_y, transform = detector_pixel_coordinates(
        xpix,
        ypix,
        image_header,
    )
    x_centers = _axis_centers(psf_header, "X")
    y_centers = _axis_centers(psf_header, "Y")

    if x_centers.size == 0 and y_centers.size == 0:
        plane_index = int(cube.shape[0] // 2)
        return SpherexPsfSelection(
            image=np.asarray(cube[plane_index], dtype=float),
            plane_index=plane_index,
            oversamp=oversamp,
            detector_x_px=detector_x,
            detector_y_px=detector_y,
            zone_x_index=None,
            zone_y_index=None,
            zone_x_center_px=None,
            zone_y_center_px=None,
            ncols=None,
            nrows=None,
            coordinate_transform=transform,
            selection_method="central-plane-fallback",
        )
    if x_centers.size == 0 or y_centers.size == 0:
        raise ValueError("Incomplete XCTR_n/YCTR_n PSF-center metadata")

    ncols = int(x_centers.size)
    nrows = int(y_centers.size)
    expected_planes = ncols * nrows
    if expected_planes != int(cube.shape[0]):
        raise ValueError(
            "PSF-center grid does not match cube: "
            f"{ncols} x {nrows} centers imply {expected_planes} planes, "
            f"but cube has {cube.shape[0]}"
        )

    ix = int(np.argmin(np.abs(x_centers - detector_x)))
    iy = int(np.argmin(np.abs(y_centers - detector_y)))
    plane_index = int(iy * ncols + ix)
    return SpherexPsfSelection(
        image=np.asarray(cube[plane_index], dtype=float),
        plane_index=plane_index,
        oversamp=oversamp,
        detector_x_px=detector_x,
        detector_y_px=detector_y,
        zone_x_index=ix,
        zone_y_index=iy,
        zone_x_center_px=float(x_centers[ix]),
        zone_y_center_px=float(y_centers[iy]),
        ncols=ncols,
        nrows=nrows,
        coordinate_transform=transform,
        selection_method="x-fast-cartesian-grid",
    )
