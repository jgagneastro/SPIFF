"""Shared SPHEREx binning helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SPEED_OF_LIGHT = 299792458.0
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_BINS_PATH = DATA_DIR / "spherex_bins.csv"


def load_spherex_bins() -> pd.DataFrame:
    bins = pd.read_csv(DEFAULT_BINS_PATH)
    if bins.empty:
        raise ValueError("No local SPHEREx bin rows found.")
    bins = bins.rename(
        columns={
            "index": "id",
            "wavelength_center_um": "wavelength_um",
            "wavelength_width_um": "wavelength_wid_um",
        }
    )
    bins["id"] = pd.to_numeric(bins["id"], errors="coerce").astype(int)
    bins["wavelength_um"] = pd.to_numeric(bins["wavelength_um"], errors="coerce")
    if "wavelength_wid_um" in bins.columns:
        bins["wavelength_wid_um"] = pd.to_numeric(bins["wavelength_wid_um"], errors="coerce")
    bins = bins.dropna(subset=["wavelength_um"]).sort_values("wavelength_um", kind="stable").reset_index(drop=True)
    if bins.empty:
        raise ValueError("No valid local SPHEREx bin wavelengths found.")
    return bins


def nearest_bin_indices(bin_centers_um: np.ndarray, wavelengths_um: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bin_centers_um, wavelengths_um)
    idx = np.clip(idx, 1, len(bin_centers_um) - 1)
    left = idx - 1
    right = idx
    dist_left = np.abs(bin_centers_um[left] - wavelengths_um)
    dist_right = np.abs(bin_centers_um[right] - wavelengths_um)
    use_right = dist_right < dist_left
    return np.where(use_right, right, left)


def _to_flambda_from_ujy(wavelength_um: np.ndarray, flux_ujy: np.ndarray) -> np.ndarray:
    wavelength_m = wavelength_um * 1e-6
    flux_jy = flux_ujy * 1e-6
    return flux_jy * 1e-32 * SPEED_OF_LIGHT / np.square(wavelength_m) * 1e-10


def _resolve_input_columns(frame: pd.DataFrame) -> tuple[str, str, str, str | None, str | None]:
    columns = set(frame.columns)
    if {"psf_un_wv_um", "psf_un_flux_uJy", "psf_un_flux_uJy_err"}.issubset(columns):
        snr_col = "psf_un_snr" if "psf_un_snr" in columns else None
        n_pix_col = "n_pix_used_in_fit" if "n_pix_used_in_fit" in columns else None
        return "psf_un_wv_um", "psf_un_flux_uJy", "psf_un_flux_uJy_err", snr_col, n_pix_col
    if {"wavelength_um", "flux_ujy", "flux_err_ujy"}.issubset(columns):
        snr_col = "snr" if "snr" in columns else ("flux_snr" if "flux_snr" in columns else None)
        n_pix_col = "n_pix_used_in_fit" if "n_pix_used_in_fit" in columns else None
        return "wavelength_um", "flux_ujy", "flux_err_ujy", snr_col, n_pix_col
    raise ValueError(
        "Binning requires SPIFF-like columns "
        "['psf_un_wv_um', 'psf_un_flux_uJy', 'psf_un_flux_uJy_err'] or "
        "['wavelength_um', 'flux_ujy', 'flux_err_ujy']."
    )


def build_binned_spherex_spectrum(
    frame: pd.DataFrame,
    *,
    min_snr: float = 0.1,
    min_n_pix_used_in_fit: int = 3,
) -> pd.DataFrame:
    """Bin SPIFF-like rows with the same nearest-bin weighted averaging used in SQL."""

    wavelength_col, flux_col, flux_err_col, snr_col, n_pix_col = _resolve_input_columns(frame)

    work = frame.copy()
    work[wavelength_col] = pd.to_numeric(work[wavelength_col], errors="coerce")
    work[flux_col] = pd.to_numeric(work[flux_col], errors="coerce")
    work[flux_err_col] = pd.to_numeric(work[flux_err_col], errors="coerce")

    keep = (
        np.isfinite(work[wavelength_col].to_numpy(dtype=float))
        & np.isfinite(work[flux_col].to_numpy(dtype=float))
        & np.isfinite(work[flux_err_col].to_numpy(dtype=float))
        & (work[wavelength_col].to_numpy(dtype=float) > 0)
    )
    if n_pix_col is not None:
        work[n_pix_col] = pd.to_numeric(work[n_pix_col], errors="coerce")
        keep &= np.isfinite(work[n_pix_col].to_numpy(dtype=float))
        keep &= work[n_pix_col].to_numpy(dtype=float) >= float(min_n_pix_used_in_fit)
    if snr_col is not None:
        work[snr_col] = pd.to_numeric(work[snr_col], errors="coerce")
        keep &= np.isfinite(work[snr_col].to_numpy(dtype=float))
        keep &= work[snr_col].to_numpy(dtype=float) >= float(min_snr)
    work = work.loc[keep].copy()
    if work.empty:
        raise ValueError("No valid SPIFF-like rows remain for SPHEREx binning after filtering.")

    valid_weight = np.isfinite(work[flux_err_col].to_numpy(dtype=float)) & (work[flux_err_col].to_numpy(dtype=float) > 0)
    work = work.loc[valid_weight].copy()
    if work.empty:
        raise ValueError("No SPIFF-like rows with positive finite flux errors remain for SPHEREx binning.")

    bins = load_spherex_bins()
    bin_centers = bins["wavelength_um"].to_numpy(dtype=float)
    if len(bin_centers) < 2:
        raise ValueError("Need at least two local SPHEREx bins.")

    nearest_idx = nearest_bin_indices(bin_centers, work[wavelength_col].to_numpy(dtype=float))
    work["bin_idx"] = nearest_idx
    work["weight"] = 1.0 / np.square(work[flux_err_col].to_numpy(dtype=float))
    work["flux_over_var"] = work[flux_col].to_numpy(dtype=float) * work["weight"].to_numpy(dtype=float)

    grouped = (
        work.groupby("bin_idx", sort=True, as_index=False)
        .agg(
            sum_w=("weight", "sum"),
            sum_fw=("flux_over_var", "sum"),
            n_points=("bin_idx", "size"),
        )
        .sort_values("bin_idx", kind="stable")
        .reset_index(drop=True)
    )
    grouped = grouped[grouped["sum_w"] > 0].copy()
    if grouped.empty:
        raise ValueError("No occupied SPHEREx bins remain after weighted aggregation.")

    grouped["flux_ujy_weighted_mean"] = grouped["sum_fw"] / grouped["sum_w"]
    grouped["flux_ujy_weighted_sem"] = np.sqrt(1.0 / grouped["sum_w"])

    grouped["bin_id"] = bins.iloc[grouped["bin_idx"].to_numpy(dtype=int)]["id"].to_numpy(dtype=int)
    grouped["bin_center_um"] = bins.iloc[grouped["bin_idx"].to_numpy(dtype=int)]["wavelength_um"].to_numpy(dtype=float)
    if "wavelength_wid_um" in bins.columns:
        grouped["bin_width_um"] = bins.iloc[grouped["bin_idx"].to_numpy(dtype=int)]["wavelength_wid_um"].to_numpy(dtype=float)
    else:
        grouped["bin_width_um"] = np.nan

    grouped["wavelength_angstrom"] = grouped["bin_center_um"] * 1e4
    grouped["flux_flambda"] = _to_flambda_from_ujy(
        grouped["bin_center_um"].to_numpy(dtype=float),
        grouped["flux_ujy_weighted_mean"].to_numpy(dtype=float),
    )
    grouped["flux_flambda_unc"] = _to_flambda_from_ujy(
        grouped["bin_center_um"].to_numpy(dtype=float),
        grouped["flux_ujy_weighted_sem"].to_numpy(dtype=float),
    )
    grouped["ignored"] = 0

    return grouped[
        [
            "bin_id",
            "bin_center_um",
            "bin_width_um",
            "n_points",
            "wavelength_angstrom",
            "flux_ujy_weighted_mean",
            "flux_ujy_weighted_sem",
            "flux_flambda",
            "flux_flambda_unc",
            "ignored",
        ]
    ].sort_values("bin_center_um", kind="stable").reset_index(drop=True)


def write_binned_spherex_spectrum_csv(frame: pd.DataFrame, output_csv: str | Path) -> str:
    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    binned = build_binned_spherex_spectrum(frame)
    binned.to_csv(out_path, index=False)
    return str(out_path)
