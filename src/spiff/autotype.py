#!/usr/bin/env python3
"""
Autotype a comparison spectrum against SPHEREx templates.

Input modes:
  - Local CSV from SPIFF pipeline output
  - Local reduced-spectrum CSV with wavelength / flux / uncertainty columns
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from .spherex_binning import build_binned_spherex_spectrum, load_spherex_bins


SPEED_OF_LIGHT = 299792458.0
MAX_TEMPLATE_DIST_ANGSTROM = 100.0
MIN_MATCH_POINTS = 3
PEC_A = -0.25380665466370705
PEC_B = 0.019559107265402373
MIN_SIGMA_FLOOR_MEDIAN_FLUX_FRAC = 0.02
# Cap per-point weight to prevent near-zero flux points from dominating fits.
MAX_WEIGHT_MEDIAN_RATIO = 25.0
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_TEMPLATES_PATH = DATA_DIR / "spherex_templates.joblib"


def _to_flambda_from_fnu_jy(wv_um: np.ndarray, flux_jy: np.ndarray) -> np.ndarray:
    wv_m = wv_um * 1e-6
    return flux_jy * 1e-32 * SPEED_OF_LIGHT / (wv_m * wv_m) * 1e-10


def _to_comparison_df_from_spiff_units(
    wv_um: np.ndarray,
    flux_ujy: np.ndarray,
    flux_ujy_err: np.ndarray,
) -> pd.DataFrame:
    flux_jy = flux_ujy * 1e-6
    flux_jy_err = flux_ujy_err * 1e-6
    wv_ang = wv_um * 1e4
    flux_flambda = _to_flambda_from_fnu_jy(wv_um, flux_jy)
    flux_flambda_err = _to_flambda_from_fnu_jy(wv_um, flux_jy_err)
    return pd.DataFrame(
        {
            "wavelength_um": wv_um,
            "flux_ujy": flux_ujy,
            "flux_err_ujy": flux_ujy_err,
            "wavelength_angstrom": wv_ang,
            "flux_flambda": flux_flambda,
            "flux_flambda_unc": flux_flambda_err,
        }
    )


def _read_csv_spectrum(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = set(df.columns)
    if {"psf_un_wv_um", "psf_un_flux_uJy", "psf_un_flux_uJy_err"}.issubset(cols):
        wv_um = df["psf_un_wv_um"].astype(float).to_numpy()
        flux_ujy = df["psf_un_flux_uJy"].astype(float).to_numpy()
        flux_ujy_err = df["psf_un_flux_uJy_err"].astype(float).to_numpy()
        out = _to_comparison_df_from_spiff_units(wv_um, flux_ujy, flux_ujy_err)
        if "psf_un_snr" in df.columns:
            out["psf_un_snr"] = pd.to_numeric(df["psf_un_snr"], errors="coerce").to_numpy(dtype=float)
        if "n_pix_used_in_fit" in df.columns:
            out["n_pix_used_in_fit"] = pd.to_numeric(df["n_pix_used_in_fit"], errors="coerce").to_numpy(dtype=float)
        return out
    elif {"wavelength_um", "flux_ujy", "flux_err_ujy"}.issubset(cols):
        wv_um = df["wavelength_um"].astype(float).to_numpy()
        flux_ujy = df["flux_ujy"].astype(float).to_numpy()
        flux_ujy_err = df["flux_err_ujy"].astype(float).to_numpy()
        out = _to_comparison_df_from_spiff_units(wv_um, flux_ujy, flux_ujy_err)
        if "snr" in df.columns:
            out["psf_un_snr"] = pd.to_numeric(df["snr"], errors="coerce").to_numpy(dtype=float)
        elif "flux_snr" in df.columns:
            out["psf_un_snr"] = pd.to_numeric(df["flux_snr"], errors="coerce").to_numpy(dtype=float)
        if "n_pix_used_in_fit" in df.columns:
            out["n_pix_used_in_fit"] = pd.to_numeric(df["n_pix_used_in_fit"], errors="coerce").to_numpy(dtype=float)
        return out
    elif {"wavelength_angstrom", "flux_flambda", "flux_flambda_unc"}.issubset(cols):
        out = df.copy()
        for col in ("wavelength_angstrom", "flux_flambda", "flux_flambda_unc"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if "ignored" in out.columns:
            out["ignored"] = pd.to_numeric(out["ignored"], errors="coerce").fillna(0).astype(int)
        return out
    else:
        raise ValueError(
            "CSV missing required columns. Supported schemas are "
            "['psf_un_wv_um', 'psf_un_flux_uJy', 'psf_un_flux_uJy_err'] or "
            "['wavelength_um', 'flux_ujy', 'flux_err_ujy'] or "
            "['wavelength_angstrom', 'flux_flambda', 'flux_flambda_unc']."
        )

def _load_spherex_bins() -> pd.DataFrame:
    return load_spherex_bins()

def _bin_spiff_like_spectrum(comp_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    try:
        binned = build_binned_spherex_spectrum(comp_df)
    except ValueError as exc:
        raise ValueError(
            "--bin requires SPIFF-like wavelength/flux/error columns in um/uJy/uJy_err. "
            "Supported for --csv and --raw inputs."
        ) from exc

    if len(binned) < MIN_MATCH_POINTS:
        raise ValueError("Not enough occupied SPHEREx bins remain for --bin.")

    out = binned[["wavelength_angstrom", "flux_flambda", "flux_flambda_unc", "ignored"]].copy()
    return out, int(len(out))


def _load_templates() -> dict[int, dict[str, object]]:
    templates = joblib.load(DEFAULT_TEMPLATES_PATH)
    if not isinstance(templates, dict) or not templates:
        raise ValueError("Local SPHEREx template bundle is empty or invalid.")
    return templates


def _clean_comparison(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["wavelength_angstrom", "flux_flambda", "flux_flambda_unc"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "ignored" in out.columns:
        out["ignored"] = pd.to_numeric(out["ignored"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["wavelength_angstrom", "flux_flambda", "flux_flambda_unc"])
    out = out[np.isfinite(out["wavelength_angstrom"])]
    out = out[np.isfinite(out["flux_flambda"]) & np.isfinite(out["flux_flambda_unc"])]
    out = out[(out["flux_flambda"] > 0) & (out["flux_flambda_unc"] > 0)]
    out = out.sort_values("wavelength_angstrom")
    return out.reset_index(drop=True)


def _nearest_model_flux(
    model_w: np.ndarray, model_f: np.ndarray, data_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(model_w, data_w)
    idx = np.clip(idx, 1, len(model_w) - 1)

    left = idx - 1
    right = idx
    dist_left = np.abs(model_w[left] - data_w)
    dist_right = np.abs(model_w[right] - data_w)

    use_right = dist_right < dist_left
    best_idx = np.where(use_right, right, left)
    best_dist = np.where(use_right, dist_right, dist_left)
    best_flux = model_f[best_idx]
    return best_flux, best_dist


def _rolling_median_centered(values: np.ndarray, window: int) -> np.ndarray:
    w = int(window)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    return (
        pd.Series(values)
        .rolling(window=w, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _flag_high_uncertainty_points(
    comp_df: pd.DataFrame,
    *,
    window_points: int = 9,
    err_factor: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Method 1: flag points with unusually high uncertainty vs local median."""
    data_e = comp_df["flux_flambda_unc"].to_numpy(dtype=float)
    expected = _rolling_median_centered(data_e, window_points)
    floor = np.nanmedian(expected[np.isfinite(expected) & (expected > 0)])
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-30
    expected = np.where(np.isfinite(expected) & (expected > 0), expected, floor)
    flags = data_e > (float(err_factor) * expected)
    return flags.astype(bool), expected


def _flag_residual_outliers_bestfit(
    comp_df: pd.DataFrame,
    best_row: pd.Series,
    template: dict[str, object],
    *,
    min_sigma_cut: float = 5.0,
    robust_z_cut: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Method 2: flag outliers in |residual|/error distribution for best-fit template."""
    data_w = comp_df["wavelength_angstrom"].to_numpy(dtype=float)
    data_f = comp_df["flux_flambda"].to_numpy(dtype=float)
    data_e = comp_df["flux_flambda_unc"].to_numpy(dtype=float)
    model_w = template["wavelength_angstrom"].astype(float)
    model_f = template["flux_flambda"].astype(float)

    mflux, dist = _nearest_model_flux(model_w, model_f, data_w)
    ok = dist <= MAX_TEMPLATE_DIST_ANGSTROM
    flags = np.zeros(len(data_w), dtype=bool)
    if not np.any(ok):
        return flags, float(min_sigma_cut)

    scale_s = best_row.get("scale_s", None)
    if scale_s is None or not np.isfinite(scale_s):
        d = data_f[ok]
        e = data_e[ok]
        m = mflux[ok]
        den = np.sum(m * m / (e * e))
        if not np.isfinite(den) or den <= 0:
            return flags, float(min_sigma_cut)
        num = np.sum(d * m / (e * e))
        scale_s = num / den

    abs_norm_resid = np.abs((data_f[ok] - float(scale_s) * mflux[ok]) / data_e[ok])
    med = float(np.nanmedian(abs_norm_resid))
    mad = float(np.nanmedian(np.abs(abs_norm_resid - med)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        sigma_cut = float(min_sigma_cut)
    else:
        sigma_cut = max(float(min_sigma_cut), med + float(robust_z_cut) * robust_sigma)
    flags_ok = abs_norm_resid > sigma_cut
    flags[np.where(ok)[0][flags_ok]] = True
    return flags, float(sigma_cut)


def _compute_chi2(
    data_w: np.ndarray,
    data_f: np.ndarray,
    data_e: np.ndarray,
    model_w: np.ndarray,
    model_f: np.ndarray,
    drop_worst_n: int,
    chi2_sigma_cap: float,
) -> dict[str, Optional[float]]:
    if len(model_w) < 2:
        return {
            "n_used": 0,
            "scale_s": None,
            "reduced_chi2": None,
            "chi2_raw": None,
            "chi2_raw_10pct": None,
            "reduced_chi2_10pct": None,
            "red_chi2_raw_10pct": None,
            "red_chi2_10pct": None,
            "n_red_used": None,
            "chi2_raw_10pct_cap": None,
            "reduced_chi2_10pct_cap": None,
        }

    mflux, dist = _nearest_model_flux(model_w, model_f, data_w)
    ok = dist <= MAX_TEMPLATE_DIST_ANGSTROM
    if not np.any(ok):
        return {
            "n_used": 0,
            "scale_s": None,
            "reduced_chi2": None,
            "chi2_raw": None,
            "chi2_raw_10pct": None,
            "reduced_chi2_10pct": None,
            "red_chi2_raw_10pct": None,
            "red_chi2_10pct": None,
            "n_red_used": None,
            "chi2_raw_10pct_cap": None,
            "reduced_chi2_10pct_cap": None,
        }

    w = data_w[ok]
    d = data_f[ok]
    e = data_e[ok]
    m = mflux[ok]
    med_flux = float(np.nanmedian(d))
    if not np.isfinite(med_flux) or med_flux <= 0:
        med_flux = float(np.nanmedian(data_f[np.isfinite(data_f) & (data_f > 0)]))
    if not np.isfinite(med_flux) or med_flux <= 0:
        med_flux = 0.0
    # Prevent near-zero flux points from getting unrealistically tiny sigma floors.
    d_for_floor = np.maximum(d, MIN_SIGMA_FLOOR_MEDIAN_FLUX_FRAC * med_flux)

    # Use the same 10%-of-flux uncertainty floor for scale fitting as for scoring.
    sigma_eff = np.maximum(e, 0.1 * d_for_floor)
    weights = 1.0 / (sigma_eff * sigma_eff)
    med_w = float(np.nanmedian(weights[np.isfinite(weights) & (weights > 0)]))
    if np.isfinite(med_w) and med_w > 0:
        weights = np.minimum(weights, MAX_WEIGHT_MEDIAN_RATIO * med_w)
    num = np.sum(d * m * weights)
    den = np.sum(m * m * weights)
    n_used = len(d)

    if n_used < MIN_MATCH_POINTS or not np.isfinite(den) or den <= 0:
        return {
            "n_used": n_used,
            "scale_s": None,
            "reduced_chi2": None,
            "chi2_raw": None,
            "chi2_raw_10pct": None,
            "reduced_chi2_10pct": None,
            "red_chi2_raw_10pct": None,
            "red_chi2_10pct": None,
            "n_red_used": None,
            "chi2_raw_10pct_cap": None,
            "reduced_chi2_10pct_cap": None,
        }

    s = num / den
    resid = (d - s * m)
    contrib = (resid * resid) * weights
    effective_drop_worst_n = min(int(max(0, drop_worst_n)), max(0, n_used - MIN_MATCH_POINTS))
    if effective_drop_worst_n > 0:
        # Keep at least MIN_MATCH_POINTS so small bundled example spectra still score.
        keep_idx = np.argsort(contrib)[: n_used - effective_drop_worst_n]
        w = w[keep_idx]
        d = d[keep_idx]
        e = e[keep_idx]
        m = m[keep_idx]
        n_used = len(d)
        med_flux = float(np.nanmedian(d))
        if not np.isfinite(med_flux) or med_flux <= 0:
            med_flux = float(np.nanmedian(data_f[np.isfinite(data_f) & (data_f > 0)]))
        if not np.isfinite(med_flux) or med_flux <= 0:
            med_flux = 0.0
        d_for_floor = np.maximum(d, MIN_SIGMA_FLOOR_MEDIAN_FLUX_FRAC * med_flux)
        sigma_eff = np.maximum(e, 0.1 * d_for_floor)
        weights = 1.0 / (sigma_eff * sigma_eff)
        med_w = float(np.nanmedian(weights[np.isfinite(weights) & (weights > 0)]))
        if np.isfinite(med_w) and med_w > 0:
            weights = np.minimum(weights, MAX_WEIGHT_MEDIAN_RATIO * med_w)
        num = np.sum(d * m * weights)
        den = np.sum(m * m * weights)
        if not np.isfinite(den) or den <= 0 or n_used < MIN_MATCH_POINTS:
            return {
                "n_used": n_used,
                "scale_s": None,
                "reduced_chi2": None,
                "chi2_raw": None,
                "chi2_raw_10pct": None,
                "reduced_chi2_10pct": None,
                "red_chi2_raw_10pct": None,
                "red_chi2_10pct": None,
                "n_red_used": None,
                "chi2_raw_10pct_cap": None,
                "reduced_chi2_10pct_cap": None,
            }
        s = num / den
        resid = (d - s * m)
        contrib = (resid * resid) * weights

    chi2_raw = float(np.sum(contrib))
    reduced = chi2_raw / n_used
    sigma_eff = np.maximum(e, 0.1 * d_for_floor)
    weights = 1.0 / (sigma_eff * sigma_eff)
    med_w = float(np.nanmedian(weights[np.isfinite(weights) & (weights > 0)]))
    if np.isfinite(med_w) and med_w > 0:
        weights = np.minimum(weights, MAX_WEIGHT_MEDIAN_RATIO * med_w)
    contrib_10pct = (resid * resid) * weights
    chi2_raw_10pct = float(np.sum(contrib_10pct))
    reduced_10pct = chi2_raw_10pct / n_used
    red_mask = w >= 14000.0
    n_red_used = int(np.sum(red_mask))
    if n_red_used > 0:
        red_chi2_raw_10pct = float(np.sum(contrib_10pct[red_mask]))
        red_chi2_10pct = red_chi2_raw_10pct / n_red_used
    else:
        red_chi2_raw_10pct = None
        red_chi2_10pct = None
    cap2 = float(chi2_sigma_cap) * float(chi2_sigma_cap)
    contrib_10pct_cap = np.minimum(contrib_10pct, cap2)
    chi2_raw_10pct_cap = float(np.sum(contrib_10pct_cap))
    reduced_10pct_cap = chi2_raw_10pct_cap / n_used
    return {
        "n_used": n_used,
        "scale_s": float(s),
        "reduced_chi2": float(reduced),
        "chi2_raw": float(chi2_raw),
        "chi2_raw_10pct": float(chi2_raw_10pct),
        "reduced_chi2_10pct": float(reduced_10pct),
        "red_chi2_raw_10pct": red_chi2_raw_10pct,
        "red_chi2_10pct": red_chi2_10pct,
        "n_red_used": n_red_used,
        "chi2_raw_10pct_cap": float(chi2_raw_10pct_cap),
        "reduced_chi2_10pct_cap": float(reduced_10pct_cap),
    }


def _autotype(
    comp_df: pd.DataFrame,
    templates: dict[int, dict[str, object]],
    drop_worst_n: int,
    chi2_sigma_cap: float,
    nonfield_odds_k: float,
    nonfield_extreme_odds_k: float,
) -> pd.DataFrame:
    data_w = comp_df["wavelength_angstrom"].to_numpy(dtype=float)
    data_f = comp_df["flux_flambda"].to_numpy(dtype=float)
    data_e = comp_df["flux_flambda_unc"].to_numpy(dtype=float)

    rows = []
    penalty_nonfield = 2.0 * math.log(float(nonfield_odds_k))
    penalty_nonfield_extreme = 2.0 * math.log(float(nonfield_extreme_odds_k))
    for tid, t in templates.items():
        model_w = t["wavelength_angstrom"]
        model_f = t["flux_flambda"]
        result = _compute_chi2(
            data_w, data_f, data_e, model_w, model_f, drop_worst_n, chi2_sigma_cap
        )
        grid_type = str(t.get("grid_type") or "")
        grid_norm = grid_type.strip().lower()
        if grid_norm == "field":
            penalty = 0.0
        elif grid_norm in {"extreme subdwarfs", "extremely low gravity"}:
            penalty = penalty_nonfield_extreme
        else:
            penalty = penalty_nonfield
        score = None
        if result.get("chi2_raw_10pct_cap") is not None:
            score = float(result["chi2_raw_10pct_cap"]) + float(penalty)
        rows.append(
            {
                "template_id": tid,
                "spectral_type": t["spectral_type"],
                "spectral_type_number": t["spectral_type_number"],
                "grid_type": t.get("grid_type"),
                "selection_penalty": float(penalty),
                "selection_score": score,
                **result,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["selection_score", "spectral_type_number"],
        ascending=[True, True],
        na_position="last",
    )
    return df.reset_index(drop=True)


def _plot_comparison(
    comp_df: pd.DataFrame,
    templates: dict[int, dict[str, object]],
    top_df: pd.DataFrame,
    outpath: Optional[str],
    title_suffix: Optional[str],
    yrange_from_data: bool,
    full_yrange: bool,
    show_data_spline: bool,
    show_plot: bool,
    save_plot: bool,
    flagged_mask: Optional[np.ndarray] = None,
    overplot_binned_df: Optional[pd.DataFrame] = None,
) -> None:
    if top_df.empty:
        print("No valid templates with finite chi2 to plot.")
        return

    data_w = comp_df["wavelength_angstrom"].to_numpy(dtype=float)
    data_f = comp_df["flux_flambda"].to_numpy(dtype=float)
    data_e = comp_df["flux_flambda_unc"].to_numpy(dtype=float)
    data_w_um = data_w / 1e4
    binned_w_um: Optional[np.ndarray] = None
    binned_f: Optional[np.ndarray] = None
    binned_e: Optional[np.ndarray] = None
    if overplot_binned_df is not None and not overplot_binned_df.empty:
        binned_w = pd.to_numeric(overplot_binned_df["wavelength_angstrom"], errors="coerce").to_numpy(dtype=float)
        binned_f = pd.to_numeric(overplot_binned_df["flux_flambda"], errors="coerce").to_numpy(dtype=float)
        binned_e = pd.to_numeric(overplot_binned_df["flux_flambda_unc"], errors="coerce").to_numpy(dtype=float)
        binned_ok = np.isfinite(binned_w) & np.isfinite(binned_f) & np.isfinite(binned_e) & (binned_e > 0)
        if np.any(binned_ok):
            binned_w_um = binned_w[binned_ok] / 1e4
            binned_f = binned_f[binned_ok]
            binned_e = binned_e[binned_ok]
        else:
            binned_w_um = None
            binned_f = None
            binned_e = None
    if flagged_mask is None or len(flagged_mask) != len(data_w):
        flagged_mask = np.zeros(len(data_w), dtype=bool)
    flagged_mask = flagged_mask.astype(bool)
    good_mask = ~flagged_mask
    data_f_for_range = data_f[good_mask] if np.any(good_mask) else data_f

    fig, ax = plt.subplots(figsize=(10, 10))

    data_span = np.nanmax(data_f_for_range) - np.nanmin(data_f_for_range)
    if not np.isfinite(data_span) or data_span <= 0:
        data_span = np.nanmax(data_f)
    if not np.isfinite(data_span) or data_span <= 0:
        data_span = 1.0

    if yrange_from_data:
        offset = 1.15 * data_span
        base_span = data_span
    else:
        # Base spacing on template span to avoid outliers in comparison data.
        base_span = 0.0
        for row in top_df.itertuples(index=False):
            t = templates[int(row.template_id)]
            model_f = t["flux_flambda"].astype(float)
            scale_s = row.scale_s if row.scale_s is not None else 1.0
            model_f = model_f * float(scale_s)
            span = np.nanmax(model_f) - np.nanmin(model_f)
            if np.isfinite(span):
                base_span = max(base_span, span)
        if not np.isfinite(base_span) or base_span <= 0:
            base_span = data_span
        offset = 1.15 * base_span

    free_space = max(0.0, offset - base_span)

    y_min_global = np.inf
    y_max_global = -np.inf

    for i, row in enumerate(top_df.itertuples(index=False)):
        tid = int(row.template_id)
        t = templates[tid]
        model_w = t["wavelength_angstrom"].astype(float)
        model_f = t["flux_flambda"].astype(float)
        scale_s = row.scale_s if row.scale_s is not None else 1.0
        model_f = model_f * float(scale_s)
        model_w_um = model_w / 1e4

        shift = (len(top_df) - 1 - i) * offset

        # Show the zero-flux baseline for this offset row.
        ax.axhline(shift, color="#b9d9f7", linewidth=0.9, alpha=0.9, zorder=0)

        # comparison spectrum (points + error bars)
        if np.any(good_mask):
            ax.errorbar(
                data_w_um[good_mask],
                (data_f + shift)[good_mask],
                yerr=data_e[good_mask],
                fmt="o",
                ms=4.2,
                mfc="white",
                mec="0.4",
                mew=2,
                ecolor="0.6",
                elinewidth=2,
                capsize=0,
                zorder=2,
            )
        if np.any(flagged_mask):
            ax.errorbar(
                data_w_um[flagged_mask],
                (data_f + shift)[flagged_mask],
                yerr=data_e[flagged_mask],
                fmt="x",
                ms=5.0,
                mfc="none",
                mec="tab:orange",
                mew=1.5,
                ecolor="tab:orange",
                elinewidth=1.2,
                capsize=0,
                zorder=0,
                alpha=0.2,
                label="Flagged bad data points" if i == 0 else None,
            )
        if binned_w_um is not None and binned_f is not None and binned_e is not None:
            ax.errorbar(
                binned_w_um,
                binned_f + shift,
                yerr=binned_e,
                fmt="o",
                ms=4.4,
                mfc="#1f77b4",
                mec="#1f77b4",
                mew=0.8,
                ecolor="#1f77b4",
                elinewidth=1.3,
                capsize=0,
                alpha=0.95,
                zorder=2.6,
                label="Binned comparison spectrum" if i == 0 else None,
            )
            ax.plot(
                binned_w_um,
                binned_f + shift,
                color="#1f77b4",
                linewidth=1.1,
                alpha=0.9,
                zorder=2.5,
            )

        # black spline through comparison (optional)
        if show_data_spline:
            try:
                spline_c = UnivariateSpline(data_w_um, data_f + shift, s=0)
                w_dense = np.linspace(data_w_um.min(), data_w_um.max(), 1000)
                ax.plot(w_dense, spline_c(w_dense), color="black", linewidth=2.5, zorder=1)
            except Exception:
                pass

        # template spline
        try:
            spline_t = UnivariateSpline(model_w_um, model_f + shift, s=0)
            w_dense_t = np.linspace(model_w_um.min(), model_w_um.max(), 1000)
            ax.plot(w_dense_t, spline_t(w_dense_t), color="red", linewidth=1.5, zorder=3)
        except Exception:
            ax.plot(model_w_um, model_f + shift, color="red", linewidth=1.5, zorder=3)

        # optional template points (open red circles)
        ax.plot(
            model_w_um,
            model_f + shift,
            "o",
            ms=3.8,
            mfc="red",
            mec="red",
            mew=0.5,
            alpha=0.7,
            zorder=3,
        )

        spt = _format_spt_with_pec_suffix(
            row.spectral_type,
            row.spectral_type_number,
            row.red_chi2_10pct,
        )
        chi2 = row.reduced_chi2
        label = f"{spt}  $\\chi^2_{{\\rm r}}$={chi2:.1f}"
        x_text = data_w_um.max() - 0.005 * (data_w_um.max() - data_w_um.min())
        min_model = np.nanmin(model_f)
        y_text = min_model + shift - 0.5 * free_space
        ax.text(
            x_text,
            y_text,
            label,
            color="red",
            fontsize=14,
            fontweight="bold",
            ha="right",
            va="center",
        )

        y_min_global = min(y_min_global, np.nanmin(model_f + shift))
        y_max_global = max(y_max_global, np.nanmax(model_f + shift))

    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"Relative Spectral Flux Density + offset ($F_\lambda$)")
    title = "SPHEREx Autotype: Comparison Spectrum vs Templates"
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    if np.any(flagged_mask) or binned_w_um is not None:
        ax.legend(loc="upper right", fontsize=10)

    xpad = 0.03 * (data_w_um.max() - data_w_um.min())
    ax.set_xlim(data_w_um.min() - xpad, data_w_um.max() + xpad)
    if full_yrange:
        y_min = np.nanmin(data_f_for_range)
        y_max = np.nanmax(data_f_for_range) + (len(top_df) - 1) * offset
        y_pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    elif yrange_from_data:
        y_min = np.nanmin(data_f_for_range)
        y_max = np.nanmax(data_f_for_range) + (len(top_df) - 1) * offset
        y_pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    else:
        if not np.isfinite(y_min_global) or not np.isfinite(y_max_global):
            y_min_global = np.nanmin(data_f)
            y_max_global = np.nanmax(data_f) + (len(top_df) - 1) * offset
        y_pad = 0.05 * (y_max_global - y_min_global)
        ax.set_ylim(y_min_global - y_pad, y_max_global + y_pad)
    ax.tick_params(labelsize=13)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    saved_outpath: Optional[str] = None
    if save_plot:
        if outpath is None:
            outpath = "spiff_autotype.png"
        outpath = os.path.abspath(outpath)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=200)
        saved_outpath = outpath
        print(f"Saved plot to {outpath}")

    if show_plot:
        backend = str(matplotlib.get_backend()).lower()
        if "agg" in backend:
            if saved_outpath is None:
                print("Plot display skipped: Agg backend requires a saved plot file to open.")
            else:
                opener: Optional[list[str]] = None
                if sys.platform == "darwin":
                    opener = ["open", saved_outpath]
                elif sys.platform.startswith("linux"):
                    opener = ["xdg-open", saved_outpath]
                elif os.name == "nt":
                    os.startfile(saved_outpath)
                if opener is not None:
                    if shutil.which(opener[0]) is None:
                        print(f"Plot display skipped: opener not found: {opener[0]}")
                    else:
                        try:
                            subprocess.Popen(
                                opener,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            print(f"Plot display skipped: failed to open saved plot: {e}")
        else:
            plt.show()


def _print_table(df: pd.DataFrame) -> None:
    cols = [
        "spectral_type",
        "spectral_type_number",
        "grid_type",
        "n_used",
        "scale_s",
        "reduced_chi2",
        "red_chi2_10pct",
        "reduced_chi2_10pct_cap",
        "selection_penalty",
        "selection_score",
        "chi2_raw",
    ]
    print(df[cols].to_string(index=False))


def _norm_spectral_type_label(value: object) -> str:
    return "".join(str(value).split()).upper()


def _format_spt_with_pec_suffix(
    spectral_type: object,
    spectral_type_number: object,
    red_chi2_10pct: object,
) -> str:
    spt_display = str(spectral_type)
    try:
        pec_ratio = _compute_pec_ratio(spectral_type_number, red_chi2_10pct)
        if pec_ratio is not None:
            spt_lower = spt_display.lower()
            has_explicit_suffix_or_gravity = (
                ("pec" in spt_lower)
                or ("gamma" in spt_lower)
                or ("beta" in spt_lower)
                or ("alpha" in spt_lower)
            )
            if pec_ratio >= 2.0 and (not has_explicit_suffix_or_gravity):
                spt_display = f"{spt_display} pec"
    except Exception:
        pass
    return spt_display


def _compute_pec_ratio(
    spectral_type_number: object,
    red_chi2_10pct: object,
) -> Optional[float]:
    sptn = pd.to_numeric(spectral_type_number, errors="coerce")
    red_val = pd.to_numeric(red_chi2_10pct, errors="coerce")
    if not (np.isfinite(sptn) and np.isfinite(red_val)):
        return None
    denom = 10.0 ** (PEC_A + PEC_B * float(sptn))
    if not (np.isfinite(denom) and denom > 0):
        return None
    return float(red_val) / float(denom)


def _parse_overlay_types(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    out = []
    for part in str(value).split(","):
        norm = _norm_spectral_type_label(part)
        if norm:
            out.append(norm)
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Autotype a local spectrum against bundled SPHEREx templates.")

    ap.add_argument("--csv", type=str, required=True, help="Local CSV to use as the comparison spectrum.")
    ap.add_argument(
        "--bin",
        action="store_true",
        help=(
            "Bin higher-resolution SPIFF-style input onto the bundled local SPHEREx bin table "
            "using inverse-variance weighted means before template fitting."
        ),
    )
    ap.add_argument(
        "--overplotbin",
        action="store_true",
        help=(
            "Overplot a SPIFF-style nearest-bin weighted spectrum in blue on the "
            "comparison plot while still fitting the unbinned input."
        ),
    )

    ap.add_argument("--plot-file", type=str, default=None, help="Output plot file name")
    ap.add_argument("--no-plot", action="store_true", help="Skip interactive plot display")
    ap.add_argument("--no-save-plot", action="store_true", help="Do not save plot to disk")
    ap.add_argument("--top-n", type=int, default=3, help="Number of best templates to plot")
    ap.add_argument(
        "--drop-worst-n",
        type=int,
        default=5,
        help="Drop worst-fitting N points (by chi2 contribution) per template.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output plot file if it already exists.",
    )
    ap.add_argument(
        "--yrange-from-data",
        action="store_true",
        help="Scale y-range using comparison data instead of templates.",
    )
    ap.add_argument(
        "--full-yrange",
        action="store_true",
        help="Force y-range to include all comparison data points.",
    )
    ap.add_argument(
        "--show-data-spline",
        action="store_true",
        help="Show black spline through comparison data points.",
    )
    ap.add_argument(
        "--overlay-spectral-type",
        type=str,
        default=None,
        help="Force-overlay one template of this spectral type label (e.g., T8) on the plot. Deprecated; use --overlay-spectral-types.",
    )
    ap.add_argument(
        "--overlay-spectral-types",
        type=str,
        default=None,
        help="Comma-separated spectral types to force-overlay (e.g., T8,L0).",
    )
    ap.add_argument(
        "--baderr-window",
        type=int,
        default=9,
        help="Window size (points) for local-median error envelope flagging.",
    )
    ap.add_argument(
        "--baderr-factor",
        type=float,
        default=5.0,
        help="Flag points with error > factor * local median error.",
    )
    ap.add_argument(
        "--badresid-min-sigma",
        type=float,
        default=5.0,
        help="Minimum |residual|/error threshold for best-fit residual outlier flagging.",
    )
    ap.add_argument(
        "--badresid-robust-z",
        type=float,
        default=5.0,
        help="Robust-z multiplier around median(|residual|/error) for residual outlier flagging.",
    )
    ap.add_argument(
        "--chi2-sigma-cap",
        type=float,
        default=5.0,
        help="Per-point sigma cap used for capped 10%%-floor chi2 contribution (default: 5.0).",
    )
    ap.add_argument(
        "--nonfield-odds-k",
        type=float,
        default=500.0,
        help="Odds-factor K penalty for non-field grids; selection penalty is 2*ln(K) (default: 500).",
    )
    ap.add_argument(
        "--nonfield-extreme-odds-k",
        type=float,
        default=500000.0,
        help="Odds-factor K2 penalty for extreme subdwarfs/extremely low gravity grids; selection penalty is 2*ln(K2).",
    )

    return ap.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    overplot_binned_df: Optional[pd.DataFrame] = None

    if args.top_n <= 0:
        print("Error: --top-n must be >= 1", file=sys.stderr)
        return 2
    if args.chi2_sigma_cap <= 0:
        print("Error: --chi2-sigma-cap must be > 0", file=sys.stderr)
        return 2
    if args.nonfield_odds_k <= 0:
        print("Error: --nonfield-odds-k must be > 0", file=sys.stderr)
        return 2
    if args.nonfield_extreme_odds_k <= 0:
        print("Error: --nonfield-extreme-odds-k must be > 0", file=sys.stderr)
        return 2

    comp_df = _read_csv_spectrum(args.csv)
    source_label = f"CSV: {args.csv}"
    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    csv_dir_name = os.path.basename(csv_dir) or "comparison"
    csv_file_name = os.path.basename(os.path.abspath(args.csv))
    default_plot = os.path.join(csv_dir, f"spiff_autotype_{csv_dir_name}.png")
    title_suffix = f"{csv_dir_name} | {csv_file_name}"

    if args.overplotbin:
        overplot_binned_df, n_bins_overplot = _bin_spiff_like_spectrum(comp_df)
        source_label = f"{source_label} | overplot_binned_to_spherex_bins={n_bins_overplot}"
        if title_suffix is not None:
            title_suffix = f"{title_suffix} | overplotbin"

    if args.bin:
        comp_df, n_bins = _bin_spiff_like_spectrum(comp_df)
        source_label = f"{source_label} | binned_to_spherex_bins={n_bins}"
        if title_suffix is not None:
            title_suffix = f"{title_suffix} | binned"

    comp_df = _clean_comparison(comp_df)
    if len(comp_df) < MIN_MATCH_POINTS:
        print("Error: not enough valid comparison points after cleaning.", file=sys.stderr)
        return 2

    input_ignored_mask = np.zeros(len(comp_df), dtype=bool)
    if "ignored" in comp_df.columns:
        input_ignored_mask = comp_df["ignored"].to_numpy(dtype=int) == 1
        if np.any(input_ignored_mask):
            print(f"Loaded {int(np.sum(input_ignored_mask))} spectrum points with ignored=1; they will be marked as bad in plots.")

    fit_indices = np.where(~input_ignored_mask)[0]
    fit_df = comp_df.iloc[fit_indices].reset_index(drop=True)
    if len(fit_df) < MIN_MATCH_POINTS:
        print(
            "Warning: too few non-ignored points for fitting; falling back to all points for fit.",
            file=sys.stderr,
        )
        fit_df = comp_df.copy().reset_index(drop=True)
        input_ignored_mask = np.zeros(len(comp_df), dtype=bool)
        fit_indices = np.arange(len(comp_df), dtype=int)

    templates = _load_templates()
    # First pass fit (excluding ignored input points only)
    results_pre = _autotype(
        fit_df,
        templates,
        args.drop_worst_n,
        args.chi2_sigma_cap,
        args.nonfield_odds_k,
        args.nonfield_extreme_odds_k,
    )

    bad_err_flags_fit, _ = _flag_high_uncertainty_points(
        fit_df,
        window_points=args.baderr_window,
        err_factor=args.baderr_factor,
    )
    bad_resid_flags_fit = np.zeros(len(fit_df), dtype=bool)
    resid_sigma_cut = float(args.badresid_min_sigma)
    finite_df = results_pre[results_pre["selection_score"].notna()]
    if not finite_df.empty:
        best_row = finite_df.iloc[0]
        best_tid = int(best_row["template_id"])
        best_spt_for_flagging = _format_spt_with_pec_suffix(
            best_row.get("spectral_type"),
            best_row.get("spectral_type_number"),
            best_row.get("red_chi2_10pct"),
        )
        if "pec" in str(best_spt_for_flagging).lower():
            # For pec classifications, avoid template-residual-based bad-pixel masking.
            print(
                "Method2 template-residual bad-pixel flagging skipped for pec classification "
                f"({best_spt_for_flagging})."
            )
        else:
            bad_resid_flags_fit, resid_sigma_cut = _flag_residual_outliers_bestfit(
                fit_df,
                best_row,
                templates[best_tid],
                min_sigma_cut=args.badresid_min_sigma,
                robust_z_cut=args.badresid_robust_z,
            )

    # Second pass fit excludes all bad points from best-chi2 determination.
    fit_bad_mask = bad_err_flags_fit | bad_resid_flags_fit
    fit_df_final = fit_df[~fit_bad_mask].reset_index(drop=True)
    if len(fit_df_final) >= MIN_MATCH_POINTS:
        results = _autotype(
            fit_df_final,
            templates,
            args.drop_worst_n,
            args.chi2_sigma_cap,
            args.nonfield_odds_k,
            args.nonfield_extreme_odds_k,
        )
    else:
        print(
            "Warning: too few points after bad-pixel rejection; using first-pass fit results.",
            file=sys.stderr,
        )
        results = results_pre

    print(f"Comparison spectrum: {source_label}")
    best_row: Optional[pd.Series] = None
    if results["selection_score"].notna().any():
        best_row = results[results["selection_score"].notna()].iloc[0]
        best_spt_display = _format_spt_with_pec_suffix(
            best_row["spectral_type"],
            best_row.get("spectral_type_number"),
            best_row.get("red_chi2_10pct"),
        )
        print(
            f"Best match: {best_spt_display}  chi2={best_row['reduced_chi2']:.4f}  "
            f"score={best_row['selection_score']:.4f}  "
            f"n_used={int(best_row['n_used'])}"
        )
    else:
        print("No finite chi2 values; check comparison spectrum or templates.")

    _print_table(results)

    bad_err_flags = np.zeros(len(comp_df), dtype=bool)
    bad_resid_flags = np.zeros(len(comp_df), dtype=bool)
    bad_err_flags[fit_indices] = bad_err_flags_fit
    bad_resid_flags[fit_indices] = bad_resid_flags_fit
    flagged_mask = input_ignored_mask | bad_err_flags | bad_resid_flags
    print(
        "Flagged points: "
        f"input_ignored={int(np.sum(input_ignored_mask))}, "
        f"method1_bad_error={int(np.sum(bad_err_flags))}, "
        f"method2_bad_resid={int(np.sum(bad_resid_flags))} "
        f"(sigma_cut={resid_sigma_cut:.2f}), "
        f"combined={int(np.sum(flagged_mask))}/{len(comp_df)}"
    )

    top_df = results[results["selection_score"].notna()].head(args.top_n)
    plot_df = top_df
    overlay_targets = _parse_overlay_types(args.overlay_spectral_types)
    if args.overlay_spectral_type:
        # Back-compat: allow old single-value flag; also accept comma-separated values.
        for old_target in _parse_overlay_types(args.overlay_spectral_type):
            if old_target not in overlay_targets:
                overlay_targets.append(old_target)
    if overlay_targets:
        labels = results["spectral_type"].astype(str).map(_norm_spectral_type_label)
        top_tids = set(top_df["template_id"].astype(int).tolist())
        forced_rows = []
        for target in overlay_targets:
            matches = results[labels == target]
            if matches.empty:
                print(f"Requested overlay spectral type not found in templates: {target}")
                continue
            forced = matches.iloc[[0]]
            forced_tid = int(forced.iloc[0]["template_id"])
            if forced_tid in top_tids:
                print(f"Overlay spectral type already included in top-{args.top_n}: {target}")
                continue
            forced_rows.append(forced)
            top_tids.add(forced_tid)
            print(f"Overlaying spectral type {target} (template_id={forced_tid})")
        if forced_rows:
            plot_df = pd.concat([top_df] + forced_rows, ignore_index=True)
    show_plot = not args.no_plot
    save_plot = not args.no_save_plot

    if save_plot and (not args.overwrite) and os.path.exists(args.plot_file or default_plot):
        print(f"Output plot exists, skipping: {args.plot_file or default_plot}")
        save_plot = False

    if show_plot or save_plot:
        _plot_comparison(
            comp_df,
            templates,
            plot_df,
            outpath=args.plot_file or default_plot,
            title_suffix=title_suffix,
            yrange_from_data=args.yrange_from_data,
            full_yrange=args.full_yrange,
            show_data_spline=args.show_data_spline,
            show_plot=show_plot,
            save_plot=save_plot,
            flagged_mask=flagged_mask,
            overplot_binned_df=overplot_binned_df,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
