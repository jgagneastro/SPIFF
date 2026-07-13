#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os, sys, json, math, shutil, tempfile, pathlib, subprocess, time
import traceback
import random
import io
import urllib.request
import warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.utils.data import download_file
from astroquery.utils.tap.core import TapPlus
from astroquery.ipac.irsa import Irsa

from pyvo.dal.adhoc import DatalinkResults
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .spherex_binning import write_binned_spherex_spectrum_csv

# --- helpers for safe naming ---

import re as _re

def _append_log(log_path, text: str):
    """Append a line of text to a log file, best-effort."""
    try:
        lp = pathlib.Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "a") as fh:
            fh.write(text.rstrip("\n") + "\n")
    except Exception:
        pass


def _write_row_jsonl(handle, row: dict) -> str:
    """Append one serialized result row and return the emitted JSON payload."""
    payload = json.dumps(
        row,
        allow_nan=True,
        default=lambda value: value.item() if hasattr(value, "item") else str(value),
    )
    handle.write(payload + "\n")
    handle.flush()
    return payload


def _coerce_finite_float(value: object) -> float:
    """Return a finite float for numeric-like scalar metadata, else NaN."""
    try:
        coerced = pd.to_numeric(value, errors="coerce")
        if not np.isscalar(coerced):
            return np.nan
        result = float(coerced)
    except (TypeError, ValueError):
        return np.nan
    return result if math.isfinite(result) else np.nan


def _infer_obs_collection(obs_collection: object, fits_url: object = None, hdr: object = None) -> str:
    """
    Infer SPHEREx collection name with preference order:
      1) explicit FITS header keywords
      2) FITS URL path markers
      3) provided obs_collection value
    """
    # 1) Header-based collection if present
    try:
        if hdr is not None:
            for k in ("OBS_COLLECTION", "OBS-COLL", "COLLECTION", "OBSCOL"):
                v = hdr.get(k, None)
                if v is not None and str(v).strip():
                    sv = str(v).strip().lower()
                    if "qr2_deep" in sv or "spherex_qr2_deep" in sv:
                        return "spherex_qr2_deep"
                    if "qr2" in sv or "spherex_qr2" in sv:
                        return "spherex_qr2"
                    return str(v).strip()
    except Exception:
        pass

    # 2) URL-based collection inference
    try:
        u = str(fits_url or "").strip().lower()
        if u:
            if "/qr2_deep/" in u or "spherex_qr2_deep" in u:
                return "spherex_qr2_deep"
            if "/qr2/" in u or "spherex_qr2" in u:
                return "spherex_qr2"
    except Exception:
        pass

    # 3) Fallback to provided collection
    try:
        s = str(obs_collection or "").strip()
        if s:
            sv = s.lower()
            if "qr2_deep" in sv:
                return "spherex_qr2_deep"
            if "qr2" in sv:
                return "spherex_qr2"
            return s
    except Exception:
        pass
    return ""

def _extract_obsid_bandpass_from_text(text: object) -> tuple[str | None, str | None]:
    """Best-effort parse of SPHEREx OBSID and D# bandpass from URL/DID/text."""
    s = str(text or "")
    if not s:
        return None, None
    # FITS filename/path form:
    # level2_2025W37_1A_0174_3D3_spx_l2b-...
    # Note: there is no underscore between the final obsid digit and D#.
    m = _re.search(r'level2_((?:19|20)\d{2}W\d{2}_[12][AB]_\d{4}_\d)(D\d{1,2})_spx', s, flags=_re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    # DID-like form:
    # ...?2026W03_2A_0105_4/D1
    m = _re.search(r'((?:19|20)\d{2}W\d{2}_[12][AB]_\d{4}_\d)\s*[/_]\s*(D\d{1,2})', s, flags=_re.IGNORECASE)
    if m:
        return m.group(1).upper(), m.group(2).upper()
    # Bandpass-only fallback
    m = _re.search(r'(?:[/_\s]|^)(D\d{1,2})(?:[/_\s]|$)', s, flags=_re.IGNORECASE)
    bp = m.group(1).upper() if m else None
    return None, bp

def _derive_obsidbp_from_row(row: pd.Series | dict) -> str:
    """Canonical OBSID+BP key used by --skip-obsidbps matching."""
    obsid = None
    bandpass = None
    for src in (
        row.get("access_url", ""),
        row.get("obs_publisher_did", ""),
        row.get("energy_bandpassname", ""),
    ):
        o, b = _extract_obsid_bandpass_from_text(src)
        if obsid is None and o:
            obsid = o
        if bandpass is None and b:
            bandpass = b
        if obsid and bandpass:
            break
    if not obsid:
        raw = str(row.get("obs_id", "") or "").strip()
        m = _re.search(r'((?:19|20)\d{2}W\d{2}_[12][AB]_\d{4}_\d)', raw, flags=_re.IGNORECASE)
        obsid = (m.group(1) if m else raw).upper() if raw else ""
    if not bandpass:
        bandpass = "UNKNOWNBP"
    return f"{obsid}{bandpass}".upper()


def _safe_dirname(name: str) -> str:
    """Return a filesystem-safe version of `name` (ASCII, dashes/underscores only)."""
    if not isinstance(name, str):
        return ""
    # strip leading/trailing whitespace
    s = name.strip()
    # replace whitespace with single dash
    s = _re.sub(r"\s+", "-", s)
    # drop any character that isn't alnum, dash, underscore, or dot
    s = _re.sub(r"[^A-Za-z0-9._-]", "", s)
    # collapse repeated dashes/underscores
    s = _re.sub(r"[-_]{2,}", "-", s)
    # avoid empty
    return s or "target"

# --- epoch/PM/parallax projection helper ---

def project_to_epoch(ref_ra_deg: float,
                     ref_dec_deg: float,
                     ref_epoch_yr: float | None,
                     pmra_masyr: float | None,
                     pmdec_masyr: float | None,
                     target_mjd: float | None) -> tuple[float, float]:
    """
    Return (ra_deg, dec_deg) projected to the observation epoch given by target_mjd.
    If pm/parallax/epoch are missing, defaults are: pm=0, parallax=0, ref_epoch=J2000.0.
    """
    try:
        # Defaults
        pmra = 0.0 if pmra_masyr is None or not np.isfinite(pmra_masyr) else float(pmra_masyr)
        pmdec = 0.0 if pmdec_masyr is None or not np.isfinite(pmdec_masyr) else float(pmdec_masyr)
        ref_epoch = 2000.0 if ref_epoch_yr is None or not np.isfinite(ref_epoch_yr) else float(ref_epoch_yr)

        # --- tracking variable for fallback ---
        used_fallback = False
        fallback_reason = None

        # If no target epoch, return reference position
        if target_mjd is None or not np.isfinite(target_mjd):
            return float(ref_ra_deg), float(ref_dec_deg)

        # --- Attempt precise propagation with Astropy ---
        tob = Time(float(target_mjd), format='mjd')
        try:
            # Build the starting coordinate at the reference epoch (no parallax/distance)
            # Supply 100 pc distance just to avoid warning
            c0 = SkyCoord(ra=ref_ra_deg * u.deg,
                          dec=ref_dec_deg * u.deg,
                          pm_ra_cosdec=pmra * u.mas/u.yr,
                          pm_dec=pmdec * u.mas/u.yr,
                          distance=100.0*u.pc,
                          obstime=Time(ref_epoch, format='decimalyear'))
            c_obs = c0.apply_space_motion(new_obstime=tob)
            return float(c_obs.ra.deg), float(c_obs.dec.deg)
        except Exception as _e_astropy:
            used_fallback = True
            fallback_reason = str(_e_astropy)
            # Fall back to manual PM-only projection below
            pass

        # --- Manual PM-only fallback (ignores parallax) ---
        # Convert target epoch to Julian years and compute delta-time
        t_jyear = Time(float(target_mjd), format='mjd').jyear
        dt_yr = float(t_jyear - ref_epoch)
        # pmRA* is mu_alpha cos(delta); convert to degrees of RA
        cos_dec = math.cos(math.radians(ref_dec_deg)) if np.isfinite(ref_dec_deg) else 1.0
        deg_per_mas = 1.0 / (1000.0 * 3600.0)
        dra_deg = (pmra / max(cos_dec, 1e-12)) * deg_per_mas * dt_yr
        ddec_deg = pmdec * deg_per_mas * dt_yr
        ra_new = (ref_ra_deg + dra_deg) % 360.0
        dec_new = ref_dec_deg + ddec_deg
        # Clamp Dec to valid range
        dec_new = max(min(dec_new, 90.0), -90.0)
        msg = "   -> [pm-fallback] applied manual PM-only projection"
        if used_fallback and fallback_reason:
            msg += f" | astropy error: {fallback_reason}"
        print(msg)
        return float(ra_new), float(dec_new)
    except Exception as _e_all:
        # Final safety: return reference coords
        return float(ref_ra_deg), float(ref_dec_deg)

# NOTE on PM/Parallax projection:
# We first try Astropy's full space-motion propagation (supports PM + parallax).
# If that fails for any reason (version/metadata issues), we fall back to a
# manual small-angle PM-only update. This ensures large-PM objects still move
# between epochs (e.g., 2016 → 2025) rather than silently returning the
# reference position.

# ---------- user knobs ----------
DEFAULT_RA  = 124.3726489270054
DEFAULT_DEC = -61.91325512786739
SEARCH_RADIUS_ARCMIN = 1.0     # radius for ObsCore search
#CATALOGS = ("spherex_qr", "spherex_qr_deep")
CATALOGS = ("spherex_qr2", "spherex_qr2_deep")
FIT_RADIUS_PX = 4.0
ULTRANEST_QUIET = True
USE_ULTRANEST = True  # set False to skip UltraNest and use SciPy-only in analyzer
SAVE_FIGS = True
# Epoch (decimal year) to which we project reference coords BEFORE querying ObsCore
QUERY_EPOCH_DECIMAL_YEAR = 2025.5
OUTDIR = (pathlib.Path.cwd() / "spiff_outputs").resolve()
RESULTS_CSV = OUTDIR / "results.csv"
RESULTS_JSONL = OUTDIR / "results.jsonl"
DOWNLOAD_MANIFEST_JSONL = OUTDIR / "download_manifest.jsonl"
DOWNLOAD_MANIFEST_CSV = OUTDIR / "download_manifest.csv"
BINNED_SPECTRUM_CSV = OUTDIR / "binned_spectrum.csv"
# Exact CSV schema (order matters!)
CSV_FIELDS = [
    # PARAMETERS FOR THE BATCH FITS DOWNLOADER
    "target_name",
    "reference_ra_deg",
    "reference_dec_deg",
    "reference_crd_epoch_yr",
    "reference_pmra_masyr",
    "reference_pmdec_masyr",

    # ANALYZER CODE PARAMETERS
    "input_ra_deg",
    "input_dec_deg",
    "fit_radius_px",
    "box_size_bcg_subtract",

    # FITS-PARSER RELATED STUFF (wrapper)
    "fits_path",
    "access_url",
    "fits_analysis_status",
    "fits_analysis_comment",
    "analyzer_stdout",
    "obs_collection",

    # FITS-HEADER RELATED STUFF
    "obsid",
    "bandpass",
    "expid",
    "mjd_avg",
    "detector_id",
    "psf_index",
    "omega_sr",
    "px_scale_arcsec",

    # FLAGS
    "near_cutout_edge",
    "near_detector_edge",
    "near_bcg_star",
    "n_pix_flagged_in_fit",
    "n_pix_used_in_fit",
    "n_pix_total_in_fit",

    # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
    "ap_radius_px",
    "ap_radius_forced",
    "ap_flux_MJysr",
    "ap_flux_MJysr_err",
    "ap_snr",
    "ap_centroid_err_px",
    "ap_flux_uJy",
    "ap_flux_uJy_err",
    "ap_xcen_cutout",
    "ap_ycen_cutout",
    "ap_xcen_fullim",
    "ap_ycen_fullim",

    # CENTER OF MASS OUTPUTS
    "com_xcen_cutout",
    "com_ycen_cutout",
    "com_xcen_fullim",
    "com_ycen_fullim",
    "com_ra_deg",
    "com_dec_deg",
    "com_sep_as",
    "com_wv_um",
    "com_wv_width_um",

    # PSF-SCIPY OUTPUTS
    "psf_scipy_method_used",
    "psf_scipy_status",
    "psf_scipy_flux_MJysr",
    "psf_scipy_flux_MJysr_err",
    "psf_scipy_snr",
    "psf_scipy_flux_uJy",
    "psf_scipy_flux_uJy_err",
    "psf_scipy_dx",
    "psf_scipy_dy",
    "psf_scipy_xcen_cutout",
    "psf_scipy_ycen_cutout",
    "psf_scipy_xcen_fullim",
    "psf_scipy_ycen_fullim",
    "psf_scipy_ra_deg",
    "psf_scipy_ra_err_mas",
    "psf_scipy_dec_deg",
    "psf_scipy_dec_err_mas",
    "psf_scipy_ra_dec_cov_mas2",
    "psf_scipy_sep_as",
    "psf_scipy_wv_um",
    "psf_scipy_wv_um_err",
    "psf_scipy_wv_width_um",
    "psf_scipy_wv_width_um_err",
    "psf_scipy_chi2",
    "psf_scipy_dof",

    # PSF-ultranest outputs
    "psf_un_flux_MJysr",
    "psf_un_flux_MJysr_err",
    "psf_un_snr",
    "psf_un_flux_uJy",
    "psf_un_flux_uJy_err",
    "psf_un_dx",
    "psf_un_dy",
    "psf_un_xcen_cutout",
    "psf_un_ycen_cutout",
    "psf_un_xcen_fullim",
    "psf_un_ycen_fullim",
    "psf_un_xcen_err",
    "psf_un_ycen_err",
    "psf_un_ra_deg",
    "psf_un_ra_err_mas",
    "psf_un_dec_deg",
    "psf_un_dec_err_mas",
    "psf_un_sep_as",
    "psf_un_ra_dec_cov_mas2",
    "psf_un_wv_um",
    "psf_un_wv_um_err",
    "psf_un_wv_width_um",
    "psf_un_wv_width_um_err",
    "psf_un_chi2",
    "psf_un_dof",
]
# ---------- user knobs ----------
MAX_ROWS = None  # if set, limit how many rows/images to process
DISCOVERY_MAXREC = 25000  # max rows returned by IRSA SIA/TAP discovery queries
# --------------------------------
FIGS_DIR = OUTDIR / "figs"
# Sibling directory for kept FITS files
FITS_DIR = OUTDIR / "fits"


def _download_manifest_row_from_fits(
    *,
    fits_path: pathlib.Path,
    access_url: str,
    obs_collection: str,
    obsid: object,
    bandpass: object,
    expid: object,
    mjd_avg: object,
    detector_id: object,
    approx_wv_um: object,
) -> dict[str, object]:
    return {
        "local_fits_path": str(fits_path),
        "access_url": str(access_url or ""),
        "obs_collection": str(obs_collection or ""),
        "obsid": None if obsid is None else str(obsid),
        "bandpass": None if bandpass is None else str(bandpass),
        "expid": None if expid is None else str(expid),
        "mjd_avg": mjd_avg,
        "detector_id": detector_id,
        "approx_wv_um": approx_wv_um,
    }


def _append_download_manifest(row: dict[str, object]) -> None:
    header_needed = not DOWNLOAD_MANIFEST_CSV.exists()
    pd.DataFrame([row]).to_csv(DOWNLOAD_MANIFEST_CSV, index=False, mode="a", header=header_needed)
    with open(DOWNLOAD_MANIFEST_JSONL, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, allow_nan=True, default=lambda o: (o.item() if hasattr(o, "item") else str(o))) + "\n")


def _inventory_predownloaded_fits() -> pd.DataFrame:
    if DOWNLOAD_MANIFEST_CSV.exists():
        try:
            df = pd.read_csv(DOWNLOAD_MANIFEST_CSV)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[batch] warning: failed to read download manifest CSV {DOWNLOAD_MANIFEST_CSV}: {e}")

    if not FITS_DIR.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for path in sorted(FITS_DIR.glob("*.fits")):
        if not path.is_file():
            continue
        obsid = None
        bandpass = None
        expid = None
        mjd_avg = None
        detector_id = None
        approx_wv_um = np.nan
        try:
            with fits.open(path) as hdul:
                hdr = (hdul[1].header if len(hdul) > 1 else hdul[0].header)
                mjd_avg = hdr.get('MJD-AVG', hdr.get('MJD_EPOCH_AVG', hdr.get('MJD', None)))
                detector_id = hdr.get('DETECTOR', None)
                bandpass = "D" + str(detector_id) if detector_id is not None else None
                obsid = hdr.get('OBSID', hdr.get('OBS_ID', None))
                expid = hdr.get('EXPIDN', hdr.get('EXPID', None))
        except Exception as e:
            print(f"[batch] warning: failed to inspect local FITS {path}: {e}")
        rows.append(
            _download_manifest_row_from_fits(
                fits_path=path.resolve(),
                access_url="",
                obs_collection="",
                obsid=obsid,
                bandpass=bandpass,
                expid=expid,
                mjd_avg=mjd_avg,
                detector_id=detector_id,
                approx_wv_um=approx_wv_um,
            )
        )
    return pd.DataFrame(rows)
# --------------------------------

IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
IRSA_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
ANALYZER_MODULE = "spiff.single_fit"

# --- IRSA TAP transient error retry knobs ---
TAP_MAX_RETRIES = 3          # number of attempts for transient TAP errors
TAP_RETRY_SLEEP_SEC = 30     # sleep between retries (seconds)

# --- FITS download transient error retry knobs ---
DOWNLOAD_MAX_RETRIES = 5        # number of attempts for transient download errors
DOWNLOAD_RETRY_SLEEP_SEC = 45   # sleep between retries (seconds)
# First 8 retries are intentionally short (<=5s) for quick recovery from brief hiccups.
DOWNLOAD_RETRY_SCHEDULE_SEC = [1, 1, 2, 2, 3, 3, 4, 5, 15, 30, 60, 120, 300, 900]
DEFAULT_S3_CUTOUT_SIZE_PX = 20

def _is_transient_download_error(e: Exception) -> bool:
    """Return True for transient HTTP/proxy/network errors during FITS download."""
    try:
        msg = (str(e) or "").lower()
    except Exception:
        msg = ""

    # Common urllib/HTTP proxy failures
    transient_markers = [
        "502",
        "503",
        "504",
        "proxy error",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "timed out",
        "timeout",
        "connection reset",
        "reset by peer",
        "connection aborted",
        "remote disconnected",
        "temporary failure",
        "name resolution",
        "dns",
        "incompleteread",
    ]
    if any(m in msg for m in transient_markers):
        return True

    # Detect urllib HTTPError codes when available
    try:
        code = getattr(e, "code", None)
        if code in (502, 503, 504):
            return True
    except Exception:
        pass

    # Fallback: class name heuristics
    try:
        cls = e.__class__.__name__.lower()
    except Exception:
        cls = ""
    transient_class_markers = [
        "httperror",
        "urlerror",
        "timeout",
        "sockettimeout",
        "connectionerror",
        "proxyerror",
        "protocolerror",
    ]
    return any(m in cls for m in transient_class_markers)

def _infer_download_failure_phase(e: Exception) -> str:
    """Best-effort categorizer for where a download likely failed."""
    try:
        msg = (str(e) or "").lower()
    except Exception:
        msg = ""
    try:
        cls = (e.__class__.__name__ or "").lower()
    except Exception:
        cls = ""
    s = f"{cls} {msg}"
    if any(k in s for k in ("name resolution", "dns", "gaierror", "nodename nor servname")):
        return "dns"
    if any(k in s for k in ("connecttimeout", "connection timed out", "failed to establish a new connection", "connection refused")):
        return "connect"
    if any(k in s for k in ("ssl", "certificate", "tls")):
        return "tls"
    if any(k in s for k in ("readtimeout", "timed out", "timeout")):
        return "read"
    if any(k in s for k in ("incompleteread", "chunkedencodingerror", "protocolerror", "remote disconnected", "connection reset", "reset by peer", "connection aborted")):
        return "stream"
    if any(k in s for k in ("502", "503", "504", "bad gateway", "service unavailable", "gateway timeout")):
        return "http_gateway"
    return "unknown"

def _download_file_via_curl(url: str, out_path: str, timeout: int = 600, resume: bool = True) -> str:
    """Download a URL to out_path via curl and return out_path."""
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(min(max(int(timeout), 5), 120)),
        "--max-time",
        str(max(int(timeout), 30)),
        "--output",
        out_path,
        str(url),
    ]
    if resume:
        cmd[cmd.index("--output"):cmd.index("--output")] = ["--continue-at", "-"]
    subprocess.run(cmd, check=True)
    return out_path

def _download_file_via_wget(url: str, out_path: str, timeout: int = 600) -> str:
    """Download a URL to out_path via wget and return out_path."""
    cmd = [
        "wget",
        "--show-progress",
        "--tries=1",
        "--dns-timeout=30",
        "--connect-timeout",
        str(min(max(int(timeout), 5), 120)),
        "--read-timeout",
        str(max(int(timeout), 30)),
        "-O",
        out_path,
        str(url),
    ]
    subprocess.run(cmd, check=True)
    return out_path

def _s3_key_from_fits_url(url: str) -> str:
    """Map an IRSA FITS URL to an S3 key inside the SPHEREx bucket."""
    u = str(url or "").strip()
    if not u:
        raise ValueError("empty URL")
    if u.startswith("s3://"):
        parsed = urlparse(u)
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"Cannot parse S3 key from URL: {u}")
        return key
    parsed = urlparse(u)
    path = parsed.path or ""
    marker = "/ibe/data/spherex/"
    i = path.find(marker)
    if i >= 0:
        key = path[i + len(marker):].lstrip("/")
        if key:
            return key
    marker2 = "/data/spherex/"
    i2 = path.find(marker2)
    if i2 >= 0:
        key = path[i2 + len(marker2):].lstrip("/")
        if key:
            return key
    raise ValueError(f"Cannot map FITS URL to S3 key: {u}")

def _write_fits_cutout(
    hdul,
    out_path: str,
    *,
    xpix: float,
    ypix: float,
    cutout_size_px: int = DEFAULT_S3_CUTOUT_SIZE_PX,
) -> tuple[int, int, int, int]:
    """Write detector-plane sections without materializing full remote images."""
    if len(hdul) < 4:
        raise RuntimeError("S3 cutout requires at least HDUs 1/2/3 with image data.")

    hdr_img = hdul[1].header
    if int(hdr_img.get("NAXIS", 0) or 0) != 2:
        raise RuntimeError("S3 cutout IMAGE HDU is not a two-dimensional image.")
    nx = int(hdr_img.get("NAXIS1", 0) or 0)
    ny = int(hdr_img.get("NAXIS2", 0) or 0)
    if nx <= 0 or ny <= 0:
        raise RuntimeError("S3 cutout IMAGE HDU has invalid detector dimensions.")

    size = max(8, int(cutout_size_px))
    width = min(size, nx)
    height = min(size, ny)
    cx = int(round(float(xpix)))
    cy = int(round(float(ypix)))
    x0 = min(max(0, cx - width // 2), nx - width)
    y0 = min(max(0, cy - height // 2), ny - height)
    x1 = x0 + width
    y1 = y0 + height

    out_hdus = [fits.PrimaryHDU(header=hdul[0].header.copy())]
    for idx in range(1, len(hdul)):
        hdu = hdul[idx]
        hdr = hdu.header
        is_detector_plane = (
            int(hdr.get("NAXIS", 0) or 0) == 2
            and int(hdr.get("NAXIS1", 0) or 0) == nx
            and int(hdr.get("NAXIS2", 0) or 0) == ny
        )
        if not is_detector_plane:
            out_hdus.append(hdu.copy())
            continue

        # ImageHDU.section is the key to issuing byte-range reads for only the
        # requested rows. Accessing hdu.data here would fetch the full detector.
        arr = np.asarray(hdu.section[y0:y1, x0:x1])
        out_hdr = hdr.copy()
        if "CRPIX1" in out_hdr:
            try:
                out_hdr["CRPIX1"] = float(out_hdr["CRPIX1"]) - float(x0)
            except Exception:
                pass
        if "CRPIX2" in out_hdr:
            try:
                out_hdr["CRPIX2"] = float(out_hdr["CRPIX2"]) - float(y0)
            except Exception:
                pass
        if idx == 1:
            out_hdr["SPXORX0"] = (int(x0), "Original detector X0 for S3 cutout")
            out_hdr["SPXORY0"] = (int(y0), "Original detector Y0 for S3 cutout")
            out_hdr["SPXNX"] = (int(nx), "Original detector NAXIS1 before S3 cutout")
            out_hdr["SPXNY"] = (int(ny), "Original detector NAXIS2 before S3 cutout")
        out_hdus.append(fits.ImageHDU(data=arr, header=out_hdr, name=hdu.name))

    fits.HDUList(out_hdus).writeto(out_path, overwrite=True)
    return x0, x1, y0, y1

def _download_file_via_s3(url: str, out_path: str) -> str:
    """Download a FITS file via S3 using s3fs."""
    try:
        import s3fs  # type: ignore
    except Exception as e:
        raise RuntimeError("s3 downloader requested but s3fs is not available") from e
    bucket = str(globals().get("S3_BUCKET", "nasa-irsa-spherex")).strip() or "nasa-irsa-spherex"
    key = _s3_key_from_fits_url(url)
    print(f"[batch] S3 download: bucket={bucket} key={key}")
    use_cutout = bool(globals().get("S3_CUTOUT", True))
    cutout_size_px = int(globals().get("S3_CUTOUT_SIZE_PX", DEFAULT_S3_CUTOUT_SIZE_PX))
    center_ra_deg = globals().get("S3_CUTOUT_CENTER_RA_DEG", None)
    center_dec_deg = globals().get("S3_CUTOUT_CENTER_DEC_DEG", None)
    ref_epoch_yr = globals().get("REFERENCE_CRD_EPOCH_YR", None)
    pmra_masyr = globals().get("REFERENCE_PMRA_MASYR", None)
    pmdec_masyr = globals().get("REFERENCE_PMDEC_MASYR", None)
    s3 = s3fs.S3FileSystem(anon=True)
    if use_cutout and center_ra_deg is not None and center_dec_deg is not None:
        try:
            remote_fh = s3.open(
                f"{bucket}/{key}",
                mode="rb",
                block_size=256 * 1024,
                cache_type="bytes",
            )
            with remote_fh, fits.open(
                remote_fh,
                memmap=False,
                lazy_load_hdus=True,
            ) as hdul:
                hdr_img = hdul[1].header
                obs_mjd = hdr_img.get("MJD-AVG", hdr_img.get("MJD_EPOCH_AVG", hdr_img.get("MJD", None)))
                adj_ra_deg, adj_dec_deg = project_to_epoch(
                    ref_ra_deg=float(center_ra_deg),
                    ref_dec_deg=float(center_dec_deg),
                    ref_epoch_yr=ref_epoch_yr,
                    pmra_masyr=pmra_masyr,
                    pmdec_masyr=pmdec_masyr,
                    target_mjd=obs_mjd,
                )
                w = WCS(hdr_img)
                sky = SkyCoord(adj_ra_deg * u.deg, adj_dec_deg * u.deg, frame="icrs")
                xpix, ypix = w.world_to_pixel(sky)
                x0, x1, y0, y1 = _write_fits_cutout(
                    hdul,
                    out_path,
                    xpix=float(xpix),
                    ypix=float(ypix),
                    cutout_size_px=cutout_size_px,
                )
                print(
                    f"[batch] S3 cutout wrote local FITS: x=[{x0}:{x1}) y=[{y0}:{y1}) "
                    f"size={x1-x0}x{y1-y0}px"
                )
                return out_path
        except Exception as e:
            print("[batch] ==================================================================")
            print("[batch] WARNING: S3 cutout failed; FALLING BACK TO FULL-FILE DOWNLOAD NOW")
            print(f"[batch] reason: {e}")
            print("[batch] ==================================================================")
            print(f"[batch] Full S3 object download in progress: s3://{bucket}/{key}")
    elif use_cutout and (center_ra_deg is None or center_dec_deg is None):
        print("[batch] ==================================================================")
        print("[batch] WARNING: S3 cutout requested but target coords were unavailable")
        print("[batch] FALLING BACK TO FULL-FILE DOWNLOAD NOW")
        print("[batch] ==================================================================")
        print(f"[batch] Full S3 object download in progress: s3://{bucket}/{key}")
    s3.get(f"{bucket}/{key}", out_path)
    try:
        full_bytes = os.path.getsize(out_path)
    except Exception:
        full_bytes = None
    if full_bytes is not None:
        print(f"[batch] Full S3 download complete: {full_bytes} bytes")
    else:
        print("[batch] Full S3 download complete")
    return out_path

def _download_file_with_retries(url: str, *, max_retries: int | None = None, timeout: int = 600):
    """Download a file with retries on transient errors. Returns local path string."""
    schedule = list(globals().get("DOWNLOAD_RETRY_SCHEDULE_SEC", DOWNLOAD_RETRY_SCHEDULE_SEC))
    if max_retries is None:
        max_retries = int(globals().get("DOWNLOAD_MAX_RETRIES", DOWNLOAD_MAX_RETRIES))
    sleep_sec = float(globals().get("DOWNLOAD_RETRY_SLEEP_SEC", DOWNLOAD_RETRY_SLEEP_SEC))
    downloader = str(globals().get("DOWNLOADER", "s3")).strip().lower()
    if downloader not in ("wget", "curl", "astropy", "s3"):
        downloader = "s3"
    use_curl_resume = bool(globals().get("CURL_RESUME", False))
    s3_bucket = str(globals().get("S3_BUCKET", "nasa-irsa-spherex")).strip() or "nasa-irsa-spherex"
    s3_cutout = bool(globals().get("S3_CUTOUT", True))
    s3_cutout_size_px = int(globals().get("S3_CUTOUT_SIZE_PX", DEFAULT_S3_CUTOUT_SIZE_PX))
    s3_cutout_expand_on_masked = bool(globals().get("S3_CUTOUT_EXPAND_ON_MASKED", True))
    s3_cutout_retry_size_px = int(globals().get("S3_CUTOUT_RETRY_SIZE_PX", 64))

    last_exc = None
    if schedule:
        attempts = len(schedule) + 1
    else:
        attempts = max_retries
    if downloader == "s3":
        print(
            f"[batch] Download backend: s3 (bucket={s3_bucket}, "
            f"cutout={'on' if s3_cutout else 'off'}, cutout_size_px={s3_cutout_size_px}, "
            f"expand_on_masked={'on' if s3_cutout_expand_on_masked else 'off'}, "
            f"retry_size_px={s3_cutout_retry_size_px})"
        )
    elif downloader == "curl":
        print(f"[batch] Download backend: curl (resume={'on' if use_curl_resume else 'off'})")
    else:
        print(f"[batch] Download backend: {downloader}")
    # For subprocess-based modes, use one temp output file across attempts.
    dl_out_path = None
    if downloader in ("wget", "curl", "s3"):
        fd, dl_out_path = tempfile.mkstemp(prefix="spherex_dl_", suffix=".fits")
        os.close(fd)

    for attempt in range(1, attempts + 1):
        t0 = time.monotonic()
        try:
            if downloader == "curl":
                try:
                    out = _download_file_via_curl(
                        url,
                        out_path=str(dl_out_path),
                        timeout=timeout,
                        resume=use_curl_resume,
                    )
                except subprocess.CalledProcessError as ce:
                    # curl exit 33: server/path does not support byte ranges (resume).
                    # Fallback to one-shot full download for this same attempt.
                    if use_curl_resume and int(getattr(ce, "returncode", -1)) == 33:
                        try:
                            if os.path.exists(str(dl_out_path)):
                                os.remove(str(dl_out_path))
                        except Exception:
                            pass
                        print("[batch] Resume unsupported by server/path (curl exit 33); retrying this attempt as full download.")
                        out = _download_file_via_curl(url, out_path=str(dl_out_path), timeout=timeout, resume=False)
                    else:
                        raise
            elif downloader == "wget":
                out = _download_file_via_wget(url, out_path=str(dl_out_path), timeout=timeout)
            elif downloader == "s3":
                out = _download_file_via_s3(url, out_path=str(dl_out_path))
            else:
                out = download_file(url, cache=False, show_progress=True, timeout=timeout)
            dt = time.monotonic() - t0
            print(
                f"[batch] Download attempt {attempt}/{attempts} succeeded in {dt:.2f}s "
                f"(downloader={downloader}, timeout={timeout}s, curl_resume={use_curl_resume})"
            )
            return out
        except Exception as e:
            last_exc = e
            dt = time.monotonic() - t0
            host = ""
            try:
                host = str(url).split("://", 1)[-1].split("/", 1)[0]
            except Exception:
                host = ""
            phase = _infer_download_failure_phase(e)
            partial_size = None
            if downloader in ("wget", "curl", "s3") and dl_out_path:
                try:
                    partial_size = os.path.getsize(dl_out_path)
                except Exception:
                    partial_size = None
            partial_txt = f", partial_bytes={partial_size}" if partial_size is not None else ""
            print(
                f"[batch] Download attempt {attempt}/{attempts} failed after {dt:.2f}s "
                f"(host={host or 'unknown'}, phase={phase}, err={e.__class__.__name__}{partial_txt}): {e!r}"
            )
            if _is_transient_download_error(e) and attempt < attempts:
                wait_s = schedule[attempt - 1] if schedule else sleep_sec
                print(f"[batch] Sleeping {wait_s:.0f}s then retrying download...")
                try:
                    time.sleep(wait_s)
                except Exception:
                    pass
                continue
            if downloader in ("wget", "curl", "s3") and dl_out_path:
                try:
                    if os.path.exists(dl_out_path):
                        os.remove(dl_out_path)
                except Exception:
                    pass
            raise
    if downloader in ("wget", "curl", "s3") and dl_out_path:
        try:
            if os.path.exists(dl_out_path):
                os.remove(dl_out_path)
        except Exception:
            pass
    raise last_exc

def _is_transient_tap_error(e: Exception) -> bool:
    """Return True for transient network/proxy/TAP availability errors."""
    try:
        msg = (str(e) or "").lower()
    except Exception:
        msg = ""

    # DALServiceError messages often include HTTP status and proxy wording.
    transient_markers = [
        "502",
        "503",
        "504",
        "proxy error",
        "server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "timed out",
        "timeout",
        "connection aborted",
        "connection reset",
        "connection error",
        "remote disconnected",
        "temporary failure",
        "name resolution",
        "dns",
    ]
    if any(m in msg for m in transient_markers):
        return True

    # Fallback: detect common exception class names without importing requests/urllib3
    try:
        cls = e.__class__.__name__.lower()
    except Exception:
        cls = ""
    transient_class_markers = [
        "timeouts",
        "timeout",
        "connectionerror",
        "proxyerror",
        "protocolerror",
        "readtimeout",
        "connecttimeout",
        "dals"  # DALServiceError subclasses vary
    ]
    return any(m in cls for m in transient_class_markers)

def _irsa_query_tap_with_retries(adql: str, max_retries: int | None = None):
    """Run Irsa.query_tap(adql) with retries on transient errors."""
    if max_retries is None:
        max_retries = int(globals().get("TAP_MAX_RETRIES", TAP_MAX_RETRIES))
    sleep_sec = float(globals().get("TAP_RETRY_SLEEP_SEC", TAP_RETRY_SLEEP_SEC))

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return Irsa.query_tap(adql)
        except Exception as e:
            last_exc = e
            if _is_transient_tap_error(e) and attempt < max_retries:
                print(f"[batch] TAP transient error (attempt {attempt}/{max_retries}): {e!r}")
                print(f"[batch] Sleeping {sleep_sec:.0f}s then retrying IRSA TAP...")
                try:
                    time.sleep(sleep_sec)
                except Exception:
                    pass
                continue
            # Non-transient error OR last attempt: re-raise
            raise
    # Should not get here
    raise last_exc

def _irsa_query_sia_csv_with_retries(params: list[tuple[str, str]], max_retries: int | None = None):
    """Run IRSA SIA query returning CSV with retries on transient errors."""
    if max_retries is None:
        max_retries = int(globals().get("TAP_MAX_RETRIES", TAP_MAX_RETRIES))
    sleep_sec = float(globals().get("TAP_RETRY_SLEEP_SEC", TAP_RETRY_SLEEP_SEC))

    qs = urlencode(params, doseq=True)
    url = f"{IRSA_SIA_URL}?{qs}"
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return text
        except Exception as e:
            last_exc = e
            if _is_transient_tap_error(e) and attempt < max_retries:
                print(f"[batch] SIA transient error (attempt {attempt}/{max_retries}): {e!r}")
                print(f"[batch] Sleeping {sleep_sec:.0f}s then retrying IRSA SIA...")
                try:
                    time.sleep(sleep_sec)
                except Exception:
                    pass
                continue
            raise
    raise last_exc

def query_obscore_sia2(ra, dec, radius_arcmin=1.0, collections=("spherex_qr2", "spherex_qr2_deep")):
    """
    Query SPHEREx discovery via IRSA SIA2 in one step.
    Returns DataFrame with the same downstream columns used by query_obscore().
    """
    radius_deg = radius_arcmin / 60.0
    params: list[tuple[str, str]] = []
    for c in collections or ():
        if str(c).strip():
            params.append(("COLLECTION", str(c).strip()))
    params.extend([
        ("POS", f"CIRCLE {float(ra)} {float(dec)} {float(radius_deg)}"),
        ("RESPONSEFORMAT", "CSV"),
        ("MAXREC", str(int(globals().get("DISCOVERY_MAXREC", DISCOVERY_MAXREC)))),
    ])
    text = _irsa_query_sia_csv_with_retries(params)
    if not text or not text.strip():
        return pd.DataFrame()
    # SIA CSV may include comment/header lines beginning with '#'
    lines = [ln for ln in text.splitlines() if ln.strip() and (not ln.lstrip().startswith("#"))]
    if not lines:
        return pd.DataFrame()
    # IRSA's cloud_access JSON can contain unescaped commas, leaving extra
    # trailing CSV fields.  Without index_col=False pandas shifts leading
    # columns into an implicit index, corrupting em_min/em_max and other fields.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
        df = pd.read_csv(io.StringIO("\n".join(lines)), index_col=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize headers and drop duplicate columns to avoid pandas alignment issues
    # (some SIA CSV responses may expose repeated names that trigger MultiIndex unions).
    try:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")]
        df = df.reset_index(drop=True)
    except Exception:
        pass

    # Normalize to lowercase lookup
    colmap = {str(c).strip().lower(): c for c in df.columns}
    def _to_series(v):
        # If duplicate column names exist, pandas may return a DataFrame; take first column.
        if isinstance(v, pd.DataFrame):
            if v.shape[1] >= 1:
                return v.iloc[:, 0]
            return pd.Series([np.nan] * len(df))
        if isinstance(v, pd.Series):
            return v
        try:
            return pd.Series(v)
        except Exception:
            return pd.Series([np.nan] * len(df))

    def getcol(*cands):
        for c in cands:
            key = str(c).strip().lower()
            if key in colmap:
                return _to_series(df[colmap[key]])
        return None

    # Heuristic selector for schema variants/mislabeling
    def _find_by_predicate(name_hint: str, pred):
        cols = [c for c in df.columns if name_hint in str(c).strip().lower()]
        for c in cols:
            s = _to_series(df[c]).astype(str)
            try:
                if bool(pred(s)):
                    return _to_series(df[c])
            except Exception:
                pass
        for c in df.columns:
            s = _to_series(df[c]).astype(str)
            try:
                if bool(pred(s)):
                    return _to_series(df[c])
            except Exception:
                pass
        return None

    obs_collection = getcol("obs_collection", "collection")
    if obs_collection is None:
        obs_collection = _find_by_predicate("collection", lambda s: s.str.lower().str.contains("spherex|exposure", na=False).mean() > 0.2)
    if obs_collection is None:
        obs_collection = pd.Series([""] * len(df))

    obs_title = getcol("obs_title", "title")
    if obs_title is None:
        obs_title = pd.Series([""] * len(df))

    obs_publisher_did = getcol("obs_publisher_did", "publisher_did")
    if obs_publisher_did is None:
        obs_publisher_did = _find_by_predicate("did", lambda s: s.str.contains("ivo://", na=False).mean() > 0.2)
    if obs_publisher_did is None:
        obs_publisher_did = pd.Series([""] * len(df))

    obs_id = getcol("obs_id", "obsid")
    if obs_id is None:
        obs_id = _find_by_predicate("obs", lambda s: s.str.contains(r"\d{4}W\d{2}", regex=True, na=False).mean() > 0.05)
    if obs_id is None:
        obs_id = pd.Series([""] * len(df))

    energy_bandpassname = getcol("energy_bandpassname", "bandpass")
    if energy_bandpassname is None:
        energy_bandpassname = _find_by_predicate("bandpass", lambda s: s.str.contains(r"\bD\d\b", regex=True, na=False).mean() > 0.05)
    if energy_bandpassname is None:
        energy_bandpassname = pd.Series([""] * len(df))

    access_url = getcol("access_url")
    # Some SIA schema variants may map wrong here; enforce URL-like content.
    if access_url is None or (access_url.astype(str).str.contains(r"^https?://|^s3://", regex=True, na=False).mean() < 0.5):
        access_url = _find_by_predicate("access", lambda s: s.str.contains(r"^https?://|^s3://", regex=True, na=False).mean() > 0.5)
    if access_url is None:
        access_url = pd.Series([""] * len(df))

    t_exptime = getcol("t_exptime", "exptime")
    if t_exptime is None or (pd.to_numeric(t_exptime, errors="coerce").notna().mean() < 0.5):
        t_exptime = _find_by_predicate("exptime", lambda s: pd.to_numeric(s, errors="coerce").notna().mean() > 0.5)
    if t_exptime is None:
        t_exptime = pd.Series([np.nan] * len(df))

    em_min = getcol("em_min")
    if em_min is None:
        em_min = pd.Series([np.nan] * len(df))
    em_max = getcol("em_max")
    if em_max is None:
        em_max = pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "obs_collection": obs_collection,
        "obs_title": obs_title,
        "obs_publisher_did": obs_publisher_did,
        "obs_id": obs_id,
        "energy_bandpassname": energy_bandpassname,
        "access_url": access_url,
        "t_exptime": t_exptime,
        "em_min": em_min,
        "em_max": em_max,
    }, index=np.arange(len(df)))

    # Helpful one-line schema debug for SIA mode
    try:
        url_like_frac = out["access_url"].astype(str).str.contains(r"^https?://|^s3://", regex=True, na=False).mean()
        print(f"[batch] SIA schema map: access_url_url_like_frac={url_like_frac:.2f}")
    except Exception:
        pass
    return out

def find_analyze_callable():
    """
    Try to import analyze_file from your analyzer module.
    If not found, return None and we will try CLI mode.
    """
    try:
        mod = importlib.import_module(ANALYZER_MODULE)
        if hasattr(mod, "analyze_file") and callable(mod.analyze_file):
            return mod.analyze_file
        return None
    except Exception:
        return None

def resolve_fits_from_datalink(datalink_url: str, verbose: bool = True):
    """
    Given a DataLink URL (as returned in spherex.obscore.access_url), return a best-guess
    direct FITS file URL by inspecting the DataLink table.

    Strategy:
      - load datalink table
      - identify rows that look like FITS (content_type mentions 'fits' or URL endswith .fits)
      - score candidates: '#this' (primary) preferred, then productType mentions 'image',
        then content_type mentions 'fits'
    Returns (fits_url: str, dl_table_preview: str).

    Raises RuntimeError if no FITS-like entries are found.
    """
    dl = DatalinkResults.from_result_url(str(datalink_url))
    tab = dl.to_table()
    # Build a short preview string for logs
    preview_cols = [c for c in ("semantics", "content_type", "productType", "access_url") if c in tab.colnames]
    preview = str(tab[preview_cols][:8]) if preview_cols else str(tab[:8])

    # Column-safe access (Astropy Table has no .get)
    def col_or_empty(name):
        return np.array([str(x) for x in tab[name]]) if name in tab.colnames else np.array([""] * len(tab))

    semantics = np.char.lower(col_or_empty("semantics"))
    ctype     = np.char.lower(col_or_empty("content_type"))
    ptype     = np.char.lower(col_or_empty("productType"))
    urls      = col_or_empty("access_url")

    # Identify FITS-like rows
    mentions_fits_ct  = (np.char.find(ctype, "fits") >= 0)
    mentions_fits_url = np.array([u.lower().endswith(".fits") for u in urls])
    is_fits           = mentions_fits_ct | mentions_fits_url
    if not is_fits.any():
        raise RuntimeError("DataLink has no obvious FITS links; preview:\n" + preview)

    is_primary   = (semantics == "#this")
    is_img_ptype = (np.char.find(ptype, "image") >= 0)

    # Score: prefer primary (#this), then image-like productType, then FITS-y content_type
    score = (is_primary.astype(int) * 2) + is_img_ptype.astype(int) + mentions_fits_ct.astype(int)

    idxs = np.where(is_fits)[0]
    best = idxs[np.argmax(score[idxs])]
    fits_url = urls[best]

    if verbose:
        print("   [datalink] preview:")
        print(preview)
        print(f"   [datalink] chosen FITS URL: {fits_url}")

    return fits_url, preview

def _apply_datalink_jitter():
    """Small randomized delay before each DataLink resolve to avoid bursty request patterns."""
    jmin = float(globals().get("DATALINK_JITTER_MIN_SEC", 0.2))
    jmax = float(globals().get("DATALINK_JITTER_MAX_SEC", 1.0))
    if not np.isfinite(jmin):
        jmin = 0.0
    if not np.isfinite(jmax):
        jmax = 0.0
    if jmax < jmin:
        jmin, jmax = jmax, jmin
    if jmax <= 0:
        return
    wait_s = random.uniform(max(0.0, jmin), max(0.0, jmax))
    if wait_s > 0:
        print(f"   -> datalink jitter sleep: {wait_s:.2f}s")
        time.sleep(wait_s)

def _extract_best_fits_by_id_from_datalink_table(tab):
    """
    Given a DataLink table containing one or many IDs, pick one best FITS URL per ID.
    Returns (id_to_url: dict[str,str], preview: str).
    """
    preview_cols = [c for c in ("ID", "id", "semantics", "content_type", "productType", "access_url") if c in tab.colnames]
    preview = str(tab[preview_cols][:8]) if preview_cols else str(tab[:8])

    def col_or_empty(name):
        return np.array([str(x) for x in tab[name]]) if name in tab.colnames else np.array([""] * len(tab))

    id_col = None
    for c in ("ID", "id"):
        if c in tab.colnames:
            id_col = c
            break
    if id_col is None:
        raise RuntimeError("DataLink batch table has no ID/id column.")

    ids = col_or_empty(id_col)
    semantics = np.char.lower(col_or_empty("semantics"))
    ctype = np.char.lower(col_or_empty("content_type"))
    ptype = np.char.lower(col_or_empty("productType"))
    urls = col_or_empty("access_url")

    mentions_fits_ct = (np.char.find(ctype, "fits") >= 0)
    mentions_fits_url = np.array([u.lower().endswith(".fits") for u in urls])
    is_fits = mentions_fits_ct | mentions_fits_url
    if not is_fits.any():
        raise RuntimeError("DataLink batch response has no obvious FITS links.")

    is_primary = (semantics == "#this")
    is_img_ptype = (np.char.find(ptype, "image") >= 0)
    score = (is_primary.astype(int) * 2) + is_img_ptype.astype(int) + mentions_fits_ct.astype(int)

    id_to_url = {}
    unique_ids = np.unique(ids)
    for oid in unique_ids:
        mask = (ids == oid) & is_fits
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        best = idxs[np.argmax(score[idxs])]
        id_to_url[str(oid)] = str(urls[best])
    return id_to_url, preview

def _parse_datalink_url_for_batch(datalink_url: str):
    """
    Parse datalink URL and split into endpoint + params-without-ID + single ID value.
    Returns (endpoint_no_query, base_params_without_id, id_value).
    """
    p = urlparse(str(datalink_url))
    q = parse_qsl(p.query, keep_blank_values=True)
    ids = [v for (k, v) in q if k.lower() == "id"]
    if not ids or not str(ids[0]).strip():
        raise RuntimeError("DataLink URL has no ID parameter.")
    base_params = [(k, v) for (k, v) in q if k.lower() != "id"]
    endpoint = urlunparse((p.scheme, p.netloc, p.path, p.params, "", p.fragment))
    return endpoint, base_params, str(ids[0]).strip()

def resolve_fits_from_datalink_batch(index_url_pairs, *, max_ids_per_request=5000, verbose=True):
    """
    Batch-resolve many DataLink URLs by submitting multiple ID values in one request when possible.
    Returns dict[index] -> {"fits_url": str|None, "preview": str|None, "error": str|None}.
    """
    out = {}
    if not index_url_pairs:
        return out

    # Group by same endpoint + same non-ID params, because only ID should vary in batch.
    groups = {}
    for idx, url in index_url_pairs:
        try:
            endpoint, base_params, idv = _parse_datalink_url_for_batch(url)
            key = (endpoint, tuple(base_params))
            groups.setdefault(key, []).append((idx, idv, url))
        except Exception as e:
            out[idx] = {"fits_url": None, "preview": None, "error": f"parse failed: {e}"}

    for (endpoint, base_params_tup), rows in groups.items():
        base_params = list(base_params_tup)
        # Preserve first occurrence order of IDs
        ids_ordered = []
        seen = set()
        for _, idv, _ in rows:
            if idv not in seen:
                seen.add(idv)
                ids_ordered.append(idv)

        chunks = []
        nmax = max(1, int(max_ids_per_request))
        for i in range(0, len(ids_ordered), nmax):
            chunks.append(ids_ordered[i:i+nmax])

        id_to_url_all = {}
        preview_any = None
        batch_error = None
        for chunk_ids in chunks:
            q = list(base_params) + [("ID", x) for x in chunk_ids]
            batch_url = endpoint + "?" + urlencode(q, doseq=True)
            try:
                dl = DatalinkResults.from_result_url(batch_url)
                tab = dl.to_table()
                id_to_url, preview = _extract_best_fits_by_id_from_datalink_table(tab)
                id_to_url_all.update(id_to_url)
                if preview_any is None:
                    preview_any = preview
                if verbose:
                    print(f"   [datalink-batch] resolved {len(id_to_url)}/{len(chunk_ids)} IDs from one request")
            except Exception as e:
                batch_error = str(e)
                if verbose:
                    print(f"   [datalink-batch] failed for chunk of {len(chunk_ids)} IDs: {e}")
                break

        for idx, idv, _ in rows:
            if batch_error is not None:
                out[idx] = {"fits_url": None, "preview": preview_any, "error": f"batch failed: {batch_error}"}
            else:
                fits_url = id_to_url_all.get(idv)
                if fits_url:
                    out[idx] = {"fits_url": fits_url, "preview": preview_any, "error": None}
                else:
                    out[idx] = {"fits_url": None, "preview": preview_any, "error": f"id not found in batch response: {idv}"}
    return out


# Helper function to prefix-flatten dicts for analyzer outputs
def _prefixed(dct, prefix):
    """Return a flat dict with keys prefixed like 'prefix_key' for every key in dct.
    If dct is not a dict, returns {}.
    This helper is used to safely prefix-flatten analyzer outputs (psf/modelfree/flags/headers/summary)."""
    out = {}
    if isinstance(dct, dict):
        for k, v in dct.items():
            try:
                key = f"{prefix}{str(k)}"
            except Exception:
                key = f"{prefix}{k}"
            out[key] = v
    return out


#
# Diagnostic helper: save a simple diagnostic PNG on analyzer failures.
def _save_failure_diagnostic(fits_path, save_dir):
    """
    Save a simple diagnostic PNG next to per-image outputs to help debug failures.
    Shows the first array-like HDU with NaNs handled; highlights that this was a failure.
    """
    try:
        if not globals().get("SAVE_FIGS", True):
            return
        import matplotlib.pyplot as plt
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        diag_path = save_dir / "failure_diagnostic.png"
        with fits.open(fits_path) as hdul:
            data = None
            # pick the first HDU with array data
            for h in hdul:
                if hasattr(h, "data") and isinstance(h.data, np.ndarray):
                    data = h.data
                    if data is not None:
                        break
            if data is None:
                # nothing plottable; create a tiny placeholder so the file exists
                with open(diag_path, "wb") as _ph:
                    _ph.write(b"")
                return
            arr = np.array(data, dtype=float)
            # Replace non-finite with the median of finite values so imshow can render
            finite = np.isfinite(arr)
            if finite.any():
                med = np.nanmedian(arr[finite])
                arr[~finite] = med
            plt.figure()
            plt.imshow(arr, origin="lower")
            plt.title("Failure diagnostic (non-finite replaced); see analyzer_stdout.txt")
            plt.colorbar()
            plt.savefig(diag_path, dpi=150, bbox_inches="tight")
            plt.close()
    except Exception as _e_diag:
        print(f"   -> warning: could not write failure diagnostic: {_e_diag}")


# Standalone function (moved out of _save_failure_diagnostic)
def _derive_status_and_comment(ok: bool, res: dict | None, raw_text: str | None, err_msg: str | None) -> tuple[str, str]:
    """
    Decide a compact fits_analysis_status (<=32 chars) and a human-readable comment.
    Priority:
      - Specific masked-pixels/background-subtraction failures
      - Explicit error message from CLI wrapper
      - Generic no-result / not-ok
      - ok
    """
    text = (raw_text or "")[:100000]
    masked_markers = [
        "All pixels in cutout are masked or non-finite after background subtraction",
        "All pixels in cutout are masked",
        "Mean of empty slice",
        "invalid value encountered in divide",
    ]
    for m in masked_markers:
        if m.lower() in text.lower():
            return ("error_masked_pixels", "All pixels masked after background subtraction (check cutout/background).")
    if not ok:
        if err_msg:
            return ("error_cli", str(err_msg)[:300])
        if isinstance(res, dict):
            st = res.get("status")
            if isinstance(st, str) and st.strip():
                label = st.strip().lower().replace(" ", "_")
                if label.startswith("error"):
                    return (label[:32], st.strip()[:300])
        return ("error_no_result", "Analyzer produced no valid result.")
    return ("ok", "")

def run_analyzer_via_cli(fits_path, ra, dec, fit_radius_px, quiet, save_dir, stdout_path=None, results_path=None):
    """
    Call the analyzer as a subprocess and stream output live.
    Parse either a trailing JSON blob or a line of the form:
        [summary] { ... }
    Returns dict: {"result": <parsed or None>, "raw_text": <stdout string>}
    stdout_path: optional path to a file to tee analyzer stdout.
    """
    import re
    # Where the analyzer will write its JSON results when invoked via CLI
    if results_path is None:
        results_path = pathlib.Path(save_dir) / "analyzer_results.json"
    else:
        results_path = pathlib.Path(results_path)

    py = sys.executable
    cmd = [
        py, "-u", "-m", ANALYZER_MODULE,
        "--fits", str(fits_path),
        "--ra", str(ra),
        "--dec", str(dec),
        "--fit-radius", str(fit_radius_px),
        "--no-show",
        "--figs-dir", str(save_dir),
        "--debug",
        "--save-results",
        "--results-path", str(results_path),
    ]
    # save figs in batch
    if SAVE_FIGS:
        cmd.append("--save-figs")
    # skip UltraNest if requested
    if not globals().get("USE_ULTRANEST", True):
        cmd.append("--scipy-only")
    if globals().get("MAX_PIX_OFFSET") is not None:
        cmd.extend(["--max-pix-offset", str(globals()["MAX_PIX_OFFSET"])])
    if globals().get("AP_RADIUS_PX") is not None:
        cmd.extend(["--aperture-px", str(globals()["AP_RADIUS_PX"])])
    if bool(globals().get("NO_MASKING", False)):
        cmd.append("--no-masking")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )

    collected = []
    fh = open(stdout_path, "w") if stdout_path else None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            collected.append(line)
            if fh:
                try:
                    fh.write(line)
                except Exception:
                    pass
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if fh:
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    full_out = "".join(collected).strip()

    # Read analyzer results from the JSON file written by the analyzer
    parsed = None
    err = None
    try:
        if results_path.exists():
            with open(results_path, "r") as f:
                parsed = json.load(f)
        else:
            err = f"Expected results JSON not found: {results_path}"
    except Exception as e:
        err = f"Failed to read results JSON ({results_path}): {e}"

    ok = (parsed is not None) and (rc == 0)
    if not ok and err is None:
        err = f"Analyzer CLI exited with code {rc} and no results JSON at {results_path}"

    return {"result": parsed, "raw_text": full_out, "rc": rc, "ok": ok, "error": err, "results_path": str(results_path)}

#TODO: Verify but I think I need to remove the COLLECTIONS keyword altogether and just
# accept everything the IRSA TAP endpoint sends me without filtration
# (I think it contains the _deep images too but verify)
def query_obscore(ra, dec, radius_arcmin=1.0, collections=("spherex_qr","spherex_qr_deep")):
    """
    Use IRSA TAP to query SPHEREx ObsCore entries near (ra,dec).

    Notes / robustness:
      * IRSA returns DataLink URLs in access_url (not direct FITS). You must resolve them
        via pyvo.dal.adhoc.DatalinkResults before downloading.
      * Some IRSA deployments return non-gzipped error pages that can confuse the
        default reader. We catch and re-raise with a clearer message.
      * obs_collection values appear as 'spherex_qr...' and 'spherex_qr_deep...'
        (case-insensitive). We therefore use LOWER(obs_collection) LIKE patterns.
      * IRSA TAP prefers the `1=INTERSECTS(...)` form for spatial predicates; some servers
        may reject the `INTERSECTS(...)=1` form in certain combinations.
    """

    radius_deg = radius_arcmin / 60.0

    # TODO: This is the only variant that survived so remove the whole variant strategy
    # Force the 'no_collection_filter' variant; IRSA rejects the others on this server.
    # We post-filter by obs_collection in Python after the query returns.
    # Keep only ObsCore columns currently used downstream in this wrapper.
    # Removed unused columns (e.g., access_format/calib_level/s_ra/s_dec/s_fov/t_min/t_max/obs_release_date)
    # to reduce TAP payload and query overhead.
    adql_variants = [(
        "no_collection_filter",
        f"""
        SELECT TOP {int(globals().get("DISCOVERY_MAXREC", DISCOVERY_MAXREC))}
            obs_collection, obs_title, access_url, obs_publisher_did, t_exptime, obs_id, em_min, em_max, energy_bandpassname
        FROM spherex.obscore
        WHERE 1=INTERSECTS(
                s_region,
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
              )
        """
    )]

    last_err = None
    for tag, adql in adql_variants:
        try:
            #print(f"[batch] Using ADQL variant: {tag}")
            t = _irsa_query_tap_with_retries(adql).to_table()
            if t is not None and len(t) > 0:
                # Keep only columns we currently consume downstream.
                final_cols = [
                    "obs_collection", "obs_title", "obs_publisher_did",
                    "obs_id", "energy_bandpassname", "access_url",
                    "t_exptime", "em_min", "em_max"
                ]
                t = t[final_cols]
                print(f"[batch] ADQL '{tag}' succeeded with {len(t)} rows.")
                return t.to_pandas()
            else:
                print(f"[batch] ADQL '{tag}' returned 0 rows.")
        except Exception as e:
            last_err = e
            print(f"[batch] ADQL '{tag}' failed: {e!r}")

    # As a last resort, probe which collections are nearby to help the user
    try:
        probe_tbl = _irsa_query_tap_with_retries(f"""
        SELECT DISTINCT obs_collection
        FROM spherex.obscore
        WHERE 1=INTERSECTS(
                s_region,
                CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
              )
        """).to_table()
        avail = [str(v) for v in probe_tbl["obs_collection"].tolist()] if probe_tbl is not None else []
    except Exception:
        avail = []

    raise RuntimeError(
        "All ADQL variants failed (IRSA may be returning a non-VOTable error page or rejecting the syntax).\n"
        f"Last error: {last_err!r}\n"
        f"Collections seen by probe: {avail}"
    )

def ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    if (
        globals().get("KEEP_FITS", False)
        or globals().get("DOWNLOAD_ONLY", False)
        or globals().get("CONSUME_DOWNLOADED", False)
    ):
        FITS_DIR.mkdir(parents=True, exist_ok=True)

def main(ra=DEFAULT_RA, dec=DEFAULT_DEC):
    ensure_dirs()
    # --- transient/network failure accounting for higher-level automation ---
    transient_download_errors = 0
    transient_tap_errors = 0
    last_transient_error = None
    globals()["_LV2_TRANSIENT_ERRORS"] = 0
    globals()["_LV2_TRANSIENT_TAP_ERRORS"] = 0
    globals()["_LV2_TRANSIENT_DOWNLOAD_ERRORS"] = 0
    globals()["_LV2_TRANSIENT_ONLY"] = False
    globals()["_LV2_LAST_TRANSIENT_ERROR"] = None
    tname = globals().get("TARGET_NAME")
    if tname:
        print(f"[batch] Target: {tname}")
    print(f"[batch] Output directory: {OUTDIR}")
    # Project reference coords to the desired query epoch to avoid missing fast movers
    try:
        _qry_epoch = float(globals().get("QUERY_EPOCH_DECIMAL_YEAR", QUERY_EPOCH_DECIMAL_YEAR))
    except Exception:
        _qry_epoch = QUERY_EPOCH_DECIMAL_YEAR
    try:
        _qry_mjd = Time(_qry_epoch, format='decimalyear').mjd
    except Exception:
        _qry_mjd = None
    ra_qry, dec_qry = project_to_epoch(
        ref_ra_deg=float(ra),
        ref_dec_deg=float(dec),
        ref_epoch_yr=globals().get("REFERENCE_CRD_EPOCH_YR", None),
        pmra_masyr=globals().get("REFERENCE_PMRA_MASYR", None),
        pmdec_masyr=globals().get("REFERENCE_PMDEC_MASYR", None),
        target_mjd=_qry_mjd,
    )
    if _qry_mjd is not None:
        print(f"[batch] Query epoch: {globals().get('QUERY_EPOCH_DECIMAL_YEAR', QUERY_EPOCH_DECIMAL_YEAR)} (MJD={_qry_mjd:.2f})")
    print(f"[batch] Querying SPHEREx near projected coords RA={ra_qry:.8f}, Dec={dec_qry:.8f} (r={SEARCH_RADIUS_ARCMIN:.1f}' ) ...")
    print(f"[batch] Collections filter: {CATALOGS}")
    use_sia2 = bool(globals().get("USE_SIA2", True))
    used_sia2_discovery = False
    download_only = bool(globals().get("DOWNLOAD_ONLY", False))
    consume_downloaded = bool(globals().get("CONSUME_DOWNLOADED", False))
    download_only_count = 0
    if consume_downloaded:
        print("[batch] Discovery mode: local predownloaded FITS inventory")
        df = _inventory_predownloaded_fits()
    else:
        try:
            if use_sia2:
                print("[batch] Discovery mode: SIA2 (default)")
                try:
                    df = query_obscore_sia2(ra_qry, dec_qry, SEARCH_RADIUS_ARCMIN, CATALOGS)
                    print(f"[batch] SIA2 returned {len(df)} rows.")
                    used_sia2_discovery = True
                except Exception as e_sia:
                    print(f"[batch] SIA2 discovery failed: {e_sia!r}")
                    print("[batch] Falling back to TAP/ObsCore discovery.")
                    df = query_obscore(ra_qry, dec_qry, SEARCH_RADIUS_ARCMIN, CATALOGS)
            else:
                print("[batch] Discovery mode: TAP/ObsCore")
                df = query_obscore(ra_qry, dec_qry, SEARCH_RADIUS_ARCMIN, CATALOGS)
        except Exception as e:
            # If this looks like a transient TAP/network failure, signal to higher-level tooling to skip follow-up work.
            try:
                if _is_transient_tap_error(e):
                    transient_tap_errors += 1
                    last_transient_error = f"TAP/query failed: {e!r}"
                    globals()["_LV2_TRANSIENT_TAP_ERRORS"] = transient_tap_errors
                    globals()["_LV2_LAST_TRANSIENT_ERROR"] = last_transient_error
                    globals()["_LV2_TRANSIENT_ERRORS"] = transient_tap_errors + transient_download_errors
                    globals()["_LV2_TRANSIENT_ONLY"] = True
                    print(f"[LV2_TRANSIENT_TAP_ERROR] {str(e)[:400]}")
                    print("[batch] IRSA TAP query failed due to a transient network/proxy error; skipping processing.")
                    return 0
            except Exception:
                pass

            print("[batch] No ObsCore rows or query failed (possibly outside SPHEREx footprint). Returning without processing.")
            return 0

    # Post-filter by collections if the query had to drop the collection filter
    if CATALOGS and (not used_sia2_discovery) and (not consume_downloaded):
        allowed = tuple(str(c).strip().lower() for c in CATALOGS if str(c).strip())
        before = len(df)
        # Be robust to SIA-style collection identifiers. In some SIA responses, obs_collection
        # may be generic (e.g., "exposure"), so infer collection from DID/URL when needed.
        coll_raw = df["obs_collection"].astype(str).fillna("")
        did_raw = df["obs_publisher_did"].astype(str).fillna("") if "obs_publisher_did" in df.columns else pd.Series([""] * len(df))
        acc_raw = df["access_url"].astype(str).fillna("") if "access_url" in df.columns else pd.Series([""] * len(df))

        coll_norm = (
            coll_raw.str.lower().str.strip()
            .str.replace(r"^ivo://irsa\.ipac/", "", regex=True)
            .str.replace(r"^ivo://", "", regex=True)
        )
        did_norm = did_raw.str.lower().str.strip()
        acc_norm = acc_raw.str.lower().str.strip()

        inferred = pd.Series([""] * len(df), index=df.index, dtype=object)
        inferred = inferred.mask(did_norm.str.contains("spherex_qr2_deep", na=False), "spherex_qr2_deep")
        inferred = inferred.mask(did_norm.str.contains("spherex_qr2", na=False) & inferred.eq(""), "spherex_qr2")
        inferred = inferred.mask(acc_norm.str.contains("/qr2/level2/", na=False) & inferred.eq(""), "spherex_qr2")
        inferred = inferred.mask(acc_norm.str.contains("/qr2_deep/level2/", na=False) & inferred.eq(""), "spherex_qr2_deep")

        # Prefer explicit collection if it looks specific; otherwise use inferred value.
        coll_effective = coll_norm.copy()
        generic_mask = coll_effective.isin(["", "nan", "none", "exposure"])
        coll_effective = coll_effective.mask(generic_mask, inferred)

        mask_coll = pd.Series(False, index=df.index)
        for c in allowed:
            if not c:
                continue
            # Accept exact, prefix, or tokenized containment (/ ? _ delimiters in provider strings)
            mask_coll = mask_coll | coll_effective.eq(c) | coll_effective.str.startswith(c) | coll_effective.str.contains(
                rf"(?:^|[/?_]){_re.escape(c)}(?:$|[/?_])", regex=True
            )
        df = df[mask_coll].copy()
        print(f"[batch] Filtered to collections {CATALOGS}: {len(df)}/{before} rows remain.")
        if before > 0 and len(df) == 0:
            try:
                uniq = sorted(pd.Series(coll_raw).dropna().astype(str).unique().tolist())
                print(f"[batch] SIA/TAP obs_collection unique values (sample up to 12): {uniq[:12]}")
                uniq_did = sorted(pd.Series(did_raw).dropna().astype(str).unique().tolist())
                print(f"[batch] SIA/TAP obs_publisher_did unique values (sample up to 6): {uniq_did[:6]}")
            except Exception:
                pass
    elif CATALOGS and used_sia2_discovery:
        print(f"[batch] Collection post-filter skipped (SIA2 query already constrained by COLLECTION={CATALOGS}).")

    # Enforce minimum datapoints if requested
    min_datapoints = globals().get("MIN_DATAPOINTS")
    if min_datapoints is not None:
        if len(df) < min_datapoints:
            print(f"[batch] Found only {len(df)} IRSA query results, which is fewer than the minimum required ({min_datapoints}). Skipping processing.")
            return 0

    # Proceed only if there are rows to process
    if df.empty:
        print("[batch] No spectral images found in the search region.")
        return 0

    print(f"[batch] UltraNest enabled: {USE_ULTRANEST} (set --scipy-only to disable)")
    print(f"[batch] Found {len(df)} candidate images.")
    total_candidates = int(len(df))

    # Try function import first
    analyze_file = find_analyze_callable()
    if analyze_file:
        print("[batch] Using in-process analyze_file(...)")
    else:
        print("[batch] Falling back to analyzer CLI. (Analyzer must support --fits/--ra/--dec/--figs-dir/--save-figs/--no-show/--debug and --scipy-only.)")

    # results accumulators
    rows_for_csv = []
    skipped_count = 0  # number of candidates skipped due to --skip-obsidbps
    jsonl = open(RESULTS_JSONL, "w")

    # temp directory for downloads (deleted at end)
    with tempfile.TemporaryDirectory(prefix="spherex_dl_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        processed_count = 0  # counts only non-skipped rows actually attempted
        skipset_prefetch = {str(x).strip().upper() for x in globals().get("SKIP_OBSIDBPS", set())}
        datalink_batch_results = {}
        if bool(globals().get("DATALINK_BATCH", True)):
            # Build candidate subset matching this invocation's processing horizon (non-skipped, up to MAX_ROWS).
            batch_candidates = []
            cap = int(globals()["MAX_ROWS"]) if globals().get("MAX_ROWS") is not None else None
            for idx, r0 in df.iterrows():
                try:
                    obsidbp0 = _derive_obsidbp_from_row(r0)
                except Exception:
                    obsidbp0 = ""
                if obsidbp0 and obsidbp0 in skipset_prefetch:
                    continue
                u0 = str(r0.get("access_url", "") or "").strip()
                if not u0:
                    continue
                batch_candidates.append((idx, u0))
                if cap is not None and len(batch_candidates) >= cap:
                    break
            if batch_candidates:
                print(
                    f"[batch] Resolving DataLink in batch for up to {len(batch_candidates)} "
                    f"candidate IDs (max_ids_per_request={int(globals().get('DATALINK_BATCH_MAX_IDS', 5000))})."
                )
                datalink_batch_results = resolve_fits_from_datalink_batch(
                    batch_candidates,
                    max_ids_per_request=int(globals().get("DATALINK_BATCH_MAX_IDS", 5000)),
                    verbose=True,
                )
                n_ok_batch = sum(1 for v in datalink_batch_results.values() if not v.get("error") and v.get("fits_url"))
                n_fail_batch = len(datalink_batch_results) - n_ok_batch
                print(f"[batch] DataLink batch summary: ok={n_ok_batch}, fail={n_fail_batch}")
        for i, r in df.iterrows():
            # ---- early defaults so 'finally' / failure paths are always safe ----
            last_status = "error_unattempted"
            last_comment = ""
            row = None
            write_failure_placeholder = True  # set False for transient failures we want to re-try later without polluting CSV/DB
            # minimal identifiers (may be refined later)
            try:
                url = str(r.get("access_url", ""))
            except Exception:
                url = ""
            try:
                coll = str(r.get("obs_collection", ""))
            except Exception:
                coll = ""
            try:
                title = str(r.get("obs_title", ""))
            except Exception:
                title = ""
            try:
                exptime = float(r.get("t_exptime", np.nan))
            except Exception:
                exptime = np.nan
            # file/dir defaults (updated once we know OBSID/EXPID)
            # Create a per-candidate directory immediately so early failures (datalink/download)
            # have somewhere to write a log.
            save_dir = FIGS_DIR / f"candidate_{i+1:05d}"
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            analyzer_stdout_path = save_dir / "analyzer_stdout.txt"
            fits_path = None
            fits_local_str = ""
            # header/metadata defaults
            obs_mjd = None
            obsid = None
            bandpass = None
            expidn = None
            hdr_detector = None
            # reference/adjusted coordinates defaults
            reference_ra_deg = float(ra)
            reference_dec_deg = float(dec)
            reference_crd_epoch_yr = globals().get("REFERENCE_CRD_EPOCH_YR", None)
            reference_pmra_masyr   = globals().get("REFERENCE_PMRA_MASYR", None)
            reference_pmdec_masyr  = globals().get("REFERENCE_PMDEC_MASYR", None)
            adj_ra_deg = reference_ra_deg
            adj_dec_deg = reference_dec_deg
            # --- Robust skip logic for obsidbp ---
            obsidbp = _derive_obsidbp_from_row(r)

            skipset = {str(x).strip().upper() for x in globals().get("SKIP_OBSIDBPS", set())}
            # removed debugging breakpoint here
            if obsidbp in skipset:
                skipped_count += 1
                # optional debug print
                # print(f"[batch] skipping {obsidbp} (in skipset)")
                continue
            # Enforce MAX_ROWS counting only non-skipped rows
            if globals().get("MAX_ROWS") is not None:
                if processed_count >= int(globals()["MAX_ROWS"]):
                    print(f"[batch] Reached MAX_ROWS={globals()['MAX_ROWS']} (counting non-skipped). Stopping.")
                    break
                processed_count += 1
            
            try:
                url = r["access_url"]
                coll = r.get("obs_collection", "")
                title = r.get("obs_title", "")
                did = r.get("obs_publisher_did", "")
                exptime = float(r.get("t_exptime", np.nan))
                local_fits_path = str(r.get("local_fits_path", "") or "").strip()
                fits_url = str(url or "")

                print(f"[batch {i+1}/{len(df)}] {coll}  {title}  (exp={exptime!r}s)")
                if local_fits_path:
                    fits_path = pathlib.Path(local_fits_path).expanduser().resolve()
                    if not fits_path.exists():
                        msg = f"predownloaded FITS not found: {fits_path}"
                        print(f"   -> {msg}")
                        last_status = "error_download"
                        last_comment = msg[:300]
                        _append_log(analyzer_stdout_path, f"[wrapper] {msg}")
                        continue
                    print(f"   -> using predownloaded FITS: {fits_path}")
                else:
                    # resolve DataLink to a direct FITS (SPHEREx uses DataLink in access_url)
                    print(f"   -> resolving access URL: {url}")
                    try:
                        ent = datalink_batch_results.get(i)
                        if str(url).lower().endswith(".fits"):
                            fits_url = str(url)
                            _preview = None
                            print("   -> direct FITS URL from discovery result (no DataLink resolve needed)")
                        elif ent and (not ent.get("error")) and ent.get("fits_url"):
                            fits_url = str(ent.get("fits_url"))
                            _preview = ent.get("preview")
                            print("   -> datalink resolved via batch request")
                        else:
                            if ent and ent.get("error"):
                                print(f"   -> datalink batch unresolved ({ent.get('error')}); retrying single request")
                            _apply_datalink_jitter()
                            fits_url, _preview = resolve_fits_from_datalink(url, verbose=True)
                    except Exception as e:
                        msg = f"datalink resolution failed: {e}"
                        print(f"   -> {msg}")
                        last_status = "error_datalink"
                        last_comment = str(e)[:300]
                        _append_log(analyzer_stdout_path, f"[wrapper] {msg}")
                        # Transient DataLink/network failures should not create DB placeholder rows.
                        try:
                            if _is_transient_tap_error(e):
                                transient_tap_errors += 1
                                last_transient_error = f"datalink failed: {e!r}"
                                globals()["_LV2_TRANSIENT_TAP_ERRORS"] = transient_tap_errors
                                globals()["_LV2_LAST_TRANSIENT_ERROR"] = last_transient_error
                                write_failure_placeholder = False
                        except Exception:
                            pass
                        continue
                    # download the FITS (with retries on transient proxy/network errors)
                    print(f"   -> downloading FITS: {fits_url}")
                    coll = _infer_obs_collection(coll, fits_url=fits_url, hdr=None)
                    try:
                        globals()["S3_CUTOUT_CENTER_RA_DEG"] = float(reference_ra_deg)
                        globals()["S3_CUTOUT_CENTER_DEC_DEG"] = float(reference_dec_deg)
                        local = _download_file_with_retries(
                            str(fits_url),
                            max_retries=int(globals().get("DOWNLOAD_MAX_RETRIES", DOWNLOAD_MAX_RETRIES)),
                            timeout=10000,
                        )
                    except Exception as e:
                        msg = f"download failed: {e}"
                        print(f"   -> {msg}")
                        last_status = "error_download"
                        last_comment = str(e)[:300]
                        _append_log(analyzer_stdout_path, f"[wrapper] {msg}")

                        # Download failures: after exhausting retries, skip this row without placeholder.
                        try:
                            if _is_transient_download_error(e):
                                transient_download_errors += 1
                                last_transient_error = f"download failed: {e!r}"
                                globals()["_LV2_TRANSIENT_DOWNLOAD_ERRORS"] = transient_download_errors
                                globals()["_LV2_LAST_TRANSIENT_ERROR"] = last_transient_error
                            write_failure_placeholder = False
                        except Exception:
                            write_failure_placeholder = False
                        continue
                    fits_path = pathlib.Path(local)
                    print("   -> source: DataLink (access_url) resolved to FITS")
                    try:
                        size_mb = fits_path.stat().st_size / (1024*1024)
                        print(f"   -> saved to {fits_path}  ({size_mb:.2f} MB)")
                    except Exception:
                        print(f"   -> saved to {fits_path}")
                # Read observation metadata we’ll need for naming and projection
                obs_mjd = None
                obsid = None
                bandpass = None
                expidn = None
                hdr_detector = None
                approxwv_um = np.nan  # default if we can’t derive it

                try:
                    with fits.open(fits_path) as hdul:
                        hdr = (hdul[1].header if len(hdul) > 1 else hdul[0].header)
                        coll = _infer_obs_collection(coll, fits_url=fits_url, hdr=hdr)

                        # MJD for epoch projection
                        obs_mjd = hdr.get('MJD-AVG', hdr.get('MJD_EPOCH_AVG', hdr.get('MJD', None)))

                        # Identifiers for folder naming
                        # (Common SPHEREx L2B keywords; adjust if your files use different spellings)
                        hdr_detector = hdr.get('DETECTOR', None)
                        bandpass = "D" + str(hdr_detector) if hdr_detector is not None else None
                        obsid  = hdr.get('OBSID',  hdr.get('OBS_ID', None))
                        expidn = hdr.get('EXPIDN', hdr.get('EXPID',  None))
                        #import pdb; pdb.set_trace()
                        # 1) Try ObsCore em_min/em_max midpoint (meters -> microns)
                        em_min = _coerce_finite_float(r.get("em_min", np.nan))
                        em_max = _coerce_finite_float(r.get("em_max", np.nan))
                        if math.isfinite(em_min) and math.isfinite(em_max):
                            approxwv_um = 0.5 * (em_min + em_max) * 1e6
                        else:
                            # 2) Fall back to WCS-WAVE median wavelength (µm) if present
                            #    SPHEREx L2B typically stores a grid in an extension named 'WCS-WAVE'
                            #    with VALUES[..., 0] ≡ λ(µm). We take the median over that grid.
                            try:
                                # find an extension that carries the WCS-WAVE VALUES field
                                wcswave_hdu = None
                                for hdu in hdul:
                                    if hasattr(hdu, "name") and isinstance(hdu.name, str):
                                        if hdu.name.strip().upper() in ("WCS-WAVE", "WCSWAVE", "WAVEWCS"):
                                            wcswave_hdu = hdu
                                            break
                                if (wcswave_hdu is not None) and hasattr(wcswave_hdu, "data"):
                                    data = wcswave_hdu.data
                                    # Some builds pack a column named 'VALUES'; others expose a structured array
                                    if isinstance(data, np.ndarray):
                                        if data.dtype.fields and "VALUES" in data.dtype.fields:
                                            vals = data["VALUES"]  # shape ≈ (Ny, Nx, 2)
                                        else:
                                            vals = data  # already the 3-D VALUES cube
                                        vals = np.array(vals)
                                        if vals.ndim == 3 and vals.shape[-1] >= 1:
                                            approxwv_um = float(np.nanmedian(vals[..., 0]))
                            except Exception:
                                pass

                except Exception as e:
                    print(f"   -> warning: could not read header metadata: {e}")

                # Clean up identifiers for folder naming
                obsid_str  = str(obsid)  if (obsid  is not None and str(obsid).strip())  else "unknownOBSID"
                expidn_str = str(expidn) if (expidn is not None and str(expidn).strip()) else "unknownEXPID"
                wv_str     = ("%.1f" % approxwv_um) if np.isfinite(approxwv_um) else None
                bandpass_str = str(bandpass) if (bandpass is not None and str(bandpass).strip()) else "unknownBP"

                reference_ra_deg = float(ra)
                reference_dec_deg = float(dec)
                adj_ra_deg, adj_dec_deg = project_to_epoch(
                    ref_ra_deg=reference_ra_deg,
                    ref_dec_deg=reference_dec_deg,
                    ref_epoch_yr=globals().get("REFERENCE_CRD_EPOCH_YR"),
                    pmra_masyr=globals().get("REFERENCE_PMRA_MASYR"),
                    pmdec_masyr=globals().get("REFERENCE_PMDEC_MASYR"),
                    target_mjd=obs_mjd,
                )
                print(f"   -> epoch-adjusted input coords: RA={adj_ra_deg:.9f} deg, Dec={adj_dec_deg:.9f} deg (MJD={obs_mjd})")
                pmra_log = globals().get("REFERENCE_PMRA_MASYR")
                pmdec_log = globals().get("REFERENCE_PMDEC_MASYR")
                ref_ep_log = globals().get("REFERENCE_CRD_EPOCH_YR")
                print(f"   -> ref: RA={reference_ra_deg:.9f} deg, Dec={reference_dec_deg:.9f} deg, epoch={ref_ep_log}, pmRA*={pmra_log} mas/yr, pmDec={pmdec_log} mas/yr")
                # Per-image figs dir: obsid_expid_approxwv
                # (computed entirely before any fitting)
                img_stub = f"{obsid_str}{bandpass_str}_{expidn_str}"
                if wv_str is not None:
                    img_stub = f"{img_stub}_{wv_str}"
                save_dir = FIGS_DIR / img_stub
                # ensure per-image figs directory exists before analyzer runs
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                
                # file that will capture the analyzer's full stdout for this image
                analyzer_stdout_path = save_dir / "analyzer_stdout.txt"

                if download_only:
                    dest_dir = globals().get("FITS_DIR", OUTDIR / "fits")
                    try:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    dest = dest_dir / f"{img_stub}.fits"
                    if fits_path is not None:
                        try:
                            if fits_path.resolve() != dest.resolve():
                                if dest.exists():
                                    fits_path.unlink()
                                    fits_path = dest
                                else:
                                    shutil.move(str(fits_path), str(dest))
                                    fits_path = dest
                        except Exception:
                            if not dest.exists():
                                raise
                    manifest_row = _download_manifest_row_from_fits(
                        fits_path=pathlib.Path(fits_path),
                        access_url=str(url or ""),
                        obs_collection=str(coll or ""),
                        obsid=obsid,
                        bandpass=bandpass,
                        expid=expidn,
                        mjd_avg=obs_mjd,
                        detector_id=hdr_detector,
                        approx_wv_um=approxwv_um,
                    )
                    _append_download_manifest(manifest_row)
                    download_only_count += 1
                    print(f"   -> download-only saved FITS: {fits_path}")
                    continue
                
                # run the analyzer
                last_status = "error_unknown"
                last_comment = ""
                row = None
                if analyze_file:
                    # Capture analyzer stdout/stderr into the per-image log file AND tee to terminal.
                    try:
                        print("[batch] Launching analyzer directly in-process")

                        class Tee:
                            def __init__(self, *targets):
                                self.targets = targets
                            def write(self, data):
                                for t in self.targets:
                                    try:
                                        t.write(data)
                                        t.flush()
                                    except Exception:
                                        pass
                            def flush(self):
                                for t in self.targets:
                                    try:
                                        t.flush()
                                    except Exception:
                                        pass

                        with open(analyzer_stdout_path, "w") as _logfh:
                            tee_out = Tee(sys.__stdout__, _logfh)
                            tee_err = Tee(sys.__stderr__, _logfh)
                            old_out, old_err = sys.stdout, sys.stderr
                            sys.stdout, sys.stderr = tee_out, tee_err
                            try:
                                res = analyze_file(
                                    str(fits_path), adj_ra_deg, adj_dec_deg,
                                    #TODO: Make cutout size a tunable parameter and output the choice in the results
                                    cutout_size=(15,15),
                                    debug=True,
                                    fit_radius_px=FIT_RADIUS_PX,
                                    use_ultranest=globals().get("USE_ULTRANEST", True),
                                    show_figs=False,
                                    save_figs=SAVE_FIGS,
                                    figs_dir=str(save_dir) if SAVE_FIGS else None,
                                    max_pix_offset=globals().get("MAX_PIX_OFFSET", None),
                                    ap_radius_px=globals().get("AP_RADIUS_PX", None),
                                    no_masking=bool(globals().get("NO_MASKING", False)),
                                )
                            finally:
                                sys.stdout, sys.stderr = old_out, old_err
                    except Exception as _inproc_e:
                        # Mirror the behavior of the CLI branch: ensure we have a log file and raise to outer handler.
                        try:
                            with open(analyzer_stdout_path, "a") as _logfh:
                                _logfh.write("\n=== EXCEPTION (in-process analyze_file) ===\n")
                                _logfh.write(str(_inproc_e) + "\n")
                                _logfh.flush()
                        except Exception:
                            pass
                        raise
                    # Read back the collected stdout for parity with CLI path
                    try:
                        with open(analyzer_stdout_path, "r") as _rfh:
                            raw_text = _rfh.read()
                    except Exception:
                        raw_text = None
                    # Determine status/comment for in-process path; treat presence/quality of res as success flag
                    ok_inproc = isinstance(res, dict)
                    last_status, last_comment = _derive_status_and_comment(ok_inproc, res, raw_text, None)
                else:
                    print("[batch] Launching analyzer via CLI:")
                    print("         " + " ".join([
                        sys.executable, "-u", "-m", ANALYZER_MODULE,
                        "--fits", str(fits_path),
                        "--ra", str(adj_ra_deg),
                        "--dec", str(adj_dec_deg),
                        "--fit-radius", str(FIT_RADIUS_PX),
                        "--no-show",
                        "--figs-dir", str(save_dir) if SAVE_FIGS else str(OUTDIR),
                        "--debug",
                        "--save-results",
                        "--results-path", str(save_dir / "analyzer_results.json"),
                    ] + (
                        ["--save-figs"] if SAVE_FIGS else []
                    ) + (
                        ["--scipy-only"] if not globals().get("USE_ULTRANEST", True) else []
                    ) + (
                        ["--max-pix-offset", str(globals()["MAX_PIX_OFFSET"])] if globals().get("MAX_PIX_OFFSET") is not None else []
                    ) + (
                        ["--aperture-px", str(globals()["AP_RADIUS_PX"])] if globals().get("AP_RADIUS_PX") is not None else []
                    ) + (
                        ["--no-masking"] if bool(globals().get("NO_MASKING", False)) else []
                    )
                    ))
                    
                    out = run_analyzer_via_cli(
                        str(fits_path), adj_ra_deg, adj_dec_deg,
                        fit_radius_px=FIT_RADIUS_PX,
                        quiet=ULTRANEST_QUIET,
                        save_dir=str(save_dir) if SAVE_FIGS else str(OUTDIR),
                        stdout_path=str(analyzer_stdout_path),  # tee live output here
                        results_path=str(save_dir / "analyzer_results.json"),
                    )
                    res = out.get("result")
                    raw_text = out.get("raw_text")
                    if out.get("results_path"):
                        print(f"   -> analyzer results loaded from {out['results_path']}")
                    ok_cli = bool(out.get("ok", False))
                    err_cli = out.get("error")
                    last_status, last_comment = _derive_status_and_comment(ok_cli, res, raw_text, err_cli)

                # Adaptive retry for S3 cutout mode:
                # If we hit the masked/background failure class with a small cutout,
                # re-download a larger cutout and retry analyzer once.
                if (
                    last_status == "error_masked_pixels"
                    and str(globals().get("DOWNLOADER", "")).lower() == "s3"
                    and bool(globals().get("S3_CUTOUT", True))
                    and bool(globals().get("S3_CUTOUT_EXPAND_ON_MASKED", True))
                ):
                    cur_cutout_size = int(globals().get("S3_CUTOUT_SIZE_PX", DEFAULT_S3_CUTOUT_SIZE_PX))
                    retry_cutout_size = int(globals().get("S3_CUTOUT_RETRY_SIZE_PX", 64))
                    retry_cutout_size = max(retry_cutout_size, 60, cur_cutout_size)
                    if retry_cutout_size > cur_cutout_size:
                        print(
                            f"[batch] Adaptive retry: masked/background failure with cutout={cur_cutout_size}px; "
                            f"re-downloading with cutout={retry_cutout_size}px and retrying analyzer once."
                        )
                        old_cutout_size = cur_cutout_size
                        try:
                            globals()["S3_CUTOUT_SIZE_PX"] = retry_cutout_size
                            globals()["S3_CUTOUT_CENTER_RA_DEG"] = float(reference_ra_deg)
                            globals()["S3_CUTOUT_CENTER_DEC_DEG"] = float(reference_dec_deg)
                            local_retry = _download_file_with_retries(
                                str(fits_url),
                                max_retries=int(globals().get("DOWNLOAD_MAX_RETRIES", DOWNLOAD_MAX_RETRIES)),
                                timeout=10000,
                            )
                            fits_path = pathlib.Path(local_retry)
                            print(f"[batch] Adaptive retry FITS ready: {fits_path}")

                            if analyze_file:
                                class Tee:
                                    def __init__(self, *targets):
                                        self.targets = targets
                                    def write(self, data):
                                        for t in self.targets:
                                            try:
                                                t.write(data)
                                                t.flush()
                                            except Exception:
                                                pass
                                    def flush(self):
                                        for t in self.targets:
                                            try:
                                                t.flush()
                                            except Exception:
                                                pass

                                with open(analyzer_stdout_path, "a") as _logfh:
                                    _logfh.write(
                                        f"\n=== ADAPTIVE RETRY: re-run analyzer with enlarged S3 cutout ({retry_cutout_size}px) ===\n"
                                    )
                                    tee_out = Tee(sys.__stdout__, _logfh)
                                    tee_err = Tee(sys.__stderr__, _logfh)
                                    old_out, old_err = sys.stdout, sys.stderr
                                    sys.stdout, sys.stderr = tee_out, tee_err
                                    try:
                                        res = analyze_file(
                                            str(fits_path), adj_ra_deg, adj_dec_deg,
                                            cutout_size=(15, 15),
                                            debug=True,
                                            fit_radius_px=FIT_RADIUS_PX,
                                            use_ultranest=globals().get("USE_ULTRANEST", True),
                                            show_figs=False,
                                            save_figs=SAVE_FIGS,
                                            figs_dir=str(save_dir) if SAVE_FIGS else None,
                                            max_pix_offset=globals().get("MAX_PIX_OFFSET", None),
                                            ap_radius_px=globals().get("AP_RADIUS_PX", None),
                                            no_masking=bool(globals().get("NO_MASKING", False)),
                                        )
                                    finally:
                                        sys.stdout, sys.stderr = old_out, old_err
                                try:
                                    with open(analyzer_stdout_path, "r") as _rfh:
                                        raw_text = _rfh.read()
                                except Exception:
                                    raw_text = None
                                ok_inproc = isinstance(res, dict)
                                last_status, last_comment = _derive_status_and_comment(ok_inproc, res, raw_text, None)
                                print(f"[batch] Adaptive retry analyzer status: {last_status}")
                            else:
                                print(
                                    "[batch] Adaptive masked-pixel retry currently supports in-process analyzer mode only."
                                )
                        except Exception as _retry_e:
                            print(f"[batch] Adaptive retry failed: {_retry_e}")
                        finally:
                            globals()["S3_CUTOUT_SIZE_PX"] = old_cutout_size

                # If analyzer failed (CLI rc non-zero) or produced no result, write a diagnostic image so something is saved.
                try:
                    analyzer_failed = False
                    if not analyze_file:
                        analyzer_failed = not bool(out.get("ok", False))
                    else:
                        # Heuristics: no result dict, or result explicitly reports an error-like status/snr
                        analyzer_failed = (res is None)
                        if isinstance(res, dict):
                            status_key = res.get("status", "").lower() if isinstance(res.get("status", ""), str) else ""
                            analyzer_failed = analyzer_failed or ("error" in status_key)
                            # Consider empty/zero used-pixel counts as failure-like
                            n_used = res.get("n_pix_used_in_fit", None)
                            if n_used is not None:
                                try:
                                    analyzer_failed = analyzer_failed or (float(n_used) <= 0)
                                except Exception:
                                    pass
                    if analyzer_failed and fits_path is not None:
                        _save_failure_diagnostic(fits_path, save_dir)
                except Exception:
                    pass

                # If keeping FITS files, move to OUTDIR/fits with a friendly name
                if globals().get("KEEP_FITS", False):
                    try:
                        # Ensure destination exists
                        dest_dir = globals().get("FITS_DIR", OUTDIR / "fits")
                        try:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass

                        # Use the same stub used for figs; ensure unique filename
                        dest = dest_dir / f"{img_stub}.fits"
                        if dest.exists():
                            k = 1
                            while True:
                                alt = dest_dir / f"{img_stub}_{k}.fits"
                                if not alt.exists():
                                    dest = alt
                                    break
                                k += 1

                        shutil.move(str(fits_path), str(dest))
                        fits_path = dest
                        print(f"   -> kept FITS at {dest}")
                    except Exception as _e_keep:
                        print(f"   -> warning: could not move FITS into {dest_dir}: {_e_keep}")

                # ----- Build a single, strict row for CSV in the exact order -----
                ok = True
                err_msg = None
                if not analyze_file:
                    ok = bool(out.get("ok", False))
                    if not ok:
                        err_msg = out.get("error", f"Analyzer exited with code {out.get('rc')}")

                # capture run metadata/paths
                figs_dir_str = str(save_dir)
                fits_local_str = str(fits_path)
                analyzer_stdout_str = str(analyzer_stdout_path)

                # Reference parameters (currently equal to inputs per spec)
                reference_ra_deg = float(ra)
                reference_dec_deg = float(dec)
                reference_crd_epoch_yr = globals().get("REFERENCE_CRD_EPOCH_YR", None)
                reference_pmra_masyr   = globals().get("REFERENCE_PMRA_MASYR", None)
                reference_pmdec_masyr  = globals().get("REFERENCE_PMDEC_MASYR", None)

                # Helper to safely grab keys from analyzer result
                def g(key):
                    if res is None:
                        return np.nan
                    if isinstance(res, dict) and (key in res):
                        return res.get(key)
                    return np.nan

                row = {
                    # PARAMETERS FOR THE BATCH FITS DOWNLOADER
                    "target_name": globals().get("TARGET_NAME"),
                    "reference_ra_deg": reference_ra_deg,
                    "reference_dec_deg": reference_dec_deg,
                    "reference_crd_epoch_yr": reference_crd_epoch_yr,
                    "reference_pmra_masyr": reference_pmra_masyr,
                    "reference_pmdec_masyr": reference_pmdec_masyr,

                    # ANALYZER CODE PARAMETERS
                    "input_ra_deg": adj_ra_deg,
                    "input_dec_deg": adj_dec_deg,
                    "fit_radius_px": float(FIT_RADIUS_PX) if FIT_RADIUS_PX is not None else None,
                    "box_size_bcg_subtract": 15,
                    "no_masking": g("no_masking") if isinstance(res, dict) else bool(globals().get("NO_MASKING", False)),

                    # FITS-PARSER RELATED STUFF (wrapper)
                    "fits_path": fits_local_str,
                    "access_url": url,
                    "fits_analysis_status": last_status,
                    "fits_analysis_comment": last_comment,
                    "analyzer_stdout": analyzer_stdout_str,
                    "obs_collection": coll,

                    # FITS-HEADER RELATED STUFF (populate from header/knowns even on failure)
                    "obsid": obsid,
                    "bandpass": bandpass,
                    "expid": expidn,
                    "mjd_avg": obs_mjd,
                    "detector_id": (
                        None if bandpass is None
                        else (
                            int(str(bandpass).lstrip('D')) if str(bandpass).startswith('D') and str(bandpass)[1:].isdigit()
                            else hdr_detector
                        )
                    ),
                    "psf_index": g("psf_index"),
                    "omega_sr": g("omega_sr"),
                    "px_scale_arcsec": g("px_scale_arcsec"),

                    # FLAGS
                    "near_cutout_edge": g("near_cutout_edge"),
                    "near_detector_edge": g("near_detector_edge"),
                    "near_bcg_star": g("near_bcg_star"),
                    "n_pix_flagged_in_fit": g("n_pix_flagged_in_fit"),
                    "n_pix_used_in_fit": g("n_pix_used_in_fit"),
                    "n_pix_total_in_fit": g("n_pix_total_in_fit"),

                    # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
                    "ap_radius_px": g("ap_radius_px"),
                    "ap_radius_forced": g("ap_radius_forced"),
                    "ap_flux_MJysr": g("ap_flux_MJysr"),
                    "ap_flux_MJysr_err": g("ap_flux_MJysr_err"),
                    "ap_snr": g("ap_snr"),
                    "ap_centroid_err_px": g("ap_centroid_err_px"),
                    "ap_flux_uJy": g("ap_flux_uJy"),
                    "ap_flux_uJy_err": g("ap_flux_uJy_err"),
                    "ap_xcen_cutout": g("ap_xcen_cutout"),
                    "ap_ycen_cutout": g("ap_ycen_cutout"),
                    "ap_xcen_fullim": g("ap_xcen_fullim"),
                    "ap_ycen_fullim": g("ap_ycen_fullim"),

                    # CENTER OF MASS OUTPUTS
                    "com_xcen_cutout": g("com_xcen_cutout"),
                    "com_ycen_cutout": g("com_ycen_cutout"),
                    "com_xcen_fullim": g("com_xcen_fullim"),
                    "com_ycen_fullim": g("com_ycen_fullim"),
                    "com_ra_deg": g("com_ra_deg"),
                    "com_dec_deg": g("com_dec_deg"),
                    "com_sep_as": g("com_sep_as"),
                    "com_wv_um": g("com_wv_um"),
                    "com_wv_width_um": g("com_wv_width_um"),

                    # PSF-SCIPY OUTPUTS
                    "psf_scipy_method_used": g("psf_scipy_method_used"),
                    "psf_scipy_status": g("psf_scipy_status"),
                    "psf_scipy_flux_MJysr": g("psf_scipy_flux_MJysr"),
                    "psf_scipy_flux_MJysr_err": g("psf_scipy_flux_MJysr_err"),
                    "psf_scipy_snr": g("psf_scipy_snr"),
                    "psf_scipy_flux_uJy": g("psf_scipy_flux_uJy"),
                    "psf_scipy_flux_uJy_err": g("psf_scipy_flux_uJy_err"),
                    "psf_scipy_dx": g("psf_scipy_dx"),
                    "psf_scipy_dy": g("psf_scipy_dy"),
                    "psf_scipy_xcen_cutout": g("psf_scipy_xcen_cutout"),
                    "psf_scipy_ycen_cutout": g("psf_scipy_ycen_cutout"),
                    "psf_scipy_xcen_fullim": g("psf_scipy_xcen_fullim"),
                    "psf_scipy_ycen_fullim": g("psf_scipy_ycen_fullim"),
                    "psf_scipy_ra_deg": g("psf_scipy_ra_deg"),
                    "psf_scipy_ra_err_mas": g("psf_scipy_ra_err_mas"),
                    "psf_scipy_dec_deg": g("psf_scipy_dec_deg"),
                    "psf_scipy_dec_err_mas": g("psf_scipy_dec_err_mas"),
                    "psf_scipy_ra_dec_cov_mas2": g("psf_scipy_ra_dec_cov_mas2"),
                    "psf_scipy_sep_as": g("psf_scipy_sep_as"),
                    "psf_scipy_wv_um": g("psf_scipy_wv_um"),
                    "psf_scipy_wv_um_err": g("psf_scipy_wv_um_err"),
                    "psf_scipy_wv_width_um": g("psf_scipy_wv_width_um"),
                    "psf_scipy_wv_width_um_err": g("psf_scipy_wv_width_um_err"),
                    "psf_scipy_chi2": g("psf_scipy_chi2"),
                    "psf_scipy_dof": g("psf_scipy_dof"),

                    # PSF-ultranest outputs
                    "psf_un_flux_MJysr": g("psf_un_flux_MJysr"),
                    "psf_un_flux_MJysr_err": g("psf_un_flux_MJysr_err"),
                    "psf_un_snr": g("psf_un_snr"),
                    "psf_un_flux_uJy": g("psf_un_flux_uJy"),
                    "psf_un_flux_uJy_err": g("psf_un_flux_uJy_err"),
                    "psf_un_dx": g("psf_un_dx"),
                    "psf_un_dy": g("psf_un_dy"),
                    "psf_un_xcen_cutout": g("psf_un_xcen_cutout"),
                    "psf_un_ycen_cutout": g("psf_un_ycen_cutout"),
                    "psf_un_xcen_fullim": g("psf_un_xcen_fullim"),
                    "psf_un_ycen_fullim": g("psf_un_ycen_fullim"),
                    "psf_un_xcen_err": g("psf_un_xcen_err"),
                    "psf_un_ycen_err": g("psf_un_ycen_err"),
                    "psf_un_ra_deg": g("psf_un_ra_deg"),
                    "psf_un_ra_err_mas": g("psf_un_ra_err_mas"),
                    "psf_un_dec_deg": g("psf_un_dec_deg"),
                    "psf_un_dec_err_mas": g("psf_un_dec_err_mas"),
                    "psf_un_sep_as": g("psf_un_sep_as"),
                    "psf_un_ra_dec_cov_mas2": g("psf_un_ra_dec_cov_mas2"),
                    "psf_un_wv_um": g("psf_un_wv_um"),
                    "psf_un_wv_um_err": g("psf_un_wv_um_err"),
                    "psf_un_wv_width_um": g("psf_un_wv_width_um"),
                    "psf_un_wv_width_um_err": g("psf_un_wv_width_um_err"),
                    "psf_un_chi2": g("psf_un_chi2"),
                    "psf_un_dof": g("psf_un_dof"),
                }

                # Only keep the exact schema in the exact order
                row = {k: row.get(k) for k in CSV_FIELDS}
                rows_for_csv.append(row)

                # Append to master CSV incrementally (header only on first write)
                try:
                    header_needed = not RESULTS_CSV.exists()
                    pd.DataFrame([row])[CSV_FIELDS].to_csv(
                        RESULTS_CSV, index=False, mode="a", header=header_needed
                    )
                    print(f"   -> appended to master CSV: {RESULTS_CSV.name}")
                except Exception as e:
                    print(f"   -> could not append to master CSV: {e}")

                # Per-image one-line CSV (written next to figs & stdout)
                try:
                    per_csv = save_dir / "result.csv"
                    pd.DataFrame([row])[CSV_FIELDS].to_csv(per_csv, index=False)
                    print(f"   -> wrote per-image CSV: {per_csv}")
                except Exception as e:
                    print(f"   -> could not write per-image CSV: {e}")
                # Persist the row and stream the same machine-readable event for
                # live wrapper-side upsert.
                try:
                    row_json = _write_row_jsonl(jsonl, row)
                    print("[LV2_ROW_JSON] " + row_json)
                except Exception as e:
                    print(f"   -> could not write/emit LV2_ROW_JSON: {e}")

            except KeyboardInterrupt:
                print("\n[batch] Interrupted by user, writing partial results …")
                break
            except Exception as e:
                tb_text = traceback.format_exc()
                # Print a clear header then the full traceback so lines/files are visible
                print("   -> analysis failed: " + f"{e.__class__.__name__}: {e}")
                for _line in tb_text.rstrip().splitlines():
                    print("      " + _line)
                # Try to write diagnostic image so PNG is present even on hard exceptions.
                try:
                    if fits_path is not None:
                        _save_failure_diagnostic(fits_path, save_dir)
                except Exception:
                    pass
                # Also append the traceback to the analyzer stdout log for this image
                try:
                    with open(analyzer_stdout_path, "a") as _fh:
                        _fh.write("\n=== EXCEPTION TRACEBACK ===\n")
                        _fh.write(tb_text)
                except Exception:
                    pass
                try:
                    last_status = "error_exception"
                    last_comment = f"{e.__class__.__name__}: {str(e)}"[:300]
                except Exception:
                    pass
            finally:
                # clean the temporary FITS downloaded via resolved DataLink
                try:
                    if (fits_path is not None) and isinstance(fits_path, pathlib.Path) and fits_path.exists() and not globals().get("KEEP_FITS", False):
                        # astropy download_file may create a non-named temp path; remove safely
                        os.remove(fits_path)
                except Exception:
                    pass
                
                if row is None and not write_failure_placeholder:
                    try:
                        print("   -> transient failure detected; skipping failure placeholder row (will retry on a future run)")
                    except Exception:
                        pass

                # Register empty rows if analysis failed
                if row is None and write_failure_placeholder:
                    # Ensure a diagnostic image and stdout log exist on failure.
                    try:
                        if fits_path is not None:
                            _save_failure_diagnostic(fits_path, save_dir)
                    except Exception:
                        pass
                    try:
                        # Touch the stdout file if it doesn't exist so CSV points to a real path
                        if not pathlib.Path(analyzer_stdout_path).exists():
                            with open(analyzer_stdout_path, "w") as _touch:
                                _touch.write("No analyzer output captured; failure placeholder created by batch wrapper.\n")
                                try:
                                    _touch.write(f"Known wrapper status: {last_status}\n")
                                    if last_comment:
                                        _touch.write(f"Known wrapper comment: {last_comment}\n")
                                except Exception:
                                    pass
                        else:
                            # Even if the file exists, make sure the placeholder header and known status/comment are present.
                            _append_log(analyzer_stdout_path, "No analyzer output captured; failure placeholder created by batch wrapper.")
                            _append_log(analyzer_stdout_path, f"Known wrapper status: {last_status}")
                            if last_comment:
                                _append_log(analyzer_stdout_path, f"Known wrapper comment: {last_comment}")
                    except Exception:
                        pass
                    # ensure string forms exist for paths
                    analyzer_stdout_str = str(analyzer_stdout_path)
                    fits_local_str = str(fits_path) if isinstance(fits_path, pathlib.Path) else ""
                    row = {
                        # PARAMETERS FOR THE BATCH FITS DOWNLOADER
                        "target_name": globals().get("TARGET_NAME"),
                        "reference_ra_deg": reference_ra_deg,
                        "reference_dec_deg": reference_dec_deg,
                        "reference_crd_epoch_yr": reference_crd_epoch_yr,
                        "reference_pmra_masyr": reference_pmra_masyr,
                        "reference_pmdec_masyr": reference_pmdec_masyr,

                        # ANALYZER CODE PARAMETERS
                        "input_ra_deg": adj_ra_deg,
                        "input_dec_deg": adj_dec_deg,
                        "fit_radius_px": float(FIT_RADIUS_PX) if FIT_RADIUS_PX is not None else None,
                        "box_size_bcg_subtract": 15,
                        "no_masking": bool(globals().get("NO_MASKING", False)),

                        # FITS-PARSER RELATED STUFF (wrapper)
                        "fits_path": fits_local_str,
                        "access_url": url,
                        "fits_analysis_status": last_status,
                        "fits_analysis_comment": last_comment,
                        "analyzer_stdout": analyzer_stdout_str,
                        "obs_collection": coll,

                        # FITS-HEADER RELATED STUFF (populate from header/knowns)
                        "obsid": obsid,
                        "bandpass": bandpass,
                        "expid": expidn,
                        "mjd_avg": obs_mjd,
                        "detector_id": (
                            None if bandpass is None
                            else (
                                int(str(bandpass).lstrip('D')) if str(bandpass).startswith('D') and str(bandpass)[1:].isdigit()
                                else hdr_detector
                            )
                        ),
                        "psf_index": None,
                        "omega_sr": None,
                        "px_scale_arcsec": None,

                        # FLAGS
                        "near_cutout_edge": None,
                        "near_detector_edge": None,
                        "near_bcg_star": None,
                        "n_pix_flagged_in_fit": None,
                        "n_pix_used_in_fit": None,
                        "n_pix_total_in_fit": None,

                        # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
                        "ap_radius_px": None,
                        "ap_radius_forced": None,
                        "ap_flux_MJysr": None,
                        "ap_flux_MJysr_err": None,
                        "ap_snr": None,
                        "ap_centroid_err_px": None,
                        "ap_flux_uJy": None,
                        "ap_flux_uJy_err": None,
                        "ap_xcen_cutout": None,
                        "ap_ycen_cutout": None,
                        "ap_xcen_fullim": None,
                        "ap_ycen_fullim": None,

                        # CENTER OF MASS OUTPUTS
                        "com_xcen_cutout": None,
                        "com_ycen_cutout": None,
                        "com_xcen_fullim": None,
                        "com_ycen_fullim": None,
                        "com_ra_deg": None,
                        "com_dec_deg": None,
                        "com_sep_as": None,
                        "com_wv_um": None,
                        "com_wv_width_um": None,

                        # PSF-SCIPY OUTPUTS
                        "psf_scipy_method_used": None,
                        "psf_scipy_status": None,
                        "psf_scipy_flux_MJysr": None,
                        "psf_scipy_flux_MJysr_err": None,
                        "psf_scipy_snr": None,
                        "psf_scipy_flux_uJy": None,
                        "psf_scipy_flux_uJy_err": None,
                        "psf_scipy_dx": None,
                        "psf_scipy_dy": None,
                        "psf_scipy_xcen_cutout": None,
                        "psf_scipy_ycen_cutout": None,
                        "psf_scipy_xcen_fullim": None,
                        "psf_scipy_ycen_fullim": None,
                        "psf_scipy_ra_deg": None,
                        "psf_scipy_ra_err_mas": None,
                        "psf_scipy_dec_deg": None,
                        "psf_scipy_dec_err_mas": None,
                        "psf_scipy_ra_dec_cov_mas2": None,
                        "psf_scipy_sep_as": None,
                        "psf_scipy_wv_um": None,
                        "psf_scipy_wv_um_err": None,
                        "psf_scipy_wv_width_um": None,
                        "psf_scipy_wv_width_um_err": None,
                        "psf_scipy_chi2": None,
                        "psf_scipy_dof": None,

                        # PSF-ultranest outputs
                        "psf_un_flux_MJysr": None,
                        "psf_un_flux_MJysr_err": None,
                        "psf_un_snr": None,
                        "psf_un_flux_uJy": None,
                        "psf_un_flux_uJy_err": None,
                        "psf_un_dx": None,
                        "psf_un_dy": None,
                        "psf_un_xcen_cutout": None,
                        "psf_un_ycen_cutout": None,
                        "psf_un_xcen_fullim": None,
                        "psf_un_ycen_fullim": None,
                        "psf_un_xcen_err": None,
                        "psf_un_ycen_err": None,
                        "psf_un_ra_deg": None,
                        "psf_un_ra_err_mas": None,
                        "psf_un_dec_deg": None,
                        "psf_un_dec_err_mas": None,
                        "psf_un_sep_as": None,
                        "psf_un_ra_dec_cov_mas2": None,
                        "psf_un_wv_um": None,
                        "psf_un_wv_um_err": None,
                        "psf_un_wv_width_um": None,
                        "psf_un_wv_width_um_err": None,
                        "psf_un_chi2": None,
                        "psf_un_dof": None,
                    }

                    # Keep exact schema/order and append
                    row = {k: row.get(k) for k in CSV_FIELDS}
                    rows_for_csv.append(row)

                    # Append to master CSV incrementally (header only on first write)
                    try:
                        header_needed = not RESULTS_CSV.exists()
                        pd.DataFrame([row])[CSV_FIELDS].to_csv(
                            RESULTS_CSV, index=False, mode="a", header=header_needed
                        )
                        print(f"   -> appended to master CSV: {RESULTS_CSV.name} (failure placeholder)")
                    except Exception as e:
                        print(f"   -> could not append to master CSV (failure placeholder): {e}")

                    # Per-image one-line CSV (written next to figs & stdout)
                    try:
                        per_csv = save_dir / "result.csv"
                        pd.DataFrame([row])[CSV_FIELDS].to_csv(per_csv, index=False)
                        print(f"   -> wrote per-image CSV: {per_csv}")
                    except Exception as e:
                        print(f"   -> could not write per-image CSV (failure placeholder): {e}")
                    # Persist the failure row and stream the same machine-readable
                    # event for live wrapper-side upsert.
                    try:
                        row_json = _write_row_jsonl(jsonl, row)
                        print("[LV2_ROW_JSON] " + row_json)
                    except Exception as e:
                        print(f"   -> could not write/emit LV2_ROW_JSON (failure placeholder): {e}")

    jsonl.close()

    # Update global counters for the __main__ footer / outer wrappers.
    globals()["_LV2_TRANSIENT_DOWNLOAD_ERRORS"] = transient_download_errors
    globals()["_LV2_TRANSIENT_TAP_ERRORS"] = transient_tap_errors
    globals()["_LV2_TRANSIENT_ERRORS"] = transient_download_errors + transient_tap_errors
    if last_transient_error:
        globals()["_LV2_LAST_TRANSIENT_ERROR"] = last_transient_error

    # write CSV
    if rows_for_csv:
        df_out = pd.DataFrame(rows_for_csv)
        df_out = df_out.reindex(columns=CSV_FIELDS)
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"[batch] Wrote {len(df_out)} rows to {RESULTS_CSV}")
        print(f"[batch] Wrote JSONL with covariances to {RESULTS_JSONL}")
        if bool(globals().get("WRITE_BINNED_SPECTRUM", True)):
            try:
                binned_out = write_binned_spherex_spectrum_csv(
                    df_out,
                    globals().get("BINNED_SPECTRUM_CSV", OUTDIR / "binned_spectrum.csv"),
                )
                print(f"[batch] Wrote binned SPHEREx spectrum to {binned_out}")
            except Exception as exc:
                print(f"[batch] warning: could not write binned SPHEREx spectrum: {exc}")
        #import pdb; pdb.set_trace()
        if skipped_count > 0:
            print(f"[batch] Skipped {skipped_count} candidate(s) already processed.")
        final_count = len(df_out)
    else:
        print("[batch] No rows to write.")
        final_count = 0
        if download_only:
            print(f"[batch] Download-only mode saved {int(download_only_count)} FITS file(s).")

        # If we encountered only transient connectivity/proxy errors (and therefore purposely
        # skipped failure placeholders), tell higher-level tooling to skip follow-up work.
        try:
            n_transient = int(globals().get("_LV2_TRANSIENT_ERRORS", 0) or 0)
        except Exception:
            n_transient = 0
        if n_transient > 0:
            globals()["_LV2_TRANSIENT_ONLY"] = True
            last_te = globals().get("_LV2_LAST_TRANSIENT_ERROR")
            print(f"[LV2_TRANSIENT_ERRORS] {n_transient}")
            if globals().get("_LV2_TRANSIENT_DOWNLOAD_ERRORS", 0):
                print(f"[LV2_TRANSIENT_DOWNLOAD_ERRORS] {int(globals().get('_LV2_TRANSIENT_DOWNLOAD_ERRORS', 0) or 0)}")
            if globals().get("_LV2_TRANSIENT_TAP_ERRORS", 0):
                print(f"[LV2_TRANSIENT_TAP_ERRORS] {int(globals().get('_LV2_TRANSIENT_TAP_ERRORS', 0) or 0)}")
            if last_te:
                print(f"[LV2_LAST_TRANSIENT_ERROR] {str(last_te)[:500]}")
            print("[LV2_SKIP_FOLLOWUP] transient_connectivity")

        if total_candidates > 0 and skipped_count == total_candidates:
            print(f"[batch] All {skipped_count} candidates were in the skip list; wrote no rows.")
            # Machine-parseable tag for the outer driver
            print(f"[LV2_ALL_SKIPPED] {skipped_count}")
        elif skipped_count > 0:
            print(f"[batch] Skipped {skipped_count} candidate(s) already processed; remaining produced no rows.")
    if download_only:
        print(f"[LV2_DOWNLOADED_COUNT] {int(download_only_count)}")
    return final_count

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download SPHEREx spectral images near a target and run forced photometry (PSF + model-free).")
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--target-name", type=str, default=None,
                   help="Human-friendly target identifier; also used to name the output directory when provided.")
    p.add_argument("--reference-crd-epoch-yr", type=float, required=True,
                help="Reference coordinate epoch (Julian years).")
    p.add_argument("--reference-pmra-masyr", type=float, required=True,
                help="Reference proper motion RA* (mas/yr). Required; use 0.0 for stationary targets.")
    p.add_argument("--reference-pmdec-masyr", type=float, required=True,
                help="Reference proper motion Dec (mas/yr). Required; use 0.0 for stationary targets.")
    p.add_argument("--radius-arcmin", type=float, default=SEARCH_RADIUS_ARCMIN)
    p.add_argument("--fit-radius-px", type=float, default=FIT_RADIUS_PX)
    p.add_argument("--scipy-only", action="store_true", default=False,
                   help="Skip UltraNest; use SciPy-only fitting in the analyzer.")
    p.add_argument("--quiet", action="store_true", default=ULTRANEST_QUIET)
    p.add_argument("--save-figs", action="store_true", default=SAVE_FIGS)
    p.add_argument("--no-figures", action="store_true", default=False,
                   help="Do not write any figures to disk (overrides --save-figs).")
    p.add_argument("--outdir", default=str(OUTDIR))
    p.add_argument(
        "--download-only",
        action="store_true",
        default=False,
        help="Discover/resolve/download FITS cutouts into OUTDIR/fits and write a local download manifest, but do not run the analyzer.",
    )
    p.add_argument(
        "--consume-downloaded",
        action="store_true",
        default=False,
        help="Skip discovery/download and analyze predownloaded local FITS from OUTDIR/fits or download_manifest.csv.",
    )
    p.add_argument("--collections", type=str, default=",".join(CATALOGS),
                   help="Comma-separated list of obs_collection prefixes (e.g., 'spherex_qr,spherex_qr_deep').")
    p.add_argument(
        "--use-sia2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use IRSA SIA2 discovery (single-step URLs) with TAP fallback on failure (default: on).",
    )
    p.add_argument(
        "--include-qr2-deep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also include 'spherex_qr2_deep' in collections filtering (default: on).",
    )
    p.add_argument("--max-rows", type=int, default=None,
                   help="Max number of images to download/process (useful for quick tests).")
    p.add_argument("--skip-obsidbps", type=str, default="",
               help="Comma-separated list of concatenated OBSID and Bandpass (eg D6) strings to skip (already processed).")
    p.add_argument(
        "--skip-obsidbps-file",
        type=str,
        default="",
        help="Path to text file containing OBSID+bandpass keys to skip (comma and/or newline separated).",
    )
    p.add_argument("--min-datapoints", type=int, default=None,
               help="Minimum number of IRSA query results required to process target.")
    p.add_argument(
        "--keep-fits",
        action="store_true",
        default=False,
        help="Keep downloaded Level 2 FITS images instead of deleting them."
    )
    p.add_argument(
        "--max-pix-offset",
        type=float,
        default=None,
        help="Max absolute pixel offset for dx and dy (pixels). If 0, dx and dy are fixed to 0."
    )
    p.add_argument(
        "--ap-radius-px",
        type=float,
        default=None,
        help="Override aperture radius in pixels for photometry (default: analyzer's own setting)."
    )
    p.add_argument("--query-epoch-decimal-year", type=float, default=QUERY_EPOCH_DECIMAL_YEAR,
                   help="Decimal year epoch to which the reference RA,Dec are projected BEFORE querying ObsCore (default: 2025.5).")
    p.add_argument(
        "--no-masking",
        action="store_true",
        default=False,
        help="Do not mask any pixels during the PSF fitting step in the analyzer (ignores flag-based masking). Does not affect BCG/background-subtraction masking.",
    )
    p.add_argument(
        "--downloader",
        type=str,
        default="s3",
        choices=("wget", "curl", "astropy", "s3"),
        help="Downloader backend for FITS files (default: s3).",
    )
    p.add_argument(
        "--s3-bucket",
        type=str,
        default="nasa-irsa-spherex",
        help="S3 bucket for --downloader s3 (default: nasa-irsa-spherex).",
    )
    p.add_argument(
        "--s3-cutout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using --downloader s3, fetch a local cutout FITS around the target (default: on).",
    )
    p.add_argument(
        "--s3-cutout-size-px",
        type=int,
        default=DEFAULT_S3_CUTOUT_SIZE_PX,
        help="Cutout width/height in detector pixels for --downloader s3 (default: 20).",
    )
    p.add_argument(
        "--s3-cutout-expand-on-masked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If analyzer reports masked-pixels/background failure, re-download a larger S3 cutout and retry once (default: on).",
    )
    p.add_argument(
        "--s3-cutout-retry-size-px",
        type=int,
        default=64,
        help="Larger cutout width/height in pixels used for one masked-pixels retry (default: 64).",
    )
    p.add_argument(
        "--curl-resume",
        action="store_true",
        default=False,
        help="Enable resumable curl downloads via byte-range requests (default: off).",
    )
    p.add_argument(
        "--datalink-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resolve DataLink URLs in server-side ID batches before per-image processing (default: off; IRSA may allow only one ID per request).",
    )
    p.add_argument(
        "--datalink-batch-max-ids",
        type=int,
        default=5000,
        help="Maximum IDs per DataLink batch request (default: 5000).",
    )
    p.add_argument(
        "--datalink-jitter-min-sec",
        type=float,
        default=0.2,
        help="Minimum randomized sleep before each DataLink resolve (seconds; default: 0.2).",
    )
    p.add_argument(
        "--datalink-jitter-max-sec",
        type=float,
        default=1.0,
        help="Maximum randomized sleep before each DataLink resolve (seconds; default: 1.0).",
    )
    p.add_argument(
        "--write-binned-spectrum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write OUTDIR/binned_spectrum.csv at the end using the SQL-equivalent SPHEREx nearest-bin weighted spectrum method (default: on).",
    )
    p.add_argument(
        "--binned-spectrum-path",
        type=str,
        default=None,
        help="Output path for --write-binned-spectrum (default: OUTDIR/binned_spectrum.csv).",
    )
    return p


def cli_main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.download_only and args.consume_downloaded:
        raise SystemExit("Error: --download-only and --consume-downloaded are mutually exclusive.")
    if args.no_figures:
        args.save_figs = False

    # update globals from args (simplifies downstream)
    globals()["SEARCH_RADIUS_ARCMIN"] = args.radius_arcmin
    globals()["FIT_RADIUS_PX"] = args.fit_radius_px
    globals()["ULTRANEST_QUIET"] = args.quiet
    globals()["SAVE_FIGS"] = args.save_figs
    globals()["TARGET_NAME"] = args.target_name
    _default_dir = f"RA{args.ra:.6f}_DEC{args.dec:.6f}"
    if args.target_name:
        _safe = _safe_dirname(args.target_name) or "target"
        base_dirname = f"{_safe}_RA{args.ra:.6f}_DEC{args.dec:.6f}"
    else:
        base_dirname = _default_dir
    globals()["OUTDIR"] = (pathlib.Path(args.outdir) / base_dirname).resolve()
    globals()["RESULTS_CSV"] = OUTDIR / "results.csv"
    globals()["RESULTS_JSONL"] = OUTDIR / "results.jsonl"
    globals()["DOWNLOAD_MANIFEST_JSONL"] = OUTDIR / "download_manifest.jsonl"
    globals()["DOWNLOAD_MANIFEST_CSV"] = OUTDIR / "download_manifest.csv"
    globals()["BINNED_SPECTRUM_CSV"] = (
        pathlib.Path(args.binned_spectrum_path).expanduser().resolve()
        if args.binned_spectrum_path
        else (OUTDIR / "binned_spectrum.csv")
    )
    globals()["FIGS_DIR"] = OUTDIR / "figs"
    globals()["FITS_DIR"] = OUTDIR / "fits"
    globals()["DOWNLOAD_ONLY"] = bool(args.download_only)
    globals()["CONSUME_DOWNLOADED"] = bool(args.consume_downloaded)
    _cats = [s.strip() for s in args.collections.split(",") if s.strip()]
    if args.include_qr2_deep and ("spherex_qr2_deep" not in _cats):
        _cats.append("spherex_qr2_deep")
    globals()["CATALOGS"] = tuple(_cats)
    globals()["USE_SIA2"] = bool(args.use_sia2)
    globals()["MAX_ROWS"] = args.max_rows
    globals()["USE_ULTRANEST"] = not args.scipy_only
    globals()["REFERENCE_CRD_EPOCH_YR"] = args.reference_crd_epoch_yr
    globals()["REFERENCE_PMRA_MASYR"]  = args.reference_pmra_masyr
    globals()["REFERENCE_PMDEC_MASYR"] = args.reference_pmdec_masyr
    globals()["MIN_DATAPOINTS"] = args.min_datapoints
    globals()["QUERY_EPOCH_DECIMAL_YEAR"] = args.query_epoch_decimal_year
    _skip_raw = (args.skip_obsidbps or "").strip()
    _skip_set = set([s for s in _skip_raw.split(",") if s.strip()])
    _skip_file = (args.skip_obsidbps_file or "").strip()
    if _skip_file:
        try:
            _txt = pathlib.Path(_skip_file).read_text()
            for _sep in (",", "\n", "\r", "\t"):
                _txt = _txt.replace(_sep, " ")
            _parts = _txt.split()
            _skip_set |= {s for s in _parts if s}
        except Exception as e:
            print(f"[batch] warning: failed to read --skip-obsidbps-file '{_skip_file}': {e}")
    globals()["SKIP_OBSIDBPS"] = _skip_set
    globals()["KEEP_FITS"] = args.keep_fits
    if bool(args.download_only):
        globals()["KEEP_FITS"] = True
    globals()["MAX_PIX_OFFSET"] = args.max_pix_offset
    globals()["AP_RADIUS_PX"] = args.ap_radius_px
    globals()["NO_MASKING"] = args.no_masking
    globals()["DOWNLOADER"] = args.downloader
    globals()["CURL_RESUME"] = args.curl_resume
    globals()["S3_BUCKET"] = args.s3_bucket
    globals()["S3_CUTOUT"] = bool(args.s3_cutout)
    globals()["S3_CUTOUT_SIZE_PX"] = int(args.s3_cutout_size_px)
    globals()["S3_CUTOUT_EXPAND_ON_MASKED"] = bool(args.s3_cutout_expand_on_masked)
    globals()["S3_CUTOUT_RETRY_SIZE_PX"] = int(args.s3_cutout_retry_size_px)
    globals()["DATALINK_BATCH"] = bool(args.datalink_batch)
    globals()["DATALINK_BATCH_MAX_IDS"] = max(1, int(args.datalink_batch_max_ids))
    globals()["DATALINK_JITTER_MIN_SEC"] = float(args.datalink_jitter_min_sec)
    globals()["DATALINK_JITTER_MAX_SEC"] = float(args.datalink_jitter_max_sec)
    globals()["WRITE_BINNED_SPECTRUM"] = bool(args.write_binned_spectrum)

    exit_code = 0
    try:
        _count = main(args.ra, args.dec)
    except SystemExit as _se:
        # Allow explicit exit codes to propagate (we still print signature lines below).
        try:
            exit_code = int(_se.code) if _se.code is not None else 0
        except Exception:
            exit_code = 1
        _count = 0
    except Exception as e:
        # Ensure failures still produce a signed count line
        try:
            print(f"[lv2-error] {e}")
        except Exception:
            pass
        _count = 0
        exit_code = 1
    finally:
        # If this run was transient-only, prefer a TEMPFAIL-like exit code so higher-level tooling can skip follow-up work.
        try:
            if bool(globals().get("_LV2_TRANSIENT_ONLY", False)) and int(globals().get("_LV2_TRANSIENT_ERRORS", 0) or 0) > 0 and int(_count) == 0:
                # 75 is EX_TEMPFAIL (sysexits) on many systems.
                exit_code = 75
                print(f"[LV2_TRANSIENT_ERRORS] {int(globals().get('_LV2_TRANSIENT_ERRORS', 0) or 0)}")
                if globals().get("_LV2_LAST_TRANSIENT_ERROR"):
                    print(f"[LV2_LAST_TRANSIENT_ERROR] {str(globals().get('_LV2_LAST_TRANSIENT_ERROR'))[:500]}")
                print("[LV2_SKIP_FOLLOWUP] transient_connectivity")
        except Exception:
            pass

        # Verbose count line kept for human-readable logs.
        try:
            print(f"[batch] Total FITS analyzed (non-skipped, including failures): {_count}")
        except Exception:
            pass
        # New, succinct, machine-parseable signature line:
        # Machine-parseable summary tag for downstream tools.
        try:
            print(f"[LV2_ANALYZED_COUNT] {_count}")
        except Exception:
            pass

        return exit_code


if __name__ == "__main__":
    raise SystemExit(cli_main())
