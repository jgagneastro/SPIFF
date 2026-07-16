import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import shift
import time
from scipy.optimize import minimize
from math import pi
import contextlib
import io
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage
import json as _json

from typing import Optional, Any

from .spherex_psf_selection import select_spherex_psf

try:
    import ultranest
except ImportError:  # pragma: no cover - exercised in environments without ultranest
    ultranest = None

@dataclass
class FitOutputs:
    # CODE PARAMETERS
    input_ra_deg: float
    input_dec_deg: float
    fit_radius_px: Optional[float]
    box_size_bcg_subtract: int
    no_masking: bool

    # FITS-HEADER RELATED STUFF
    obsid: Optional[str]
    bandpass: Optional[str]
    expid: Optional[int]
    mjd_avg: Optional[float]
    detector_id: Optional[int]
    psf_index: Optional[int]
    omega_sr: Optional[float]
    px_scale_arcsec: Optional[float]

    # FLAGS
    near_cutout_edge: bool
    near_detector_edge: bool
    near_bcg_star: bool
    n_pix_flagged_in_fit: int
    n_pix_used_in_fit: int
    n_pix_total_in_fit: int

    # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
    ap_radius_px: Optional[float]
    ap_radius_forced: Optional[bool]
    ap_flux_MJysr: Optional[float]
    ap_flux_MJysr_err: Optional[float]
    ap_snr: Optional[float]
    ap_centroid_err_px: Optional[float]
    ap_flux_uJy: Optional[float]
    ap_flux_uJy_err: Optional[float]
    ap_xcen_cutout: Optional[float]
    ap_ycen_cutout: Optional[float]
    ap_xcen_fullim: Optional[float]
    ap_ycen_fullim: Optional[float]

    # CENTER OF MASS OUTPUTS
    com_xcen_cutout: Optional[float]
    com_ycen_cutout: Optional[float]
    com_xcen_fullim: Optional[float]
    com_ycen_fullim: Optional[float]
    com_ra_deg: Optional[float]
    com_dec_deg: Optional[float]
    com_sep_as: Optional[float]
    com_wv_um: Optional[float]
    com_wv_width_um: Optional[float]

    # PSF-SCIPY OUTPUTS
    psf_scipy_method_used: Optional[str]
    psf_scipy_status: Optional[str]
    psf_scipy_flux_MJysr: Optional[float]
    psf_scipy_flux_MJysr_err: Optional[float]
    psf_scipy_snr: Optional[float]
    psf_scipy_flux_uJy: Optional[float]
    psf_scipy_flux_uJy_err: Optional[float]
    psf_scipy_dx: Optional[float]
    psf_scipy_dy: Optional[float]
    psf_scipy_xcen_cutout: Optional[float]
    psf_scipy_ycen_cutout: Optional[float]
    psf_scipy_xcen_fullim: Optional[float]
    psf_scipy_ycen_fullim: Optional[float]
    psf_scipy_ra_deg: Optional[float]
    psf_scipy_ra_err_mas: Optional[float]
    psf_scipy_dec_deg: Optional[float]
    psf_scipy_dec_err_mas: Optional[float]
    psf_scipy_ra_dec_cov_mas2: Optional[Any]
    psf_scipy_sep_as: Optional[float]
    psf_scipy_wv_um: Optional[float]
    psf_scipy_wv_um_err: Optional[float]
    psf_scipy_wv_width_um: Optional[float]
    psf_scipy_wv_width_um_err: Optional[float]
    psf_scipy_chi2: Optional[float]
    psf_scipy_dof: Optional[int]

    # PSF-ULTRANEST OUTPUTS
    psf_un_flux_MJysr: Optional[float]
    psf_un_flux_MJysr_err: Optional[float]
    psf_un_snr: Optional[float]
    psf_un_flux_uJy: Optional[float]
    psf_un_flux_uJy_err: Optional[float]
    psf_un_dx: Optional[float]
    psf_un_dy: Optional[float]
    psf_un_xcen_cutout: Optional[float]
    psf_un_ycen_cutout: Optional[float]
    psf_un_xcen_fullim: Optional[float]
    psf_un_ycen_fullim: Optional[float]
    psf_un_xcen_err: Optional[float]
    psf_un_ycen_err: Optional[float]
    psf_un_ra_deg: Optional[float]
    psf_un_ra_err_mas: Optional[float]
    psf_un_dec_deg: Optional[float]
    psf_un_dec_err_mas: Optional[float]
    psf_un_sep_as: Optional[float]
    psf_un_ra_dec_cov_mas2: Optional[Any]
    psf_un_wv_um: Optional[float]
    psf_un_wv_um_err: Optional[float]
    psf_un_wv_width_um: Optional[float]
    psf_un_wv_width_um_err: Optional[float]
    psf_un_chi2: Optional[float]
    psf_un_dof: Optional[int]

def analyze_file(
    fits_path: str,
    ra: float,
    dec: float,
    psf_det_sum: float = None,
    cutout_size: Tuple[int, int] = (15, 15),
    fit_radius_px: float = 4.0,
    ap_radius_px: Optional[float] = 2.5,
    debug: bool = True,
    show_figs: bool = True,
    save_figs: bool = False,
    figs_dir: Optional[str] = None,
    ultranest_quiet: bool = True,
    #Increased from 200 to 500 on Sep 19 12:47PM but will need to avoid printing them
    posterior_keep: int = 500,     # how many posterior samples to keep in results (None to keep none)
    use_ultranest: bool = True,
    save_results: bool = False,
    results_path: Optional[str] = None,
    max_pix_offset: Optional[float] = None,
    no_masking: bool = False,
) -> Dict[str, Any]:
    """
    Run the single-image fit and return a JSON-serializable dict with all results.
    If show_figs is False and save_figs is True, figures are written to figs_dir.
    """
    box_size_bcg_subtract = 15
    if use_ultranest and ultranest is None:
        raise ImportError(
            "UltraNest is not installed. Re-run with --scipy-only or install the "
            "optional ultranest extra."
        )

    def _abort_with_nans(reason: str):
        dprint(f"[ERROR] Aborting analyze_file early: {reason}")
        # Header-derived values available at this point
        detector_id = int(hdr_img.get('DETECTOR')) if hdr_img.get('DETECTOR') is not None else np.nan
        bandpass = ("D" + str(detector_id)) if (detector_id is not None and detector_id is not np.nan) else np.nan
        expid = int(hdr_img.get('EXPIDN')) if hdr_img.get('EXPIDN') is not None else np.nan
        mjd_avg = float(hdr_img.get('MJD-AVG')) if hdr_img.get('MJD-AVG') is not None else np.nan

        # Edge flags using target location (not a fit result). Use image dims if available; else set NaN.
        try:
            near_detector_edge = bool((xpix < 10) or (ypix < 10) or ((w_img - xpix) < 10) or ((h_img - ypix) < 10))
        except Exception:
            near_detector_edge = np.nan

        # Without a cutout we cannot compute these; set to NaN / False where appropriate.
        near_cutout_edge = np.nan

        outputs = FitOutputs(
            # CODE PARAMETERS
            input_ra_deg=float(ra) if np.isfinite(ra) else np.nan,
            input_dec_deg=float(dec) if np.isfinite(dec) else np.nan,
            fit_radius_px=float(fit_radius_px) if np.isfinite(fit_radius_px) else np.nan,
            box_size_bcg_subtract=box_size_bcg_subtract,
            no_masking=bool(no_masking),

            # FITS-HEADER RELATED STUFF
            obsid=hdr_img.get('OBSID', None),
            bandpass=bandpass,
            expid=expid,
            mjd_avg=mjd_avg,
            detector_id=detector_id,
            psf_index=np.nan,
            omega_sr=float(omega_sr) if np.isfinite(omega_sr) else np.nan,
            px_scale_arcsec=float(px_arcsec) if np.isfinite(px_arcsec) else np.nan,

            # FLAGS
            near_cutout_edge=near_cutout_edge,
            near_detector_edge=near_detector_edge,
            near_bcg_star=False,
            n_pix_flagged_in_fit=np.nan,
            n_pix_used_in_fit=np.nan,
            n_pix_total_in_fit=np.nan,

            # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
            ap_radius_px=np.nan,
            ap_radius_forced=np.nan,
            ap_flux_MJysr=np.nan,
            ap_flux_MJysr_err=np.nan,
            ap_snr=np.nan,
            ap_centroid_err_px=np.nan,
            ap_flux_uJy=np.nan,
            ap_flux_uJy_err=np.nan,
            ap_xcen_cutout=np.nan,
            ap_ycen_cutout=np.nan,
            ap_xcen_fullim=np.nan,
            ap_ycen_fullim=np.nan,

            # CENTER OF MASS OUTPUTS
            com_xcen_cutout=np.nan,
            com_ycen_cutout=np.nan,
            com_xcen_fullim=np.nan,
            com_ycen_fullim=np.nan,
            com_ra_deg=np.nan,
            com_dec_deg=np.nan,
            com_sep_as=np.nan,
            com_wv_um=np.nan,
            com_wv_width_um=np.nan,

            # PSF-SCIPY OUTPUTS
            psf_scipy_method_used=np.nan,
            psf_scipy_status=np.nan,
            psf_scipy_flux_MJysr=np.nan,
            psf_scipy_flux_MJysr_err=np.nan,
            psf_scipy_snr=np.nan,
            psf_scipy_flux_uJy=np.nan,
            psf_scipy_flux_uJy_err=np.nan,
            psf_scipy_dx=np.nan,
            psf_scipy_dy=np.nan,
            psf_scipy_xcen_cutout=np.nan,
            psf_scipy_ycen_cutout=np.nan,
            psf_scipy_xcen_fullim=np.nan,
            psf_scipy_ycen_fullim=np.nan,
            psf_scipy_ra_deg=np.nan,
            psf_scipy_ra_err_mas=np.nan,
            psf_scipy_dec_deg=np.nan,
            psf_scipy_dec_err_mas=np.nan,
            psf_scipy_ra_dec_cov_mas2=np.nan,
            psf_scipy_sep_as=np.nan,
            psf_scipy_wv_um=np.nan,
            psf_scipy_wv_um_err=np.nan,
            psf_scipy_wv_width_um=np.nan,
            psf_scipy_wv_width_um_err=np.nan,
            psf_scipy_chi2=np.nan,
            psf_scipy_dof=np.nan,

            # PSF-ULTRANEST OUTPUTS
            psf_un_flux_MJysr=np.nan,
            psf_un_flux_MJysr_err=np.nan,
            psf_un_snr=np.nan,
            psf_un_flux_uJy=np.nan,
            psf_un_flux_uJy_err=np.nan,
            psf_un_dx=np.nan,
            psf_un_dy=np.nan,
            psf_un_xcen_cutout=np.nan,
            psf_un_ycen_cutout=np.nan,
            psf_un_xcen_fullim=np.nan,
            psf_un_ycen_fullim=np.nan,
            psf_un_xcen_err=np.nan,
            psf_un_ycen_err=np.nan,
            psf_un_ra_deg=np.nan,
            psf_un_ra_err_mas=np.nan,
            psf_un_dec_deg=np.nan,
            psf_un_dec_err_mas=np.nan,
            psf_un_sep_as=np.nan,
            psf_un_ra_dec_cov_mas2=np.nan,
            psf_un_wv_um=np.nan,
            psf_un_wv_um_err=np.nan,
            psf_un_wv_width_um=np.nan,
            psf_un_wv_width_um_err=np.nan,
            psf_un_chi2=np.nan,
            psf_un_dof=np.nan,
        )
        result_dict = asdict(outputs)
        if save_results:
            out_path = Path(results_path) if results_path else Path(f"{fits_path}.results.json")
            with open(out_path, "w") as f:
                _json.dump(result_dict, f, indent=2)
            if debug:
                print(f"[INFO] Results saved to {out_path}")
        return result_dict

    # If use_ultranest=False, the PSF fit relies on SciPy warm-start results and skips nested sampling.

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    def save_or_show(fig, filename_stub: str):
        """
        Shows or saves the figure based on (show_figs, save_figs).
        Saves into figs_dir if provided. Always closes the figure at the end.
        """
        if show_figs:
            plt.show()
        elif save_figs:
            Path(figs_dir or ".").mkdir(parents=True, exist_ok=True)
            # Use OBSID if available to help disambiguate outputs
            obsid = hdr_img.get('OBSID', 'obs') if 'hdr_img' in locals() else 'obs'
            out = Path(figs_dir or ".") / f"{obsid}_{filename_stub}.png"
            fig.savefig(out, dpi=130)
        plt.close(fig)

    # === helpers ===
    def weighted_quantile(values, quantiles, sample_weight=None):
        values = np.asarray(values)
        quantiles = np.asarray(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        else:
            sample_weight = np.asarray(sample_weight)
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        cdf = np.cumsum(sample_weight)
        cdf /= cdf[-1]
        return np.interp(quantiles, cdf, values)

    # === load data ===
    hdul = fits.open(fits_path)
    img = hdul[1].data
    flags = hdul[2].data
    var = hdul[3].data
    psf_cube = hdul[5].data  # expected shape: (npsf, ny, nx)
    dprint("PSF cube shape:", psf_cube.shape)
    hdr_img = hdul[1].header
    hdr_psf = hdul[5].header
    wcs = WCS(hdr_img)
    oversamp = int(hdr_psf.get('OVERSAMP', 10))
    cdelt_arcsec = float(hdr_psf.get('CDELT1', 0.615))
    px_arcsec = cdelt_arcsec * oversamp
    omega_arcsec2 = float(hdr_img.get('HIERARCH OMEGA_MEDIAN', np.nan))
    # pixel solid angle (median) in steradians (arcsec^2 -> sr)
    arcsec2_to_sr = (pi / (180.0 * 3600.0))**2
    omega_sr = omega_arcsec2 * arcsec2_to_sr if np.isfinite(omega_arcsec2) else np.nan
    dprint(f"PSF CDELT={cdelt_arcsec:.3f} arcsec (high-res), OVERSAMP={oversamp}, => detector pixel ~ {px_arcsec:.3f} arcsec")
    if np.isfinite(omega_arcsec2):
        dprint(f"Image header OMEGA_MEDIAN={omega_arcsec2:.3f} arcsec^2 (=> {np.sqrt(omega_arcsec2):.3f} arcsec on a side)")
    dprint(f"Pixel solid angle (median) ~ {omega_sr:.3e} sr")

    _test_idx = psf_cube.shape[0] // 2
    _sum_pix = float(np.nansum(psf_cube[_test_idx]))
    dprint(f"PSF pixel-sum check @ idx={_test_idx}: sum≈{_sum_pix:.6g} (expect ~1 on det grid)")
    
    h_img, w_img = img.shape  # (rows, cols) = (y, x)
    det_origin_x = float(hdr_img.get('SPXORX0', 0.0))
    det_origin_y = float(hdr_img.get('SPXORY0', 0.0))
    det_w = int(hdr_img.get('SPXNX', w_img))
    det_h = int(hdr_img.get('SPXNY', h_img))
    dprint("Loaded image with shape:", img.shape)
    dprint("Loaded flags shape:", flags.shape, "variance shape:", var.shape)

    # === spectral WCS (WCS-WAVE; ext 6) ===
    # We DO NOT apply SIP here; the lookup table expects raw FITS pixel coords.
    # We'll bilinearly interpolate VALUES at the requested raw detector (x,y).
    try:
        wave_tab = hdul[6].data

        # Extract grids as 1D float arrays (should be length ~9 each)
        X_grid = np.asarray(wave_tab['X'][0], dtype=float).ravel()
        Y_grid = np.asarray(wave_tab['Y'][0], dtype=float).ravel()

        # VALUES expected shape is (2, ny, nx) or sometimes (2, nx, ny) depending on FITS layout.
        # We will lazily orient each layer when we use it.
        VALUES = wave_tab['VALUES'][0]  # do NOT force dtype=float here to avoid collapsing ragged dims

        if debug:
            try:
                _vals_arr = np.asarray(VALUES)
                dprint(f"WCS-WAVE table loaded: X_grid len={len(X_grid)}, Y_grid len={len(Y_grid)}, VALUES raw shape={_vals_arr.shape}")
            except Exception as _e:
                dprint(f"WCS-WAVE VALUES shape introspection failed: {_e}")
    
    except Exception as e:
        X_grid = Y_grid = VALUES = None
        dprint(f"[WARN] Could not load WCS-WAVE (ext 6): {e}")

    def _orient_value_layer(Vlayer, Xg, Yg):
        """
        Ensure Vlayer has shape (len(Yg), len(Xg)) i.e. (ny, nx).
        Accepts common permutations such as (ny, nx) or (nx, ny) and transposes when needed.
        Raises a clear error if shapes are inconsistent with grids.
        """
        V = np.asarray(Vlayer, dtype=float)
        V = np.squeeze(V)
        if V.ndim != 2:
            raise ValueError(f"WCS-WAVE layer must be 2D after squeeze, got ndim={V.ndim}, shape={V.shape}")

        ny, nx = V.shape
        if (ny == len(Yg)) and (nx == len(Xg)):
            return V  # already (ny, nx)
        if (ny == len(Xg)) and (nx == len(Yg)):
            return V.T  # (nx, ny) -> transpose

        # Try last-resort transpose if one axis matches and the other is 2 (some FITS oddities)
        if (ny == len(Yg)) and (nx == 2) and (len(Xg) == 2):
            return V
        if (nx == len(Xg)) and (ny == 2) and (len(Yg) == 2):
            return V

        raise ValueError(f"WCS-WAVE layer shape {V.shape} incompatible with X_grid len={len(Xg)} and Y_grid len={len(Yg)}")

    def _extract_wcswave_layers(VALUES, Xg, Yg):
        """
        From the raw VALUES array in the WCS-WAVE table, extract the two 2D layers
        (central wavelength and width) and orient each to (len(Yg), len(Xg)) = (ny, nx).
        Handles layouts: (2, ny, nx), (ny, nx, 2), (2, nx, ny), (nx, ny, 2).
        Returns (V_lam, V_wid) both shaped (ny, nx).
        """
        A = np.asarray(VALUES)

        if A.ndim == 3:
            # Identify which axis enumerates the two layers (lambda, width)
            if A.shape[-1] == 2:          # (ny, nx, 2) or (nx, ny, 2)
                Vlam_raw = A[..., 0]
                Vwid_raw = A[..., 1]
            elif A.shape[0] == 2:         # (2, ny, nx) or (2, nx, ny)
                Vlam_raw = A[0, ...]
                Vwid_raw = A[1, ...]
            else:
                raise ValueError(f"WCS-WAVE VALUES 3D shape ambiguous: {A.shape}; expected one axis==2")
        elif A.ndim == 2:
            # Single 2D plane is not enough—this would mean only one of (lambda,width) present.
            raise ValueError(f"WCS-WAVE VALUES has only one 2D layer {A.shape}; expected two layers (lambda,width)")
        else:
            raise ValueError(f"WCS-WAVE VALUES unexpected ndim={A.ndim}, shape={A.shape}")

        # Orient each to (ny, nx)
        V_lam = _orient_value_layer(Vlam_raw, Xg, Yg)
        V_wid = _orient_value_layer(Vwid_raw, Xg, Yg)
        return V_lam, V_wid

    def _bilinear_interp_on_grid(x, y, Xg, Yg, V):
        """
        Bilinear interpolate V[j,i] defined on grid (Xg[i], Yg[j]) at (x,y).
        Xg, Yg must be 1D sorted arrays (len n=9). V is (ny, nx).
        Returns interpolated scalar.
        """
        # Safety: V must be (ny, nx) matching Yg and Xg
        if V.ndim != 2 or V.shape[0] < 2 or V.shape[1] < 2:
            raise ValueError(f"_bilinear_interp_on_grid: unexpected V shape {V.shape}; expected (ny, nx) with >=2 each.")
        # clamp to grid bounds
        x = np.clip(x, Xg[0], Xg[-1])
        y = np.clip(y, Yg[0], Yg[-1])

        # find i such that Xg[i] <= x <= Xg[i+1]
        i = np.searchsorted(Xg, x) - 1
        j = np.searchsorted(Yg, y) - 1
        i = np.clip(i, 0, len(Xg)-2)
        j = np.clip(j, 0, len(Yg)-2)

        x0, x1 = Xg[i], Xg[i+1]
        y0, y1 = Yg[j], Yg[j+1]
        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

        # corners
        f00 = V[j,   i  ]
        f10 = V[j,   i+1]
        f01 = V[j+1, i  ]
        f11 = V[j+1, i+1]
        # bilinear
        return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

    def wcswave_eval_xy_fullimg(x_fullimg_px, y_fullimg_px):
        """
        Evaluate spectral WCS lookup (central wavelength, bandpass width) at a detector pixel.
        Inputs are astropy 0-based pixel indices. Lookup table expects raw FITS coords,
        which are 1-based, so we add +1 before interpolation.
        Returns (lambda_um, width_um).
        """
        if VALUES is None:
            return np.nan, np.nan
        x_raw = float(x_fullimg_px) + det_origin_x + 1.0
        y_raw = float(y_fullimg_px) + det_origin_y + 1.0

        # Lazily extract and orient the two VALUE layers to (ny, nx),
        # supporting layouts like (2, ny, nx) or (ny, nx, 2).
        try:
            V_lam, V_wid = _extract_wcswave_layers(VALUES, X_grid, Y_grid)
        except Exception as e:
            dprint(f"[WCS-WAVE] layer orientation failed: {e}")
            return np.nan, np.nan

        lam_um = _bilinear_interp_on_grid(x_raw, y_raw, X_grid, Y_grid, V_lam)
        dlam_um = _bilinear_interp_on_grid(x_raw, y_raw, X_grid, Y_grid, V_wid)
        return lam_um, dlam_um

    def wcswave_sample_stats(x_mu, y_mu, sigx, sigy, nsamp=400):
        """
        Draw Gaussian (x,y) samples in detector pixels, map via WCS-WAVE, and
        return mean and stdev for central wavelength and bandpass width.
        """
        if not (np.isfinite(x_mu) and np.isfinite(y_mu)) or VALUES is None:
            return (np.nan, np.nan), (np.nan, np.nan)
        sigx = float(sigx) if (sigx is not None and np.isfinite(sigx) and sigx > 0) else 0.5
        sigy = float(sigy) if (sigy is not None and np.isfinite(sigy) and sigy > 0) else 0.5
        xs = np.random.normal(x_mu, sigx, size=nsamp)
        ys = np.random.normal(y_mu, sigy, size=nsamp)
        lams = np.empty(nsamp); dls = np.empty(nsamp)
        for k in range(nsamp):
            lams[k], dls[k] = wcswave_eval_xy_fullimg(xs[k], ys[k])
        return (float(np.nanmean(lams)), float(np.nanstd(lams))), (float(np.nanmean(dls)), float(np.nanstd(dls)))

    # === target pixel ===
    sky = SkyCoord(ra*u.deg, dec*u.deg)
    xpix, ypix = wcs.world_to_pixel(sky)
    pos = (xpix, ypix)

    dprint(f"Target sky coords (deg): RA={ra:.8f}, Dec={dec:.8f}")
    dprint(f"Target pixel (full image): x={xpix:.3f}, y={ypix:.3f}")

    # === cutouts ===

    # --- Preserve full 15x15 cutout and build tight fit window ---
    # Guard against the source being fully off the detector (astropy raises NoOverlapError).
    try:
        cut_img = Cutout2D(img, pos, cutout_size, wcs=wcs, mode='partial', fill_value=np.nan)
        cut_flags = Cutout2D(flags, pos, cutout_size, wcs=wcs, mode='partial', fill_value=0).data
        cut_var = Cutout2D(var, pos, cutout_size, wcs=wcs, mode='partial', fill_value=np.nan).data
    except NoOverlapError:
        # Off-detector: return a full NaN payload (same structure as the "all pixels masked" abort below).
        return _abort_with_nans("Cutout2D NoOverlapError (target outside image)")

    # Preserve full 15x15 arrays for diagnostics
    cut_img_full = cut_img.data.copy()
    cut_flags_full = cut_flags.copy()
    cut_var_full = cut_var.copy()

    # compute target position in the full cutout frame
    ys, xs = cut_img.slices_original
    x0, y0 = xs.start, ys.start
    xcut_full, ycut_full = xpix - x0, ypix - y0
    h_full, w_full = cut_img_full.shape

    # ---- Background subtraction on full cutout ----
    # Build a selective bad-pixel mask on the FULL cutout (will also use for plotting)
    def bit(n):
        return (1 << int(n))

    MP = {
        'TRANSIENT': 0,
        'OVERFLOW': 1,
        'SUR_ERROR': 2,
        'NONFUNC': 6,
        'DICHROIC': 7,
        'MISSING_DATA': 9,
        'HOT': 10,
        'COLD': 11,
        'FULLSAMPLE': 12,
        'PHANMISS': 14,
        'NONLINEAR': 15,
        'PERSIST': 17,
        'OUTLIER': 19,
        'SOURCE': 21,
    }
    #Before Sep 29 1:38pm
    #BAD_BITS = (
    #    bit(MP['TRANSIENT']) | bit(MP['OVERFLOW']) | bit(MP['SUR_ERROR']) |
    #    bit(MP['NONFUNC']) | bit(MP['DICHROIC']) |
    #    bit(MP['MISSING_DATA']) | bit(MP['HOT']) | bit(MP['COLD']) |
    #    bit(MP['PHANMISS']) | bit(MP['NONLINEAR']) | bit(MP['PERSIST']) | bit(MP['FULLSAMPLE'])
    #)
    #mask_full = (cut_flags_full.astype(np.uint32) & BAD_BITS) != 0
    #BAD_BITS_BCG = (
    #    bit(MP['TRANSIENT']) | bit(MP['OVERFLOW']) | bit(MP['SUR_ERROR']) |
    #    bit(MP['NONFUNC']) | bit(MP['DICHROIC']) |
    #    bit(MP['MISSING_DATA']) | bit(MP['HOT']) | bit(MP['COLD']) |
    #    bit(MP['PHANMISS']) | bit(MP['NONLINEAR']) | bit(MP['PERSIST']) |
    #    bit(MP['SOURCE']) | bit(MP['OUTLIER']) | bit(MP['FULLSAMPLE'])
    #)
    #mask_full_bcg = (cut_flags_full.astype(np.uint32) & BAD_BITS_BCG) != 0

    #IRSA Helpdesk recommended
    BAD_BITS = (
        bit(MP['SUR_ERROR']) |
        bit(MP['NONFUNC']) | 
        bit(MP['MISSING_DATA']) | bit(MP['HOT']) | bit(MP['COLD']) |
        bit(MP['NONLINEAR']) | bit(MP['PERSIST'])
    )
    mask_full = (cut_flags_full.astype(np.uint32) & BAD_BITS) != 0

    BAD_BITS_BCG = (
        bit(MP['OVERFLOW']) | bit(MP['SUR_ERROR']) |
        bit(MP['NONFUNC']) | 
        bit(MP['MISSING_DATA']) | bit(MP['HOT']) | bit(MP['COLD']) |
        bit(MP['NONLINEAR']) | bit(MP['PERSIST']) |
        bit(MP['SOURCE']) | bit(MP['OUTLIER']) | 
        bit(MP['TRANSIENT'])
    )
    mask_full_bcg = (cut_flags_full.astype(np.uint32) & BAD_BITS_BCG) != 0

    # --- diagnostics: per-flag counts on FULL 15x15 cutout ---
    def _count_bits(flag_array, labels):
        arr = flag_array.astype(np.uint32)
        rows = []
        total_px = arr.size
        for lab in labels:
            b = bit(MP[lab])
            n = int(np.sum((arr & b) != 0))
            rows.append((lab, n, 100.0 * n / max(total_px, 1)))
        return rows, total_px

    if debug:
        # Flags that contribute to each mask; order chosen for readability
        labs_full = ['SUR_ERROR','NONFUNC','MISSING_DATA','HOT','COLD','NONLINEAR','PERSIST']  # BAD_BITS
        labs_bcg  = ['OVERFLOW','SUR_ERROR','NONFUNC','MISSING_DATA','HOT','COLD',
                     'NONLINEAR','PERSIST','SOURCE','OUTLIER','TRANSIENT']  # BAD_BITS_BCG

        rows_full, tot_full = _count_bits(cut_flags_full, labs_full)
        rows_bcg,  tot_full2 = _count_bits(cut_flags_full, labs_bcg)

        dprint("[flags] Per-bit counts on FULL 15x15 (mask_full):")
        for lab, n, pct in rows_full:
            dprint(f"    {lab:<12s}: {n:6d} px ({pct:6.2f}%)")
        dprint(f"    total pixels: {tot_full}")

        dprint("[flags] Per-bit counts on FULL 15x15 (mask_full_bcg):")
        for lab, n, pct in rows_bcg:
            dprint(f"    {lab:<12s}: {n:6d} px ({pct:6.2f}%)")
        dprint(f"    total pixels: {tot_full2}")

   #import pdb; pdb.set_trace()
    
    if debug:
        source_bit = bit(MP['SOURCE']) | bit(MP['OUTLIER'])
        n_source = int(np.sum((cut_flags_full.astype(np.uint32) & source_bit) != 0))
        dprint(f"Pixels with SOURCE or OUTLIER flag in cutout: {n_source} (kept for fit)")

    # Background from a box around the target. If the initial box is fully masked,
    # try doubling the side length up to 2 times (i.e., 1x, 2x, 4x).
    def _compute_background_with_expanding_box(initial_size_px: int, max_doubles: int = 2):
        """
        Try to estimate the background using progressively larger cutouts centered at (xpix, ypix).
        Returns (bkg_value, used_box_size, n_good) where bkg_value is NaN if all attempts fail.
        """
        # Helper to get a median ignoring both mask and non-finite pixels
        def _median_masked(a, mask):
            # consider only finite pixels AND not masked
            good = (~mask) & np.isfinite(a)
            if not np.any(good):
                return np.nan, 0
            return np.nanmedian(a[good]), int(np.sum(good))

        size = int(initial_size_px)
        for k in range(max_doubles + 1):
            try:
                cut_img_b = Cutout2D(img, pos, (size, size), wcs=wcs, mode='partial', fill_value=np.nan)
                cut_flags_b = Cutout2D(flags, pos, (size, size), wcs=wcs, mode='partial', fill_value=0).data
                # Build the BCG mask on this box
                mask_bcg = (cut_flags_b.astype(np.uint32) & BAD_BITS_BCG) != 0
                bkg_val, n_good = _median_masked(cut_img_b.data, mask_bcg)
                if np.isfinite(bkg_val):
                    return float(bkg_val), size, n_good
                else:
                    dprint(f"[bcg] box {size}×{size} has no usable pixels; expanding…")
            except Exception as _e:
                dprint(f"[bcg] expanding box {size}×{size} failed: {_e}")
            size *= 2  # double the side length and try again
        # If we are in --no-masking mode, and every attempt failed because the mask removed
        # all usable pixels, fall back to an *unmasked* median background estimate using the
        # original box size (ignore flags entirely; only require finite pixels).
        if bool(no_masking):
            try:
                size0 = int(initial_size_px)
                cut_img_b = Cutout2D(img, pos, (size0, size0), wcs=wcs, mode='partial', fill_value=np.nan)
                a = cut_img_b.data
                good = np.isfinite(a)
                if np.any(good):
                    bkg_val = float(np.nanmedian(a[good]))
                    if debug:
                        dprint(f"[bcg] --no-masking fallback: unmasked median background={bkg_val:.6g} using original box {size0}×{size0} (good px={int(np.sum(good))})")
                    return bkg_val, size0, int(np.sum(good))
                else:
                    if debug:
                        dprint(f"[bcg] --no-masking fallback failed: no finite pixels in original box {size0}×{size0}")
            except Exception as _e:
                if debug:
                    dprint(f"[bcg] --no-masking fallback exception: {_e}")
        return np.nan, None, 0

    # Try background with expanding box, starting from the configured box_size_bcg_subtract
    bkg, bkg_box_used, bkg_ngood = _compute_background_with_expanding_box(box_size_bcg_subtract, max_doubles=2)
    if debug:
        if np.isfinite(bkg):
            dprint(f"[bcg] Background (median of unmasked) = {bkg:.6g} [MJy/sr] using box {bkg_box_used}×{bkg_box_used} (good px={bkg_ngood})")
        else:
            dprint("[bcg] Background could not be estimated (all attempted boxes fully masked or non-finite)")

    data_full = cut_img_full - bkg

    # If data_full contains only NaN or masked pixels, attempt an automatic fallback:
    # if --no-masking was NOT set, toggle it ON and retry background estimation so we do not
    # fail simply because flag-based masking removed all usable pixels.
    if (not bool(no_masking)) and (np.all(mask_full) or (not np.any(np.isfinite(data_full[~mask_full])))):
        dprint("[bcg] auto-toggle: background/flags left no usable pixels; enabling --no-masking and retrying background estimation")
        no_masking = True
        bkg, bkg_box_used, bkg_ngood = _compute_background_with_expanding_box(box_size_bcg_subtract, max_doubles=2)
        if debug:
            if np.isfinite(bkg):
                dprint(f"[bcg] (retry) Background (median) = {bkg:.6g} [MJy/sr] using box {bkg_box_used}×{bkg_box_used} (good px={bkg_ngood})")
            else:
                dprint("[bcg] (retry) Background could not be estimated")
        data_full = cut_img_full - bkg

    #If data_full contains only NaN or masked pixels, abort, but still return a full outputs object
    if (not bool(no_masking) and (np.all(mask_full) or not np.any(np.isfinite(data_full[~mask_full])))) \
       or (bool(no_masking) and (not np.any(np.isfinite(data_full)))):
        dprint("[ERROR] All pixels in cutout are masked or non-finite after background subtraction. Aborting.")
        # --- failure diagnostic: show cutout and flagged pixels ---
        try:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # Use the raw full cutout (pre-bkg) to ensure we can visualize even if data_full is NaN
            im = ax.imshow(cut_img_full, origin='lower')
            # mark target
            ax.scatter([xcut_full], [ycut_full], s=50, marker='+', color='w', label='target')
            # draw intended fit radius
            try:
                circ = plt.Circle((xcut_full, ycut_full), float(fit_radius_px), fill=False, linestyle='--', color='w', label='fit radius')
                ax.add_patch(circ)
            except Exception:
                pass
            # overlay flagged pixels for fit (mask_full) in red x
            yy_f, xx_f = np.where(mask_full)
            if yy_f.size > 0:
                ax.plot(xx_f, yy_f, 'x', markersize=3, alpha=0.8, color='r', linestyle='None', label='ignored in fit (mask_full)')
            # overlay pixels flagged only for background (mask_full_bcg minus mask_full)
            yy_b, xx_b = np.where(mask_full_bcg & (~mask_full))
            if yy_b.size > 0:
                ax.plot(xx_b, yy_b, '.', markersize=2, alpha=0.8, linestyle='None', label='ignored in bkg (mask_full_bcg)')
            ax.legend(loc='upper right', frameon=True, fontsize=8)
            ax.set_title('Failure: all pixels masked / non-finite')
            plt.tight_layout()
            save_or_show(fig, "failure_all_masked")
        except Exception as _e_diag:
            dprint(f"[diag] failure figure generation failed: {_e_diag}")
        # Header-derived values available at this point
        detector_id = int(hdr_img.get('DETECTOR')) if hdr_img.get('DETECTOR') is not None else np.nan
        bandpass = ("D" + str(detector_id)) if (detector_id is not None and detector_id is not np.nan) else np.nan
        expid = int(hdr_img.get('EXPIDN')) if hdr_img.get('EXPIDN') is not None else np.nan
        mjd_avg = float(hdr_img.get('MJD-AVG')) if hdr_img.get('MJD-AVG') is not None else np.nan

        # Edge flags using target location (not a fit result)
        try:
            near_cutout_edge = bool((xcut_full < 5) or (ycut_full < 5) or ((w_full - xcut_full) < 5) or ((h_full - ycut_full) < 5))
        except Exception:
            near_cutout_edge = np.nan
        try:
            near_detector_edge = bool((xpix < 10) or (ypix < 10) or ((w_img - xpix) < 10) or ((h_img - ypix) < 10))
        except Exception:
            near_detector_edge = np.nan

        # Assemble outputs with NaNs for anything not yet computed
        outputs = FitOutputs(
            # CODE PARAMETERS
            input_ra_deg=float(ra) if np.isfinite(ra) else np.nan,
            input_dec_deg=float(dec) if np.isfinite(dec) else np.nan,
            fit_radius_px=float(fit_radius_px) if np.isfinite(fit_radius_px) else np.nan,
            box_size_bcg_subtract=box_size_bcg_subtract,
            no_masking=bool(no_masking),

            # FITS-HEADER RELATED STUFF
            obsid=hdr_img.get('OBSID', None),
            bandpass=bandpass,
            expid=expid,
            mjd_avg=mjd_avg,
            detector_id=detector_id,
            psf_index=np.nan,
            omega_sr=float(omega_sr) if np.isfinite(omega_sr) else np.nan,
            px_scale_arcsec=float(px_arcsec) if np.isfinite(px_arcsec) else np.nan,

            # FLAGS
            near_cutout_edge=near_cutout_edge,
            near_detector_edge=near_detector_edge,
            near_bcg_star=False,
            n_pix_flagged_in_fit=np.nan,
            n_pix_used_in_fit=np.nan,
            n_pix_total_in_fit=np.nan,

            # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
            ap_radius_px=np.nan,
            ap_radius_forced=np.nan,
            ap_flux_MJysr=np.nan,
            ap_flux_MJysr_err=np.nan,
            ap_snr=np.nan,
            ap_centroid_err_px=np.nan,
            ap_flux_uJy=np.nan,
            ap_flux_uJy_err=np.nan,
            ap_xcen_cutout=np.nan,
            ap_ycen_cutout=np.nan,
            ap_xcen_fullim=np.nan,
            ap_ycen_fullim=np.nan,

            # CENTER OF MASS OUTPUTS
            com_xcen_cutout=np.nan,
            com_ycen_cutout=np.nan,
            com_xcen_fullim=np.nan,
            com_ycen_fullim=np.nan,
            com_ra_deg=np.nan,
            com_dec_deg=np.nan,
            com_sep_as=np.nan,
            com_wv_um=np.nan,
            com_wv_width_um=np.nan,

            # PSF-SCIPY OUTPUTS
            psf_scipy_method_used=np.nan,
            psf_scipy_status=np.nan,
            psf_scipy_flux_MJysr=np.nan,
            psf_scipy_flux_MJysr_err=np.nan,
            psf_scipy_snr=np.nan,
            psf_scipy_flux_uJy=np.nan,
            psf_scipy_flux_uJy_err=np.nan,
            psf_scipy_dx=np.nan,
            psf_scipy_dy=np.nan,
            psf_scipy_xcen_cutout=np.nan,
            psf_scipy_ycen_cutout=np.nan,
            psf_scipy_xcen_fullim=np.nan,
            psf_scipy_ycen_fullim=np.nan,
            psf_scipy_ra_deg=np.nan,
            psf_scipy_ra_err_mas=np.nan,
            psf_scipy_dec_deg=np.nan,
            psf_scipy_dec_err_mas=np.nan,
            psf_scipy_ra_dec_cov_mas2=np.nan,
            psf_scipy_sep_as=np.nan,
            psf_scipy_wv_um=np.nan,
            psf_scipy_wv_um_err=np.nan,
            psf_scipy_wv_width_um=np.nan,
            psf_scipy_wv_width_um_err=np.nan,
            psf_scipy_chi2=np.nan,
            psf_scipy_dof=np.nan,

            # PSF-ULTRANEST OUTPUTS
            psf_un_flux_MJysr=np.nan,
            psf_un_flux_MJysr_err=np.nan,
            psf_un_snr=np.nan,
            psf_un_flux_uJy=np.nan,
            psf_un_flux_uJy_err=np.nan,
            psf_un_dx=np.nan,
            psf_un_dy=np.nan,
            psf_un_xcen_cutout=np.nan,
            psf_un_ycen_cutout=np.nan,
            psf_un_xcen_fullim=np.nan,
            psf_un_ycen_fullim=np.nan,
            psf_un_xcen_err=np.nan,
            psf_un_ycen_err=np.nan,
            psf_un_ra_deg=np.nan,
            psf_un_ra_err_mas=np.nan,
            psf_un_dec_deg=np.nan,
            psf_un_dec_err_mas=np.nan,
            psf_un_sep_as=np.nan,
            psf_un_ra_dec_cov_mas2=np.nan,
            psf_un_wv_um=np.nan,
            psf_un_wv_um_err=np.nan,
            psf_un_wv_width_um=np.nan,
            psf_un_wv_width_um_err=np.nan,
            psf_un_chi2=np.nan,
            psf_un_dof=np.nan,
        )

        result_dict = asdict(outputs)
        if save_results:
            out_path = Path(results_path) if results_path else Path(f"{fits_path}.results.json")
            with open(out_path, "w") as f:
                _json.dump(result_dict, f, indent=2)
            if debug:
                print(f"[INFO] Results saved to {out_path}")

        return result_dict

    dprint(f"Background (median of unmasked) = {bkg:.6g} [MJy/sr]")
    dprint(f"Unmasked pixels: {(~mask_full).sum()} / {mask_full.size}")

    # === MODEL-FREE PHOTOMETRY + ASTROMETRY (aperture + centroid) ===
    # We evaluate a small grid of circular apertures and pick the one with the
    # highest S/N. Photometry is done on the background-subtracted full cutout.
    ap_radii_px = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 2.7, 3.0, 3.5, 4.0, 5.0])
    YY_full, XX_full = np.indices(cut_img_full.shape)

    def aperture_metrics(r_px):
        apmask = ((XX_full - xcut_full)**2 + (YY_full - ycut_full)**2) <= (r_px**2)
        # exclude flagged pixels from both flux and noise
        good = apmask & (~mask_full) & np.isfinite(data_full) & np.isfinite(cut_var_full) & (cut_var_full > 0)
        if not np.any(good):
            return dict(r=r_px, npix=0, flux_MJysr=np.nan, sig_MJysr=np.nan, snr=np.nan,
                        xcen=np.nan, ycen=np.nan)
        # flux in surface-brightness units [MJy/sr]; error from variance map
        flux_MJysr = float(np.sum(data_full[good]))
        sig_MJysr = float(np.sqrt(np.sum(cut_var_full[good])))
        snr = flux_MJysr / sig_MJysr if sig_MJysr > 0 else np.nan

        # centroid (center-of-mass) within aperture using background-subtracted data
        w = data_full.copy()
        w[~good] = 0.0
        tot = np.sum(w)
        if tot > 0:
            xcen = float(np.sum(XX_full * w) / tot)
            ycen = float(np.sum(YY_full * w) / tot)
        else:
            xcen, ycen = np.nan, np.nan

        return dict(r=r_px, npix=int(np.sum(good)), flux_MJysr=flux_MJysr, sig_MJysr=sig_MJysr,
                    snr=snr, xcen=xcen, ycen=ycen)

    # Choose aperture strategy: scan grid if ap_radius_px is None, else forced single radius
    if ap_radius_px is None:
        ap_results = [aperture_metrics(r) for r in ap_radii_px]
        valid = [res for res in ap_results if np.isfinite(res['snr'])]
        best_ap = (max(valid, key=lambda d: d['snr']) if len(valid)
                   else dict(r=np.nan, flux_MJysr=np.nan, sig_MJysr=np.nan, snr=np.nan, xcen=np.nan, ycen=np.nan, npix=0))
        forced_ap = False
    else:
        # Compute metrics exactly at the requested radius; skip scanning others
        best_ap = aperture_metrics(float(ap_radius_px))
        ap_results = [best_ap]
        forced_ap = True
    ap_radius_forced = forced_ap
    ap_radius_px = float(best_ap['r']) if np.isfinite(best_ap['r']) else np.nan
    ap_flux_MJysr = float(best_ap['flux_MJysr']) if np.isfinite(best_ap['flux_MJysr']) else np.nan
    ap_flux_MJysr_err = float(best_ap['sig_MJysr']) if np.isfinite(best_ap['sig_MJysr']) else np.nan
    ap_snr = float(best_ap['snr']) if np.isfinite(best_ap['snr']) else np.nan
    ap_xcen_cutout = xcut_full if np.isfinite(xcut_full) else np.nan
    ap_ycen_cutout = ycut_full if np.isfinite(ycut_full) else np.nan
    com_xcen_cutout = float(best_ap['xcen']) if np.isfinite(best_ap['xcen']) else np.nan
    com_ycen_cutout = float(best_ap['ycen']) if np.isfinite(best_ap['ycen']) else np.nan

    ap_xcen_fullimg = (ap_xcen_cutout + x0) if (ap_xcen_cutout is not None and 'x0' in locals()) else np.nan
    ap_ycen_fullimg = (ap_ycen_cutout + y0) if (ap_ycen_cutout is not None and 'y0' in locals()) else np.nan
    com_xcen_fullimg = (com_xcen_cutout + x0) if (com_xcen_cutout is not None and 'x0' in locals()) else np.nan
    com_ycen_fullimg = (com_ycen_cutout + y0) if (com_ycen_cutout is not None and 'y0' in locals()) else np.nan

    com_ra_deg = None
    com_dec_deg = None
    com_sep_as = None
    if (com_xcen_fullimg is not None and com_ycen_fullimg is not None and
        np.isfinite(com_xcen_fullimg) and np.isfinite(com_ycen_fullimg)):
        sky_com = wcs.pixel_to_world(com_xcen_fullimg, com_ycen_fullimg)
        com_ra_deg, com_dec_deg = float(sky_com.ra.deg), float(sky_com.dec.deg)
        com_sep_as = np.sqrt((com_ra_deg - ra)**2*np.cos(np.radians(com_dec_deg))**2 + (com_dec_deg - dec)**2) * 3600.0
    
    com_wv_um = com_wv_width_um = com_wv_um_sig = com_wv_width_um_sig = None
    if (com_xcen_fullimg is not None and com_ycen_fullimg is not None and
        np.isfinite(com_xcen_fullimg) and np.isfinite(com_ycen_fullimg)):
        com_wv_um, com_wv_width_um = wcswave_eval_xy_fullimg(com_xcen_fullimg, com_ycen_fullimg)

    # Convert surface-brightness sum to integrated flux [µJy] via Ω_pix median
    # (approximation; a per-pixel Ω map would be more exact)
    flux_uJy_ap = best_ap['flux_MJysr'] * omega_sr * 1e12
    sig_uJy_ap  = best_ap['sig_MJysr']  * omega_sr * 1e12
    ap_flux_uJy = float(flux_uJy_ap) if np.isfinite(flux_uJy_ap) else np.nan
    ap_flux_uJy_err = float(sig_uJy_ap) if np.isfinite(sig_uJy_ap) else np.nan
    
    # Rough centroid uncertainty via CRLB ~ FWHM / (2.35*SNR)
    psf_fwhm_arcsec = float(hdr_img.get('PSF_FWHM', np.nan))
    px_scale_arcsec = px_arcsec  # from PSF header CDELT * OVERSAMP
    fwhm_px = psf_fwhm_arcsec / px_scale_arcsec if np.isfinite(psf_fwhm_arcsec) and px_scale_arcsec > 0 else np.nan
    snr_best = best_ap['snr']
    ap_snr = snr_best
    centroid_err_px = (fwhm_px / (2.3548 * snr_best)) if (np.isfinite(fwhm_px) and np.isfinite(snr_best) and snr_best > 0) else np.nan
    ap_centroid_err_px = centroid_err_px  # alias for output dataclass

    # Set dx/dy prior half-width from centroid uncertainty (10×), with safe clamps.
    if np.isfinite(centroid_err_px):
        dx_prior_half = dy_prior_half = float(np.clip(10.0 * centroid_err_px, 0.2, 1.0))
    else:
        dx_prior_half = dy_prior_half = 0.5  # fallback
    # Ensure minimum ±1 px prior half-width (i.e. at least 2 px wide window)
    dx_prior_half = max(dx_prior_half, 0.5)
    dy_prior_half = max(dy_prior_half, 0.5)
    dprint(f"dx,dy prior half-width set to ±{dx_prior_half:.3f} px (from 10× centroid σ; clipped to [0.2, 1.0], min 0.5)")

    # --- Optional hard override from CLI: --max-pix-offset ---
    # If provided, set bounds for dx and dy to ±|max_pix_offset|.
    # If set to 0, dx and dy are fixed to 0 (no movement).
    if max_pix_offset is not None:
        dx_prior_half = float(abs(max_pix_offset))
        dy_prior_half = float(abs(max_pix_offset))
        dprint(f"[bounds] Overriding dx,dy prior half-width to ±{dx_prior_half:.3f} px from --max-pix-offset")
    
    if debug:
        dprint("=== Aperture photometry (model-free) ===")
        if not forced_ap:
            for res in ap_results:
                r = res['r']
                dprint(f"  r={r:>3.1f} px: Npix={res['npix']:>3d}  flux={res['flux_MJysr']:.4g} MJy/sr  "
                      f"σ={res['sig_MJysr']:.3g}  S/N={res['snr']:.2f}  "
                      f"xcen={res['xcen']:.2f}  ycen={res['ycen']:.2f}")
        else:
            res = best_ap
            r = res['r']
            dprint(f"  [forced] r={r:>3.1f} px: Npix={res['npix']:>3d}  flux={res['flux_MJysr']:.4g} MJy/sr  "
                  f"σ={res['sig_MJysr']:.3g}  S/N={res['snr']:.2f}  "
                  f"xcen={res['xcen']:.2f}  ycen={res['ycen']:.2f}")
        sel_note = "[forced] " if forced_ap else ""
        dprint(f"-> {sel_note}Selected r={best_ap['r']:.2f} px: flux={best_ap['flux_MJysr']:.4g} MJy/sr "
               f"(≈ {flux_uJy_ap:.3g} µJy), σ≈{best_ap['sig_MJysr']:.3g} MJy/sr (≈ {sig_uJy_ap:.3g} µJy), S/N≈{snr_best:.2f}")
        dprint(f"   Centroid (cutout frame): x={best_ap['xcen']:.3f} ± {centroid_err_px:.3f} px, "
            f"y={best_ap['ycen']:.3f} ± {centroid_err_px:.3f} px  [CRLB approx]")

        # Convert centroid to full-image pixels and RA/Dec
        if np.isfinite(best_ap['xcen']) and np.isfinite(best_ap['ycen']):
            x_full_cent = best_ap['xcen'] + x_min if 'x_min' in locals() else best_ap['xcen'] + x0
            y_full_cent = best_ap['ycen'] + y_min if 'y_min' in locals() else best_ap['ycen'] + y0
            # Use original WCS on full-frame pixel coords
            sky_cent = wcs.pixel_to_world(x_full_cent + 0.0, y_full_cent + 0.0)
            ra_cent, dec_cent = float(sky_cent.ra.deg), float(sky_cent.dec.deg)
            dprint(f"   Centroid (RA,Dec): ({ra_cent:.8f}, {dec_cent:.8f}) deg")
            # --- spectral WCS diagnostics for model-free centroid (aperture COM) ---
            try:
                wv_um_ap, wv_width_um_ap = wcswave_eval_xy_fullimg(x_full_cent, y_full_cent)
                (wv_mu_ap, wv_sig_ap), (wd_mu_ap, wd_sig_ap) = wcswave_sample_stats(
                    x_full_cent, y_full_cent,
                    centroid_err_px, centroid_err_px
                )
                if (np.isfinite(wv_um_ap) and np.isfinite(wv_sig_ap)
                        and np.isfinite(wv_width_um_ap) and np.isfinite(wd_sig_ap)):
                    dprint(f"[WCS-WAVE] Aperture (COM): λ={wv_um_ap:.5f} ± {wv_sig_ap:.5f} µm ; "
                           f"width={wv_width_um_ap:.5f} ± {wd_sig_ap:.5f} µm")
            except Exception as _e:
                dprint(f"[WCS-WAVE] Aperture (COM) diagnostics failed: {_e}")

        # Diagnostic plot on FULL 15×15 with aperture and flagged pixels
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        im = ax.imshow(data_full, origin='lower')
        ax.scatter([xcut_full], [ycut_full], s=50, marker='+', color='w', label='target')
        # aperture circle
        ap_circ = plt.Circle((xcut_full, ycut_full), best_ap['r'], fill=False, linestyle='-', color='lime', lw=1.5, label=f"aperture r={best_ap['r']:.1f}px")
        ax.add_patch(ap_circ)
        # show fit radius too (white dashed)
        circ = plt.Circle((xcut_full, ycut_full), fit_radius_px, fill=False, linestyle='--', color='w', label='fit radius')
        ax.add_patch(circ)
        # mark ignored (flagged) pixels with small x's
        yy_f, xx_f = np.where(mask_full)
        if yy_f.size > 0:
            ax.plot(xx_f, yy_f, 'x', markersize=3, alpha=0.7, color='r', linestyle='None', label='ignored (flags)')
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.set_title('Aperture photometry on full 15×15 (bg-subtracted)')
        plt.tight_layout()
        save_or_show(fig, "full15_aperture")
    
    # ---- Build tight fit window after background subtraction ----
    fit_half = int(np.ceil(fit_radius_px + 1))  # one pixel margin beyond fit radius

    yc = int(np.round(ycut_full))
    xc = int(np.round(xcut_full))
    y_min = max(0, yc - fit_half)
    y_max = min(h_full, yc + fit_half + 1)
    x_min = max(0, xc - fit_half)
    x_max = min(w_full, xc + fit_half + 1)

    # Slice arrays to the fit window
    data = data_full[y_min:y_max, x_min:x_max]
    cut_flags = cut_flags_full[y_min:y_max, x_min:x_max]
    cut_var = cut_var_full[y_min:y_max, x_min:x_max]

    # Update cutout-relative coordinates for the fit window
    xcut = xcut_full - x_min
    ycut = ycut_full - y_min

    # Recompute mask and weights on the fit window
    if no_masking:
        # Disable *only* PSF-fit masking based on flags (BCG subtraction masking above is unchanged)
        mask = np.zeros_like(cut_flags, dtype=bool)
        if debug:
            dprint("[masking] --no-masking enabled: ignoring all flag-based masking in PSF fit window")
    else:
        mask = (cut_flags.astype(np.uint32) & BAD_BITS) != 0

    YY, XX = np.indices(data.shape)
    r2 = (XX - xcut)**2 + (YY - ycut)**2
    radmask = r2 <= (fit_radius_px**2)

    ivar = np.where(cut_var > 0, 1.0/cut_var, 0.0)
    fit_mask = (~mask) & radmask & np.isfinite(ivar) & (ivar > 0) & np.isfinite(data)

    # Auto-toggle: if flag-based masking leaves zero usable pixels in the PSF-fit region,
    # enable --no-masking and rebuild mask+fit_mask (BCG subtraction masking above remains unchanged).
    if (not bool(no_masking)) and (not np.any(fit_mask)) and np.any(radmask) and np.any(np.isfinite(data)):
        dprint("[masking] auto-toggle: PSF-fit region fully masked by flags; enabling --no-masking and retrying PSF fit masking")
        no_masking = True
        mask = np.zeros_like(cut_flags, dtype=bool)
        fit_mask = (~mask) & radmask & np.isfinite(ivar) & (ivar > 0) & np.isfinite(data)

    #Set data to zero where NaN to avoid NaNs winning over fit_mask
    data[~np.isfinite(data)] = 0.0
    ivar[~np.isfinite(ivar)] = 1.0

    # Fit-window bookkeeping for CSV
    n_total_in_fit = int(fit_mask.size)
    n_used_in_fit = int(np.sum(fit_mask))
    n_flagged_in_fit = int(np.sum(mask & radmask)) if not no_masking else 0

    if debug:
        dprint(f"Fit window shape: {data.shape} (y:[{y_min}:{y_max}), x:[{x_min}:{x_max}))")
        dprint(f"Fit radius = {fit_radius_px:.1f} px -> used pixels: {int(np.sum(fit_mask))} / {fit_mask.size}")

    # --- Diagnostic plot: always show full 15x15 cutout with flagged pixels and fit-radius circle ---
    if debug:
        dprint(f"Cutout bbox (x:[{x0}:{xs.stop}), y:[{y0}:{ys.stop})) size=({w_full}x{h_full})")
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(cut_img_full, origin='lower')
        # target position on full box
        ax.scatter([xcut_full],[ycut_full], s=50, marker='+', color='w', label='target')
        # fit radius drawn on full box (visual)
        circ = plt.Circle((xcut_full, ycut_full), fit_radius_px, fill=False, linestyle='--', color='w')
        ax.add_patch(circ)
        # mark ignored (flagged) pixels with small x's on FULL box
        yy_f, xx_f = np.where(mask_full)
        if yy_f.size > 0:
            ax.plot(xx_f, yy_f, 'x', markersize=3, alpha=0.7, color='r', linestyle='None', label='ignored (flags)')
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.set_title('Full 15×15 cutout: target + ignored pixels + fit radius')
        plt.tight_layout()
        save_or_show(fig, "full15_fit_radius")

    # Reconstruct the independent detector X/Y grid rather than trusting the
    # historical per-plane header pairing, and recover detector-global pixels
    # when this image is a local cutout.
    psf_selection = select_spherex_psf(
        psf_cube,
        hdr_psf,
        float(xpix),
        float(ypix),
        image_header=hdr_img,
    )
    idx_psf = psf_selection.plane_index
    psf_img = psf_selection.image
    oversamp = psf_selection.oversamp
    # IMPORTANT: do not renormalize; IRSA PSFs are already photometrically correct
    if debug:
        dprint(f"Selected PSF index: {idx_psf} / {psf_cube.shape[0]-1}")
        dprint(
            "PSF selection coordinates: "
            f"local=({xpix:.3f}, {ypix:.3f}), "
            f"detector=({psf_selection.detector_x_px:.3f}, {psf_selection.detector_y_px:.3f}), "
            f"transform={psf_selection.coordinate_transform}"
        )
        if psf_selection.zone_x_center_px is not None and psf_selection.zone_y_center_px is not None:
            dprint(
                "Zone center of selected PSF: "
                f"x={psf_selection.zone_x_center_px:.3f}, "
                f"y={psf_selection.zone_y_center_px:.3f}"
            )
            dprint(
                "Distance from target to PSF zone center: "
                f"{np.hypot(psf_selection.zone_x_center_px - psf_selection.detector_x_px, psf_selection.zone_y_center_px - psf_selection.detector_y_px):.3f} px"
            )

    # Helper: build detector-scale PSF cut (H x W) centered on (xcut, ycut) with extra dx,dy offsets
    # Strategy:
    #  1) Crop the high-res PSF to a size that is divisible by OVERSAMP around its center.
    #  2) Subpixel-shift the canvas
    #  2) **Block-sum**-downsample by OVERSAMP to obtain a detector-grid PSF (flux-conserving).
    #  3) Embed that detector PSF into an HxW canvas

    def make_psf_cut(dx=0.0, dy=0.0):
        """
        # model returns image in [MJy/sr] given flux in [MJy] and PSF in [1/sr]
        Build a flux-conserving PSF model on the cutout grid with subpixel offsets.
        Strategy: shift on the oversampled (high-res) grid first, then block-sum
        down to detector pixels. This preserves the PSF width and avoids extra
        broadening from interpolating on the coarse detector grid.
        """
        H, W = data.shape

        # 1) Centered high-res crop whose size is divisible by oversamp
        cy_hr, cx_hr = psf_img.shape[0] // 2, psf_img.shape[1] // 2
        m = (min(psf_img.shape[0], psf_img.shape[1]) // oversamp) * oversamp
        y0_hr = cy_hr - m // 2
        x0_hr = cx_hr - m // 2
        hr = psf_img[y0_hr:y0_hr + m, x0_hr:x0_hr + m]

        # 2) Place the high-res PSF onto a high-res canvas for the cutout
        Hr, Wr = H * oversamp, W * oversamp
        hr_canvas = np.zeros((Hr, Wr), dtype=hr.dtype)
        cyc_hr, cxc_hr = Hr // 2, Wr // 2
        y0c = cyc_hr - m // 2
        x0c = cxc_hr - m // 2
        y1c = y0c + m
        x1c = x0c + m
        ys0 = max(0, -y0c); xs0 = max(0, -x0c)
        ye1 = m - max(0, y1c - Hr); xe1 = m - max(0, x1c - Wr)
        if ye1 > ys0 and xe1 > xs0:
            hr_canvas[max(0, y0c):min(Hr, y1c), max(0, x0c):min(Wr, x1c)] = hr[ys0:ye1, xs0:xe1]

        # 3) Subpixel shift on the high-res grid so the center lands at (xcut,ycut)
        # Desired offset in detector pixels relative to the cutout center:
        dY_det = (ycut - (H - 1) / 2.0) + dy
        dX_det = (xcut - (W - 1) / 2.0) + dx
        # Convert to **high-res** pixels
        dY_hr = dY_det * oversamp
        dX_hr = dX_det * oversamp
        hr_shifted = ndimage.shift(hr_canvas, shift=(dY_hr, dX_hr), order=3, mode='constant', prefilter=True)

        # 4) Flux-conserving block-sum to detector pixels -> shape (H,W)
        det = hr_shifted.reshape(H, oversamp, W, oversamp).sum(axis=(1, 3))

        return det

    # Quick preview of PSF mapping for sanity (no dx,dy)
    if debug:
        psf_cut_preview = make_psf_cut(0.0, 0.0)
        dprint(f"PSF oversampling factor (OVERSAMP) = {oversamp}")

        # Also show the block-summed detector-scale PSF alone
        # Note: shift now applied in high-res before binning
        cy_hr, cx_hr = psf_img.shape[0]//2, psf_img.shape[1]//2
        m = (min(psf_img.shape[0], psf_img.shape[1]) // oversamp) * oversamp
        y0_hr = cy_hr - m//2
        x0_hr = cx_hr - m//2
        hr = psf_img[y0_hr:y0_hr+m, x0_hr:x0_hr+m]
        ny_det = m // oversamp
        nx_det = m // oversamp
        det_only = hr.reshape(ny_det, oversamp, nx_det, oversamp).sum(axis=(1,3))

        # Compute PSF normalization constant (sum on detector grid)
        psf_det_sum = float(np.nansum(det_only))
        dprint(f"[PSF norm] detector-grid PSF sum = {psf_det_sum:.6g} (should be ~1 if PSF were unit-normalized)")
        if (psf_det_sum is None) or (not np.isfinite(psf_det_sum)) or (psf_det_sum < 1e-6):
            # fallback: use sum of cutout PSF at center
            psf_det_sum = float(np.nansum(make_psf_cut(0.0, 0.0)))
        dprint("[note] We do NOT renormalize the PSF; instead we account for its discrete sum ΣPSF_det when converting the fitted amplitude [MJy/sr] to integrated flux [Jy].")

        fig, axs = plt.subplots(1,4, figsize=(16,4))
        core = psf_img[psf_img.shape[0]//2-20:psf_img.shape[0]//2+21, psf_img.shape[1]//2-20:psf_img.shape[1]//2+21]
        axs[0].imshow(core, origin='lower'); axs[0].set_title('PSF high-res core (41x41)')
        axs[1].imshow(det_only, origin='lower'); axs[1].set_title('PSF detector grid (block-sum)')
        axs[2].imshow(psf_cut_preview, origin='lower'); axs[2].scatter([xcut],[ycut], s=20, marker='+'); axs[2].set_title('PSF mapped to cutout grid')
        axs[3].imshow(data, origin='lower'); axs[3].scatter([xcut],[ycut], s=20, marker='+'); axs[3].set_title('Data (for comparison)')
        plt.tight_layout()
        save_or_show(fig, "psf_preview")

        # Sanity: pixel-sum checks (PSF is dimensionless on detector grid; sums should be ~1)
        sum_hr = float(np.nansum(hr))
        sum_det = float(np.nansum(det_only))
        dprint(f"PSF high-res crop pixel-sum ≈ {sum_hr:.6g} ; detector-grid pixel-sum ≈ {sum_det:.6g} (expect ~1)")

    # === initial guess via matched filter (linear least-squares for flux) ===
    # Model is: image [MJy/sr] = amplitude_MJysr * PSF_det [dimensionless]
    H, W = data.shape
    psf_cut0 = make_psf_cut(0.0, 0.0)  # [1/sr] on detector grid (no shift)
    # If psf_det_sum not set (e.g., debug==False), compute it here from psf_cut0
    if psf_det_sum is None:
        psf_det_sum = float(np.nansum(psf_cut0))
        if (psf_det_sum is None) or (not np.isfinite(psf_det_sum)) or (psf_det_sum < 1e-6):
            psf_det_sum = float(np.nansum(make_psf_cut(0.0, 0.0)))
        if debug:
            dprint(f"[PSF norm] detector-grid PSF sum = {psf_det_sum:.6g} (should be ~1 if PSF were unit-normalized)")
            dprint("[note] We do NOT renormalize the PSF; instead we account for its discrete sum ΣPSF_det when converting the fitted amplitude [MJy/sr] to integrated flux [Jy].")

    # Optional: debug sanity check for psf_cut0
    if debug:
        dprint(f"psf_cut0 stats: sum={np.sum(psf_cut0):.6g}, max={np.max(psf_cut0):.6g}, center={psf_cut0[H//2, W//2]:.6g}")

    # Robust flux estimate with inverse-variance weighting (mask out flagged pixels)
    if not np.any(fit_mask):
        fit_mask = np.isfinite(data)
    P   = psf_cut0[fit_mask]
    D   = data[fit_mask]             # [MJy/sr]
    Wgt = ivar[fit_mask]             # 1 / (MJy/sr)^2

    # amplitude_MJysr = (P^T W D)/(P^T W P)
    num = np.sum(P * Wgt * D)
    den = np.sum(P * Wgt * P) + 1e-30
    flux0 = num / den                        # [MJy/sr]
    init_dx, init_dy = 0.0, 0.0
    dprint("[Init seeds] Starting SciPy at the fit-window center: dx0=0.000, dy0=0.000")

    def dxdy_to_centers(dx, dy):
        """
        Return centers corresponding to (dx,dy) in three frames:
        - fit window frame (x_fit, y_fit)
        - full 15x15 cutout frame (x_fullcut, y_fullcut)
        - full image frame (x_fullimg, y_fullimg)
        """
        x_fit = xcut + dx
        y_fit = ycut + dy
        x_fullcut = x_min + x_fit
        y_fullcut = y_min + y_fit
        x_fullimg = x_fullcut + x0
        y_fullimg = y_fullcut + y0
        return x_fit, y_fit, x_fullcut, y_fullcut, x_fullimg, y_fullimg

    # Warm start with SciPy (optimize flux, dx, dy)
    def chi2_scipy(theta):
        f, dx_, dy_ = theta
        m = f * make_psf_cut(dx=dx_, dy=dy_)   # f has units MJy/sr
        return np.sum(((data - m)**2) * ivar * fit_mask)

    theta0 = np.array([flux0, init_dx, init_dy], dtype=float)
    # For safety if flux0 ~ 0:
    if not np.isfinite(theta0[0]) or theta0[0] <= 0:
        theta0[0] = 0.1
    lo = 1e-12  # MJy/sr, avoid sticking at the boundary 0
    hi = max(1000.0 * abs(theta0[0]), 10.0) + 1e-9
    bounds = [(lo, hi), (-dx_prior_half, dx_prior_half), (-dy_prior_half, dy_prior_half)]
    try:
        # 1) Primary attempt: L-BFGS-B (fast, uses gradients via finite-diff)
        opt_list = []
        opt = minimize(chi2_scipy, theta0, method="L-BFGS-B",
                    bounds=bounds, options=dict(maxiter=1000, ftol=1e-12))
        opt_list.append(opt)

        # If L-BFGS-B fails or terminates abnormally, try Powell with a few restarts
        need_fallback = (not opt.success) or ("ABNORMAL_TERMINATION_IN_LNSRCH" in str(opt.message))
        if need_fallback:
            dprint(f"[SciPy warm start] L-BFGS-B fallback: {opt.message}")
            # build a few starting points around theta0 and aperture-centric start
            starts = []
            starts.append(theta0.copy())
            starts.append(np.array([max(theta0[0], 0.1), 0.0, 0.0], float))
            starts.append(np.array([max(theta0[0]*1.2, 0.1), init_dx, init_dy], float))
            starts.append(np.array([max(theta0[0]*0.8, 0.1),
                                    np.clip(init_dx+0.1, -1, 1),
                                    np.clip(init_dy-0.1, -1, 1)], float))
            best_local = None
            for s in starts:
                opt_p = minimize(chi2_scipy, s, method="Powell",
                                bounds=bounds, options=dict(maxiter=2000, xtol=1e-6, ftol=1e-12))
                opt_list.append(opt_p)
                if (best_local is None) or (opt_p.fun < best_local.fun):
                    best_local = opt_p
            opt = best_local

        # Pick the best among all attempts (lowest chi2)
        opt_best = min(opt_list, key=lambda o: o.fun if np.isfinite(o.fun) else np.inf)
        
        # Prepare stored scipy results
        # Determine what method yielded the best results
        psf_scipy_method_used = opt_best.method if hasattr(opt_best, 'method') else "unknown"
        psf_scipy_status = opt_best.message if hasattr(opt_best, 'message') else "-"
        psf_scipy_chi2 = None
        psf_scipy_dof = n_used_in_fit - 3  # Npix - Nparams
        psf_scipy_flux_MJysr = None
        psf_scipy_dx = None
        psf_scipy_dy = None
        psf_scipy_flux_uJy = None
        psf_scipy_xcen_cutout = None
        psf_scipy_ycen_cutout = None
        psf_scipy_xcen_fullim = None
        psf_scipy_ycen_fullim = None
        psf_scipy_ra_deg = None
        psf_scipy_dec_deg = None
        psf_scipy_wv_um = None
        psf_scipy_wv_width_um = None
        psf_scipy_wv_um_err = None
        psf_scipy_wv_width_um_err = None
        psf_scipy_flux_uJy_err = None

        if opt_best.success and np.isfinite(opt_best.fun):
            flux0, init_dx, init_dy = opt_best.x
            
            #Estimate uncertainty on flux from the curvature (2nd derivative) at the minimum
            # d²(chi2)/d(flux)² ≈ (chi2(flux+δ) - 2*chi2(flux) + chi2(flux-δ)) / δ²
            delta = max(0.01 * abs(flux0), 1e-6)
            chi2_p = chi2_scipy([flux0 + delta, init_dx, init_dy])
            chi2_m = chi2_scipy([flux0 - delta, init_dx, init_dy])
            curv = (chi2_p - 2.0 * opt_best.fun + chi2_m) / (delta**2)
            if curv > 0:
                flux_err = np.sqrt(1.0 / curv)
            else:
                flux_err = np.nan  # fallback
            psf_scipy_flux_MJysr_err = flux_err
            psf_scipy_snr = flux0 / flux_err if (np.isfinite(flux_err) and flux_err > 0) else np.nan

            dprint(f"[SciPy warm start] chi2={opt_best.fun:.3f} -> flux0={flux0:.6g} MJy/sr, dx0={init_dx:.3f}, dy0={init_dy:.3f}")
            psf_scipy_chi2 = opt_best.fun
            psf_scipy_flux_MJysr = flux0
            psf_scipy_dx = init_dx
            psf_scipy_dy = init_dy

            if (psf_det_sum is not None) and np.isfinite(psf_det_sum) and psf_det_sum > 0:
                flux0_uJy = flux0 * omega_sr * psf_det_sum * 1e12
                psf_scipy_flux_uJy = flux0_uJy
                psf_scipy_flux_uJy_err = flux_err * omega_sr * psf_det_sum * 1e12
                dprint(f"[SciPy warm start] integrated flux0 ≈ {flux0_uJy:.3g} µJy  (Ω_pix={omega_sr:.3e} sr; ΣPSF_det={psf_det_sum:.4g})")
            else:
                dprint(f"[SciPy warm start] integrated flux0 ≈ {flux0 * omega_sr * 1e12:.3g} µJy  (Ω_pix={omega_sr:.3e} sr; ΣPSF_det unknown)")
            # Report centers derived from (dx,dy)
            x_fit, y_fit, x_fullcut, y_fullcut, x_fullimg, y_fullimg = dxdy_to_centers(init_dx, init_dy)

            psf_scipy_xcen_cutout = x_fullcut
            psf_scipy_ycen_cutout = y_fullcut
            psf_scipy_xcen_fullim = x_fullimg
            psf_scipy_ycen_fullim = y_fullimg

            dprint(f"[SciPy warm start] center (fit window): x={x_fit:.3f}, y={y_fit:.3f} px")
            dprint(f"[SciPy warm start] center (full 15x15): x={x_fullcut:.3f}, y={y_fullcut:.3f} px")
            
            # RA/Dec of the SciPy center
            sky_scipy = wcs.pixel_to_world(x_fullimg, y_fullimg)
            psf_scipy_ra_deg = float(sky_scipy.ra.deg)
            psf_scipy_dec_deg = float(sky_scipy.dec.deg)
            psf_scipy_sep_as = np.sqrt((psf_scipy_ra_deg - ra)**2*np.cos(np.radians(psf_scipy_dec_deg))**2 + (psf_scipy_dec_deg - dec)**2) * 3600.0

            # Determine RA/DEC errors on the scipy center using a 400-elements Monte Carlo
            nmc = 400
            x_mc = np.random.normal(loc=x_fullimg, scale=dx_prior_half/2.0, size=nmc)
            y_mc = np.random.normal(loc=y_fullimg, scale=dy_prior_half/2.0, size=nmc)
            sky_mc = wcs.pixel_to_world(x_mc, y_mc)
            ra_mc = sky_mc.ra.deg
            dec_mc = sky_mc.dec.deg
            ra_sig = np.std(ra_mc) if np.isfinite(ra_mc).all() else np.nan
            dec_sig = np.std(dec_mc) if np.isfinite(dec_mc).all() else np.nan
            if np.isfinite(ra_sig) and np.isfinite(dec_sig):
                dprint(f"[SciPy warm start] RA,Dec errors via MC: σ_RA={ra_sig*3600:.3f}\" , σ_Dec={dec_sig*3600:.3f}\"")
                psf_scipy_ra_err_mas = ra_sig * 3600.0 * 1000.0 * np.cos(np.radians(float(dec)))
                psf_scipy_dec_err_mas = dec_sig * 3600.0 * 1000.0
            else:
                psf_scipy_ra_err_arcsec = None
                psf_scipy_dec_err_arcsec = None
            # Determine RA/DEC covariance in milliarcsec
            psf_scipy_ra_dec_cov_mas2 = (np.cov(ra_mc * np.cos(np.radians(float(dec))), dec_mc)*(3600.0 * 1000.0)**2)[0,1]

            dprint(f"[SciPy warm start] center (RA,Dec): ({float(sky_scipy.ra.deg):.8f}, {float(sky_scipy.dec.deg):.8f}) deg")
            
            # --- spectral WCS diagnostics for SciPy PSF center ---
            try:
                wv_um_sci, wv_width_um_sci = wcswave_eval_xy_fullimg(x_fullimg, y_fullimg)
                (_, wv_sig_sci), (_, wd_sig_sci) = wcswave_sample_stats(
                    x_fullimg, y_fullimg,
                    dx_prior_half, dy_prior_half
                )
                psf_scipy_wv_um = wv_um_sci
                psf_scipy_wv_width_um = wv_width_um_sci
                psf_scipy_wv_um_err = wv_sig_sci
                psf_scipy_wv_width_um_err = wd_sig_sci

                if (np.isfinite(wv_um_sci) and np.isfinite(wv_sig_sci)
                        and np.isfinite(wv_width_um_sci) and np.isfinite(wd_sig_sci)):
                    dprint(f"[WCS-WAVE] SciPy PSF: λ={wv_um_sci:.5f} ± {wv_sig_sci:.5f} µm ; "
                           f"width={wv_width_um_sci:.5f} ± {wd_sig_sci:.5f} µm")
            except Exception as _e:
                dprint(f"[WCS-WAVE] SciPy PSF diagnostics failed: {_e}")
        else:
            dprint(f"[SciPy warm start] did not converge: {opt_best.message}")
    except Exception as e:
        dprint(f"[SciPy warm start] exception: {e}")

    # Cache SciPy warm-start results (even if not perfect), for optional skip of UltraNest
    scipy_flux, scipy_dx, scipy_dy = float(flux0), float(init_dx), float(init_dy)

    # === forward model function ===
    # Returns per-pixel model in MJy/sr; 'flux' here is a surface-brightness amplitude (MJy/sr).
    def model_image(params):
        flux, dx, dy = params
        cut = make_psf_cut(dx=dx, dy=dy)
        return flux * cut    # flux is surface-brightness amplitude in MJy/sr

    # === Optional early return: SciPy-only path ===
    if not use_ultranest:
        if debug:
            dprint("[SciPy-only] Skipping UltraNest; using SciPy warm-start parameters.")
        # Centers in all frames from SciPy params
        x_fit_ml, y_fit_ml, x_fullcut_ml, y_fullcut_ml, x_fullimg_ml, y_fullimg_ml = dxdy_to_centers(scipy_dx, scipy_dy)
        sky_ml = wcs.pixel_to_world(x_fullimg_ml, y_fullimg_ml)
        flux = scipy_flux
        dx = scipy_dx
        dy = scipy_dy
        # Build model and chi2
        model = model_image([flux, dx, dy])
        chi2_val = float(np.sum(((data - model) ** 2) * ivar * fit_mask))
        flux_uJy = float(flux * omega_sr * psf_det_sum * 1e12) if (np.isfinite(omega_sr) and (psf_det_sum is not None) and np.isfinite(psf_det_sum)) else np.nan

        # Estimate 1-sigma uncertainty on amplitude (flux in MJy/sr) using weighted curvature
        P_best = make_psf_cut(dx=dx, dy=dy)[fit_mask]
        W_best = ivar[fit_mask]
        den_best = np.sum(P_best * W_best * P_best) + 1e-30
        sig_flux_MJysr = float(np.sqrt(1.0 / den_best)) if (np.isfinite(den_best) and den_best > 0) else np.nan
        sig_flux_uJy = float(sig_flux_MJysr * omega_sr * psf_det_sum * 1e12) if (np.isfinite(sig_flux_MJysr) and np.isfinite(omega_sr) and (psf_det_sum is not None) and np.isfinite(psf_det_sum)) else np.nan

        # --- spectral WCS for PSF fit (ML) and aperture (model-free) ---
        # Spectral WCS (PSF center via SciPy) -> central wavelength + width
        wv_um_ml, wv_width_um_ml = wcswave_eval_xy_fullimg(x_fullimg_ml, y_fullimg_ml)
        # Uncertainty estimates from dx_sig/dy_sig are unavailable in SciPy-only path;
        # approximate using centroid_err_px as a fallback.
        sigx_ml = dx_prior_half if np.isfinite(dx_prior_half) else 0.5
        sigy_ml = dy_prior_half if np.isfinite(dy_prior_half) else 0.5
        (wv_mu_ml, wv_sig_ml), (wd_mu_ml, wd_sig_ml) = wcswave_sample_stats(x_fullimg_ml, y_fullimg_ml, sigx_ml, sigy_ml)
        if debug and np.isfinite(wv_um_ml) and np.isfinite(wv_sig_ml) and np.isfinite(wv_width_um_ml) and np.isfinite(wd_sig_ml):
            dprint(f"[WCS-WAVE] SciPy-only PSF: λ={wv_um_ml:.5f} ± {wv_sig_ml:.5f} µm ; "
                   f"width={wv_width_um_ml:.5f} ± {wd_sig_ml:.5f} µm")
        # For the aperture (model-free) center, compute only if centroid exists:
        if np.isfinite(best_ap.get('xcen', np.nan)) and np.isfinite(best_ap.get('ycen', np.nan)):
            x_full_ap = best_ap['xcen'] + x0
            y_full_ap = best_ap['ycen'] + y0
            wv_um_ap, wv_width_um_ap = wcswave_eval_xy_fullimg(x_full_ap, y_full_ap)
            # Uncertainty from centroid_err_px
            (wv_mu_ap, wv_sig_ap), (wd_mu_ap, wd_sig_ap) = wcswave_sample_stats(x_full_ap, y_full_ap, centroid_err_px, centroid_err_px)
            if debug and np.isfinite(wv_um_ap) and np.isfinite(wv_sig_ap) and np.isfinite(wv_width_um_ap) and np.isfinite(wd_sig_ap):
                dprint(f"[WCS-WAVE] Aperture (COM): λ={wv_um_ap:.5f} ± {wv_sig_ap:.5f} µm ; "
                       f"width={wv_width_um_ap:.5f} ± {wd_sig_ap:.5f} µm")
        else:
            wv_um_ap = wv_width_um_ap = wv_mu_ap = wv_sig_ap = wd_mu_ap = wd_sig_ap = np.nan

        # Diagnostic plot mirroring the UltraNest result panel
        if debug:
            # peak inside the fit circle, considering only usable pixels
            try:
                peak_data = float(np.nanmax(data[radmask & fit_mask]))
            except Exception:
                # safe fallback if mask is empty or weird
                peak_data = float(np.nanmax(data))
            fig, axs = plt.subplots(1,4,figsize=(16,4))
            im0 = axs[0].imshow(data, origin='lower'); axs[0].scatter([xcut],[ycut], s=30, marker='+')
            axs[0].set_title(f'Data (cutout) peak={peak_data:.3g} MJy/sr')
            im1 = axs[1].imshow(model, origin='lower'); axs[1].set_title(f'Model (SciPy-only) peak={np.nanmax(model):.3g}')
            im2 = axs[2].imshow(data - model, origin='lower'); axs[2].set_title('Residual')
            err = np.sqrt(np.clip(cut_var, 0, np.inf))
            try:
                vmin, vmax = np.nanpercentile(err, (5, 95))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    raise ValueError
            except Exception:
                vmin, vmax = None, None
            im3 = axs[3].imshow(err, origin='lower', vmin=vmin, vmax=vmax); axs[3].set_title('Error = sqrt(variance) [MJy/sr]')
            for ax in axs:
                ax.set_xlim(0, data.shape[1]-1); ax.set_ylim(0, data.shape[0]-1)
            for ax in axs:
                ax.add_patch(plt.Circle((xcut, ycut), fit_radius_px, fill=False, linestyle='--'))
            for ax in (axs[0], axs[1], axs[2], axs[3]):
                ax.plot([x_fit_ml],[y_fit_ml], marker='+', ms=8, color='cyan', linestyle='None', label='SciPy center')
                if np.isfinite(best_ap.get('xcen', np.nan)) and np.isfinite(best_ap.get('ycen', np.nan)):
                    ax.plot([best_ap['xcen'] - x_min], [best_ap['ycen'] - y_min],
                            marker='x', ms=6, color='yellow', linestyle='None', label='Aperture COM')
            handles, labels = axs[0].get_legend_handles_labels()
            if handles:
                axs[0].legend(loc='lower right', fontsize=8, frameon=True)
            plt.tight_layout()
            save_or_show(fig, "scipy_only_result")

        # Assemble outputs (posterior uncertainties not available in SciPy-only path)
        detector_id = int(hdr_img.get('DETECTOR')) if hdr_img.get('DETECTOR') is not None else np.nan
        outputs = FitOutputs(
            input_ra_deg=float(ra) if np.isfinite(ra) else np.nan,
            input_dec_deg=float(dec) if np.isfinite(dec) else np.nan,
            fit_radius_px=float(fit_radius_px) if np.isfinite(fit_radius_px) else np.nan,
            box_size_bcg_subtract=box_size_bcg_subtract,
            no_masking=bool(no_masking),

            # FITS-HEADER RELATED STUFF
            obsid=hdr_img.get('OBSID', None),
            bandpass="D"+str(detector_id) if detector_id is not None else np.nan,
            expid=int(hdr_img.get('EXPIDN')) if hdr_img.get('EXPIDN') is not None else np.nan,
            mjd_avg=float(hdr_img.get('MJD-AVG')) if hdr_img.get('MJD-AVG') is not None else np.nan,
            detector_id=detector_id,
            psf_index=int(idx_psf) if 'idx_psf' in locals() else np.nan,
            omega_sr=float(omega_sr) if np.isfinite(omega_sr) else np.nan,
            px_scale_arcsec=float(px_arcsec) if np.isfinite(px_arcsec) else np.nan,

            # FLAGS
            near_cutout_edge=bool((x_fullcut_ml < 5) or (y_fullcut_ml < 5) or ((w_full - x_fullcut_ml) < 5) or ((h_full - y_fullcut_ml) < 5)),
            near_detector_edge=bool(
                ((x_fullimg_ml + det_origin_x) < 10)
                or ((y_fullimg_ml + det_origin_y) < 10)
                or ((det_w - (x_fullimg_ml + det_origin_x)) < 10)
                or ((det_h - (y_fullimg_ml + det_origin_y)) < 10)
            ),
            near_bcg_star=False,
            n_pix_flagged_in_fit=int(n_flagged_in_fit),
            n_pix_used_in_fit=int(n_used_in_fit),
            n_pix_total_in_fit=int(n_total_in_fit),

            # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
            ap_radius_px=float(best_ap['r']) if np.isfinite(best_ap['r']) else np.nan,
            ap_radius_forced=bool(ap_radius_forced) if 'ap_radius_forced' in locals() else np.nan,
            ap_flux_MJysr=float(best_ap['flux_MJysr']) if np.isfinite(best_ap['flux_MJysr']) else np.nan,
            ap_flux_MJysr_err=float(best_ap['sig_MJysr']) if np.isfinite(best_ap['sig_MJysr']) else np.nan,
            ap_snr=float(best_ap['snr']) if np.isfinite(best_ap['snr']) else np.nan,
            ap_centroid_err_px=float(centroid_err_px) if np.isfinite(centroid_err_px) else np.nan,
            ap_flux_uJy=float(ap_flux_uJy) if np.isfinite(ap_flux_uJy) else np.nan,
            ap_flux_uJy_err=float(ap_flux_uJy_err) if np.isfinite(ap_flux_uJy_err) else np.nan,
            ap_xcen_cutout=float(ap_xcen_cutout) if ap_xcen_cutout is not None and np.isfinite(ap_xcen_cutout) else np.nan,
            ap_ycen_cutout=float(ap_ycen_cutout) if ap_ycen_cutout is not None and np.isfinite(ap_ycen_cutout) else np.nan,
            ap_xcen_fullim=float(ap_xcen_fullimg) if ap_xcen_fullimg is not None and np.isfinite(ap_xcen_fullimg) else np.nan,
            ap_ycen_fullim=float(ap_ycen_fullimg) if ap_ycen_fullimg is not None and np.isfinite(ap_ycen_fullimg) else np.nan,

            # CENTER OF MASS OUTPUTS
            com_xcen_cutout=float(com_xcen_cutout) if com_xcen_cutout is not None and np.isfinite(com_xcen_cutout) else np.nan,
            com_ycen_cutout=float(com_ycen_cutout) if com_ycen_cutout is not None and np.isfinite(com_ycen_cutout) else np.nan,
            com_xcen_fullim=float(com_xcen_fullimg) if com_xcen_fullimg is not None and np.isfinite(com_xcen_fullimg) else np.nan,
            com_ycen_fullim=float(com_ycen_fullimg) if com_ycen_fullimg is not None and np.isfinite(com_ycen_fullimg) else np.nan,
            com_ra_deg=float(com_ra_deg) if 'com_ra_deg' in locals() and com_ra_deg is not None and np.isfinite(com_ra_deg) else np.nan,
            com_dec_deg=float(com_dec_deg) if 'com_dec_deg' in locals() and com_dec_deg is not None and np.isfinite(com_dec_deg) else np.nan,
            com_sep_as=float(com_sep_as) if 'com_sep_as' in locals() and com_sep_as is not None and np.isfinite(com_sep_as) else np.nan,
            com_wv_um=float(com_wv_um) if 'com_wv_um' in locals() and com_wv_um is not None and np.isfinite(com_wv_um) else np.nan,
            com_wv_width_um=float(com_wv_width_um) if 'com_wv_width_um' in locals() and com_wv_width_um is not None and np.isfinite(com_wv_width_um) else np.nan,

            # PSF-SCIPY OUTPUTS
            psf_scipy_method_used=psf_scipy_method_used,
            psf_scipy_status=str(psf_scipy_status) if 'psf_scipy_status' in locals() else np.nan,
            psf_scipy_flux_MJysr=float(psf_scipy_flux_MJysr) if 'psf_scipy_flux_MJysr' in locals() and psf_scipy_flux_MJysr is not None and np.isfinite(psf_scipy_flux_MJysr) else np.nan,
            psf_scipy_flux_MJysr_err=float(psf_scipy_flux_MJysr_err) if 'psf_scipy_flux_MJysr_err' in locals() and psf_scipy_flux_MJysr_err is not None and np.isfinite(psf_scipy_flux_MJysr_err) else np.nan,
            psf_scipy_snr=float(psf_scipy_snr) if 'psf_scipy_snr' in locals() and psf_scipy_snr is not None and np.isfinite(psf_scipy_snr) else np.nan,
            psf_scipy_flux_uJy=float(psf_scipy_flux_uJy) if 'psf_scipy_flux_uJy' in locals() and psf_scipy_flux_uJy is not None and np.isfinite(psf_scipy_flux_uJy) else np.nan,
            psf_scipy_flux_uJy_err=float(psf_scipy_flux_uJy_err) if 'psf_scipy_flux_uJy_err' in locals() and psf_scipy_flux_uJy_err is not None and np.isfinite(psf_scipy_flux_uJy_err) else np.nan,
            psf_scipy_dx=float(psf_scipy_dx) if 'psf_scipy_dx' in locals() and psf_scipy_dx is not None and np.isfinite(psf_scipy_dx) else np.nan,
            psf_scipy_dy=float(psf_scipy_dy) if 'psf_scipy_dy' in locals() and psf_scipy_dy is not None and np.isfinite(psf_scipy_dy) else np.nan,
            psf_scipy_xcen_cutout=float(psf_scipy_xcen_cutout) if 'psf_scipy_xcen_cutout' in locals() and psf_scipy_xcen_cutout is not None and np.isfinite(psf_scipy_xcen_cutout) else np.nan,
            psf_scipy_ycen_cutout=float(psf_scipy_ycen_cutout) if 'psf_scipy_ycen_cutout' in locals() and psf_scipy_ycen_cutout is not None and np.isfinite(psf_scipy_ycen_cutout) else np.nan,
            psf_scipy_xcen_fullim=float(psf_scipy_xcen_fullim) if 'psf_scipy_xcen_fullim' in locals() and psf_scipy_xcen_fullim is not None and np.isfinite(psf_scipy_xcen_fullim) else np.nan,
            psf_scipy_ycen_fullim=float(psf_scipy_ycen_fullim) if 'psf_scipy_ycen_fullim' in locals() and psf_scipy_ycen_fullim is not None and np.isfinite(psf_scipy_ycen_fullim) else np.nan,
            psf_scipy_ra_deg=float(psf_scipy_ra_deg) if 'psf_scipy_ra_deg' in locals() and psf_scipy_ra_deg is not None and np.isfinite(psf_scipy_ra_deg) else np.nan,
            psf_scipy_ra_err_mas=float(psf_scipy_ra_err_mas) if 'psf_scipy_ra_err_mas' in locals() and psf_scipy_ra_err_mas is not None and np.isfinite(psf_scipy_ra_err_mas) else np.nan,
            psf_scipy_dec_deg=float(psf_scipy_dec_deg) if 'psf_scipy_dec_deg' in locals() and psf_scipy_dec_deg is not None and np.isfinite(psf_scipy_dec_deg) else np.nan,
            psf_scipy_dec_err_mas=float(psf_scipy_dec_err_mas) if 'psf_scipy_dec_err_mas' in locals() and psf_scipy_dec_err_mas is not None and np.isfinite(psf_scipy_dec_err_mas) else np.nan,
            psf_scipy_ra_dec_cov_mas2=psf_scipy_ra_dec_cov_mas2 if 'psf_scipy_ra_dec_cov_mas2' in locals() else np.nan,
            psf_scipy_sep_as=float(psf_scipy_sep_as) if 'psf_scipy_sep_as' in locals() and psf_scipy_sep_as is not None and np.isfinite(psf_scipy_sep_as) else np.nan,
            psf_scipy_wv_um=float(psf_scipy_wv_um) if 'psf_scipy_wv_um' in locals() and psf_scipy_wv_um is not None and np.isfinite(psf_scipy_wv_um) else np.nan,
            psf_scipy_wv_um_err=float(psf_scipy_wv_um_err) if 'psf_scipy_wv_um_err' in locals() and psf_scipy_wv_um_err is not None and np.isfinite(psf_scipy_wv_um_err) else np.nan,
            psf_scipy_wv_width_um=float(psf_scipy_wv_width_um) if 'psf_scipy_wv_width_um' in locals() and psf_scipy_wv_width_um is not None and np.isfinite(psf_scipy_wv_width_um) else np.nan,
            psf_scipy_wv_width_um_err=float(psf_scipy_wv_width_um_err) if 'psf_scipy_wv_width_um_err' in locals() and psf_scipy_wv_width_um_err is not None and np.isfinite(psf_scipy_wv_width_um_err) else np.nan,
            psf_scipy_chi2=float(psf_scipy_chi2) if 'psf_scipy_chi2' in locals() and psf_scipy_chi2 is not None and np.isfinite(psf_scipy_chi2) else np.nan,
            psf_scipy_dof=int(psf_scipy_dof) if 'psf_scipy_dof' in locals() and psf_scipy_dof is not None else np.nan,

            # PSF-ULTRANEST OUTPUTS (not available in SciPy-only mode)
            psf_un_flux_MJysr=None,
            psf_un_flux_MJysr_err=None,
            psf_un_snr=None,
            psf_un_flux_uJy=None,
            psf_un_flux_uJy_err=None,
            psf_un_dx=None,
            psf_un_dy=None,
            psf_un_xcen_cutout=None,
            psf_un_ycen_cutout=None,
            psf_un_xcen_fullim=None,
            psf_un_ycen_fullim=None,
            psf_un_xcen_err=None,
            psf_un_ycen_err=None,
            psf_un_ra_deg=None,
            psf_un_ra_err_mas=None,
            psf_un_dec_deg=None,
            psf_un_dec_err_mas=None,
            psf_un_sep_as=None,
            psf_un_ra_dec_cov_mas2=None,
            psf_un_wv_um=None,
            psf_un_wv_um_err=None,
            psf_un_wv_width_um=None,
            psf_un_wv_width_um_err=None,
            psf_un_chi2=None,
            psf_un_dof=None,
        )
        result_dict = asdict(outputs)

        if save_results:
            out_path = Path(results_path) if results_path else Path(f"{fits_path}.results.json")
            with open(out_path, "w") as f:
                _json.dump(result_dict, f, indent=2)
            if debug:
                print(f"[INFO] Results saved to {out_path}")

        return result_dict

    # --- quick visualization of SciPy warm start (before UltraNest) ---
    if debug:
        try:
            model0 = model_image([flux0, init_dx, init_dy])
            resid0 = data - model0
            # peak inside the fit circle, considering only usable pixels
            try:
                peak_data = float(np.nanmax(data[radmask & fit_mask]))
            except Exception:
                # safe fallback if mask is empty or weird
                peak_data = float(np.nanmax(data))
            peak_model0 = float(np.nanmax(model0))
            dprint(f"[SciPy warm start] Peak(data)={peak_data:.3g} MJy/sr ; Peak(model0)={peak_model0:.3g} MJy/sr")
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            im0 = axs[0].imshow(data, origin='lower'); axs[0].scatter([xcut],[ycut], s=30, marker='+')
            axs[0].set_title(f'Data (cutout) peak={peak_data:.3g} MJy/sr')
            im1 = axs[1].imshow(model0, origin='lower')
            axs[1].set_title(f'SciPy model  peak={peak_model0:.3g}')
            im2 = axs[2].imshow(resid0, origin='lower')
            axs[2].set_title('Residual (data - model)')
            # error panel (sqrt of variance)
            err = np.sqrt(np.clip(cut_var, 0, np.inf))
            try:
                vmin, vmax = np.nanpercentile(err, (5, 95))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                    raise ValueError
            except Exception:
                vmin, vmax = None, None
            im3 = axs[3].imshow(err, origin='lower', vmin=vmin, vmax=vmax)
            axs[3].set_title('Error = sqrt(variance) [MJy/sr]')
            for ax in axs:
                ax.set_xlim(0, data.shape[1]-1); ax.set_ylim(0, data.shape[0]-1)
            for ax in axs:
                ax.add_patch(plt.Circle((xcut, ycut), fit_radius_px, fill=False, linestyle='--', label='Fit radius'))
            # mark SciPy center and aperture centroid
            x_fit_sc, y_fit_sc, *_ = dxdy_to_centers(init_dx, init_dy)
            for ax in (axs[0], axs[1], axs[2], axs[3]):
                ax.plot([x_fit_sc],[y_fit_sc], marker='+', ms=8, color='cyan', linestyle='None', label='SciPy center')
                if np.isfinite(best_ap.get('xcen', np.nan)) and np.isfinite(best_ap.get('ycen', np.nan)):
                    # convert aperture centroid from full 15x15 into fit-window frame
                    ax.plot([best_ap['xcen'] - x_min], [best_ap['ycen'] - y_min],
                            marker='x', ms=6, color='yellow', linestyle='None', label='Aperture COM')
            handles, labels = axs[0].get_legend_handles_labels()
            if handles:
                axs[0].legend(loc='lower right', fontsize=8, frameon=True)
            plt.tight_layout()
            save_or_show(fig, "scipy_warm")
        except Exception as e:
            dprint(f"[SciPy warm start] preview plot failed: {e}")

    # Prior width: keep very broad to avoid truncation issues
    flux_hi = max(3.0*abs(flux0), 1.0)   # MJy

    if debug:
        dprint(f"[ultranest] Initial guess flux0 ~ {flux0:.6g} [MJy/sr] (matched filter / SciPy)")
        dprint(f"[ultranest] Initial guess dx,dy = ({init_dx:.3f}, {init_dy:.3f}) [px]")
        dprint(f"[ultranest] Setting uniform prior for flux in [0, {flux_hi:.6g}] MJy and dx in ±{dx_prior_half:.3f} px, dy in ±{dy_prior_half:.3f} px")
        #if use_ultranest:
        #    time.sleep(2.0)  # short pause so the seeds are visible before the sampler starts

    # === prior transform ===
    def prior_transform(cube):
        # cube in [0,1]^3
        flux = cube[0] * flux_hi                      # MJy
        dx = (cube[1] - 0.5) * 2.0 * dx_prior_half    # [-dx_prior_half, +dx_prior_half] px
        dy = (cube[2] - 0.5) * 2.0 * dy_prior_half
        return [flux, dx, dy]

    # === log-likelihood ===
    def loglike(params):
        model = model_image(params)
        chi2 = np.sum(((data - model)**2) * ivar * fit_mask)
        return -0.5*chi2

    sampler = ultranest.ReactiveNestedSampler(['flux','dx','dy'], loglike, prior_transform)
    if debug:
        dprint("About to launch UltraNest with initial guess:")
        dprint(f"  flux0={flux0:.6g} [MJy/sr], dx0={init_dx:.3f} px, dy0={init_dy:.3f} px")
        # quick peak comparison
        peak_data = float(np.nanmax(data))
        peak_model0 = float(np.nanmax(flux0 * psf_cut0))
        dprint(f"  Peak(data)={peak_data:.3g} MJy/sr ; Peak(model at init)={peak_model0:.3g} MJy/sr")
        dprint("  (If off by ~factor, PSF normalization or Omega_pix might differ.)")
        x_fit0, y_fit0, x_fullcut0, y_fullcut0, x_fullimg0, y_fullimg0 = dxdy_to_centers(init_dx, init_dy)
        dprint(f"  Initial center (fit window): x={x_fit0:.3f}, y={y_fit0:.3f} px")
        dprint(f"  Initial center (full 15x15): x={x_fullcut0:.3f}, y={y_fullcut0:.3f} px")
        dprint(f"  dx prior: ±{dx_prior_half:.3f} px ; dy prior: ±{dy_prior_half:.3f} px")

    if ultranest_quiet:
        _sink = io.StringIO()
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            result = sampler.run(min_num_live_points=200, dlogz=0.1)
    else:
        result = sampler.run(min_num_live_points=200, dlogz=0.1)

    best = result['maximum_likelihood']['point']
    flux, dx, dy = best
    
    #Aliases
    psf_un_flux_MJysr = flux
    psf_un_dx = dx
    psf_un_dy = dy
    psf_un_chi2 = result['maximum_likelihood']['logl'] * -2.0
    psf_un_dof = int(np.sum(fit_mask) - 3)

    # Report UltraNest ML centers in all frames
    x_fit_ml, y_fit_ml, x_fullcut_ml, y_fullcut_ml, x_fullimg_ml, y_fullimg_ml = dxdy_to_centers(dx, dy)
    sky_ml = wcs.pixel_to_world(x_fullimg_ml, y_fullimg_ml)
    
    psf_un_xcen_cutout = x_fullcut_ml
    psf_un_ycen_cutout = y_fullcut_ml
    psf_un_xcen_fullim = x_fullimg_ml
    psf_un_ycen_fullim = y_fullimg_ml
    psf_un_ra_deg = float(sky_ml.ra.deg)
    psf_un_dec_deg = float(sky_ml.dec.deg)
    psf_un_sep_as = np.sqrt((psf_un_ra_deg - ra)**2*np.cos(np.radians(psf_un_dec_deg))**2 + (psf_un_dec_deg - dec)**2) * 3600.0

    if debug:
        dprint(f"UltraNest ML center (fit window): x={x_fit_ml:.3f}, y={y_fit_ml:.3f} px")
        dprint(f"UltraNest ML center (full 15x15): x={x_fullcut_ml:.3f}, y={y_fullcut_ml:.3f} px")
        dprint(f"UltraNest ML center (RA,Dec): ({float(sky_ml.ra.deg):.8f}, {float(sky_ml.dec.deg):.8f}) deg")

    # --- spectral WCS at UltraNest ML center ---
    wv_um_ml, wv_width_um_ml = wcswave_eval_xy_fullimg(x_fullimg_ml, y_fullimg_ml)
    
    psf_un_wv_um = wv_um_ml
    psf_un_wv_width_um = wv_width_um_ml

    # Uncertainties: prefer posterior samples if available; otherwise fall back to dx_sig/dy_sig
    wv_mu_ml = wv_um_ml; wd_mu_ml = wv_width_um_ml
    wv_sig_ml = np.nan;  wd_sig_ml = np.nan
    if 'samples' in locals() and samples is not None and samples.size and posterior_keep:
        ns = min(samples.shape[0], int(posterior_keep))
        xs = np.empty(ns); ys = np.empty(ns)
        for k in range(ns):
            dxk = samples[k,1]; dyk = samples[k,2]
            x_fit_k, y_fit_k, x_fullcut_k, y_fullcut_k, x_fullimg_k, y_fullimg_k = dxdy_to_centers(dxk, dyk)
            xs[k], ys[k] = x_fullimg_k, y_fullimg_k
        lam_s = np.empty(ns); dlam_s = np.empty(ns)
        for k in range(ns):
            lam_s[k], dlam_s[k] = wcswave_eval_xy_fullimg(xs[k], ys[k])
        wv_sig_ml = float(np.nanstd(lam_s))
        wd_sig_ml = float(np.nanstd(dlam_s))
    else:
        # fall back to reported 1-sigma in dx,dy if available, else 0.5 px
        sigx_ml = 0.5
        sigy_ml = 0.5
        (wv_stats, wd_stats) = wcswave_sample_stats(x_fullimg_ml, y_fullimg_ml, sigx_ml, sigy_ml)
        # wv_stats = (mean_lambda, std_lambda); wd_stats = (mean_width, std_width)
        wv_mu_ml, wv_sig_ml = float(wv_stats[0]), float(wv_stats[1])
        wd_mu_ml, wd_sig_ml = float(wd_stats[0]), float(wd_stats[1])
    if debug and np.isfinite(wv_um_ml) and np.isfinite(wv_sig_ml) and np.isfinite(wv_width_um_ml) and np.isfinite(wd_sig_ml):
        dprint(f"[WCS-WAVE] UltraNest ML: λ={wv_um_ml:.5f} ± {wv_sig_ml:.5f} µm ; "
               f"width={wv_width_um_ml:.5f} ± {wd_sig_ml:.5f} µm")

    # For the aperture (model-free) center, mirror the SciPy-only logic
    if np.isfinite(best_ap.get('xcen', np.nan)) and np.isfinite(best_ap.get('ycen', np.nan)):
        x_full_ap = best_ap['xcen'] + x0
        y_full_ap = best_ap['ycen'] + y0
        wv_um_ap, wv_width_um_ap = wcswave_eval_xy_fullimg(x_full_ap, y_full_ap)
        (wv_mu_ap, wv_sig_ap), (wd_mu_ap, wd_sig_ap) = wcswave_sample_stats(x_full_ap, y_full_ap, centroid_err_px, centroid_err_px)
        if debug and np.isfinite(wv_um_ap) and np.isfinite(wv_sig_ap) and np.isfinite(wv_width_um_ap) and np.isfinite(wd_sig_ap):
            dprint(f"[WCS-WAVE] Aperture (COM): λ={wv_um_ap:.5f} ± {wv_sig_ap:.5f} µm ; "
                   f"width={wv_width_um_ap:.5f} ± {wd_sig_ap:.5f} µm")
    else:
        wv_um_ap = wv_width_um_ap = wv_mu_ap = wv_sig_ap = wd_mu_ap = wd_sig_ap = np.nan

    # Convert amplitude (MJy/sr) to integrated flux at the end
    flux_uJy = flux * omega_sr * psf_det_sum * 1e12  # integrated F = amplitude * Ω_pix * ΣPSF_det
    psf_un_flux_uJy = flux_uJy
    if debug:
        dprint(f"Flux ML (integrated): {(flux*omega_sr*psf_det_sum):.6g} MJy = {flux_uJy:.3g} µJy  (Ω_pix={omega_sr:.3e} sr; ΣPSF_det={psf_det_sum:.4g})")

    if debug:
        dprint(f"UltraNest ML point: flux={best[0]:.6g} MJy/sr, dx={best[1]:.3f} px, dy={best[2]:.3f} px")

    # posterior uncertainties
    # try multiple possible locations for samples and weights in ultranest results
    post = result.get('posterior', {}) if isinstance(result, dict) else {}
    samples = post.get('samples') if isinstance(post, dict) else np.nan
    weights = post.get('weights') if isinstance(post, dict) else np.nan
    if samples is None:
        samples = result.get('samples')
        weights = result.get('weights')

    if samples is not None:
        samples = np.array(samples)  # shape (N, 3) in order [flux, dx, dy]
        if weights is None:
            weights = np.ones(samples.shape[0])
        # compute median and 16-84% ranges
        med = np.array([weighted_quantile(samples[:,i], 0.5, weights) for i in range(samples.shape[1])])
        p16 = np.array([weighted_quantile(samples[:,i], 0.16, weights) for i in range(samples.shape[1])])
        p84 = np.array([weighted_quantile(samples[:,i], 0.84, weights) for i in range(samples.shape[1])])
        sig = 0.5*(p84 - p16)
        dprint("Posterior medians [flux[MJy], dx[px], dy[px]]:", med)
        dprint("Posterior 1-sigma widths:", sig)

        #Transform dx, dy samples into ra, dec
        ns = samples.shape[0]
        ras = np.empty(ns); decs = np.empty(ns)
        for k in range(ns):
            dxk = samples[k,1]; dyk = samples[k,2]
            _, _, _, _, x_fullimg_k, y_fullimg_k = dxdy_to_centers(dxk, dyk)
            sky_k = wcs.pixel_to_world(x_fullimg_k, y_fullimg_k)
            ras[k], decs[k] = float(sky_k.ra.deg), float(sky_k.dec.deg)
        ra_med = weighted_quantile(ras, 0.5, weights)
        ra_p16 = weighted_quantile(ras, 0.16, weights)
        ra_p84 = weighted_quantile(ras, 0.84, weights)
        dec_med = weighted_quantile(decs, 0.5, weights)
        dec_p16 = weighted_quantile(decs, 0.16, weights)
        dec_p84 = weighted_quantile(decs, 0.84, weights)
        ra_sig = 0.5 * (ra_p84 - ra_p16)*np.cos(np.radians(dec_med)) * 3600.0 * 1000.0  # in mas
        dec_sig = 0.5 * (dec_p84 - dec_p16) * 3600.0 * 1000.0  # in mas
        ra_dec_cov = (np.cov(ras*np.cos(np.radians(dec)), decs, aweights=weights) * (3600.0*1000.0)**2)[0,1]  # in mas^2
        dprint(f"Posterior RA,Dec median = ({ra_med:.8f}, {dec_med:.8f}) deg ; 1-sigma = ({ra_sig:.4g}, {dec_sig:.4g}) deg")

        #Transform dx, dy samples into wv, wv_width
        wvs = np.empty(ns); wds = np.empty(ns)
        for k in range(ns):
            dxk = samples[k,1]; dyk = samples[k,2]
            _, _, _, _, x_fullimg_k, y_fullimg_k = dxdy_to_centers(dxk, dyk)
            wvs[k], wds[k] = wcswave_eval_xy_fullimg(x_fullimg_k, y_fullimg_k)
        wv_med = weighted_quantile(wvs, 0.5, weights)
        wv_p16 = weighted_quantile(wvs, 0.16, weights)
        wv_p84 = weighted_quantile(wvs, 0.84, weights)
        wv_sig = 0.5 * (wv_p84 - wv_p16)
        wd_med = weighted_quantile(wds, 0.5, weights)
        wd_p16 = weighted_quantile(wds, 0.16, weights)
        wd_p84 = weighted_quantile(wds, 0.84, weights)
        wd_sig = 0.5 * (wd_p84 - wd_p16)
        dprint(f"Posterior λ, width median = ({wv_med:.5f}, {wd_med:.5f}) µm ; 1-sigma = ({wv_sig:.5f}, {wd_sig:.5f}) µm")

    else:
        dprint("UltraNest: posterior samples not found in result; uncertainties not computed.")

    psf_un_flux_MJysr_err = float(sig[0]) if ('sig' in locals() and len(sig) >= 1 and np.isfinite(sig[0])) else np.nan
    psf_un_xcen_err = float(sig[1]) if ('sig' in locals() and len(sig) >= 2 and np.isfinite(sig[1])) else np.nan
    psf_un_ycen_err = float(sig[2]) if ('sig' in locals() and len(sig) >= 3 and np.isfinite(sig[2])) else np.nan
    psf_un_flux_uJy_err = (float(sig[0]) * float(omega_sr) * float(psf_det_sum) * 1e12) if ('sig' in locals() and len(sig) >= 1 and np.isfinite(sig[0]) and np.isfinite(omega_sr) and (psf_det_sum is not None) and np.isfinite(psf_det_sum)) else np.nan
    psf_un_snr = float(flux) / float(sig[0]) if ('sig' in locals() and len(sig) >= 1 and np.isfinite(sig[0]) and sig[0] > 0) else np.nan
    psf_un_ra_err_mas = ra_sig if ('ra_sig' in locals() and np.isfinite(ra_sig)) else np.nan
    psf_un_dec_err_mas = dec_sig if ('dec_sig' in locals() and np.isfinite(dec_sig)) else np.nan
    psf_un_ra_dec_cov_mas2 = ra_dec_cov if ('ra_dec_cov' in locals() and ra_dec_cov is not None and np.all(np.isfinite(ra_dec_cov))) else np.nan
    psf_un_wv_um_err = wv_sig if ('wv_sig' in locals() and np.isfinite(wv_sig)) else np.nan
    psf_un_wv_width_um_err = wd_sig if ('wd_sig' in locals() and np.isfinite(wd_sig)) else np.nan

    if debug:
        model = model_image(best)
        resid = data - model
        peak_data = float(np.nanmax(data))
        fig, axs = plt.subplots(1,4,figsize=(16,4))
        im0 = axs[0].imshow(data, origin='lower'); axs[0].scatter([xcut],[ycut], s=30, marker='+')
        axs[0].set_title(f'Data (cutout) peak={peak_data:.3g} MJy/sr')
        im1 = axs[1].imshow(model, origin='lower'); axs[1].set_title(f'Model (PSF idx {idx_psf})  peak={np.nanmax(model):.3g}')
        im2 = axs[2].imshow(resid, origin='lower'); axs[2].set_title('Residual')
        # error panel (sqrt of variance)
        err = np.sqrt(np.clip(cut_var, 0, np.inf))
        try:
            vmin, vmax = np.nanpercentile(err, (5, 95))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                raise ValueError
        except Exception:
            vmin, vmax = None, None
        im3 = axs[3].imshow(err, origin='lower', vmin=vmin, vmax=vmax)
        axs[3].set_title('Error = sqrt(variance) [MJy/sr]')
        for ax in axs:
            ax.set_xlim(0, data.shape[1]-1); ax.set_ylim(0, data.shape[0]-1)
        for ax in axs:
            ax.add_patch(plt.Circle((xcut, ycut), fit_radius_px, fill=False, linestyle='--'))
        # mark ML center and aperture centroid on data/model/residual/error panels
        for ax in (axs[0], axs[1], axs[2], axs[3]):
            ax.plot([x_fit_ml],[y_fit_ml], marker='+', ms=8, color='cyan', linestyle='None', label='UltraNest ML')
            if np.isfinite(best_ap.get('xcen', np.nan)) and np.isfinite(best_ap.get('ycen', np.nan)):
                ax.plot([best_ap['xcen'] - x_min], [best_ap['ycen'] - y_min],
                        marker='x', ms=6, color='yellow', linestyle='None', label='Aperture COM')
        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            axs[0].legend(loc='lower right', fontsize=8, frameon=True)
        plt.tight_layout()
        save_or_show(fig, "ultranest_result")

    print(f"Best-fit amplitude [MJy/sr]: {flux:.6g}")
    print(f"Best-fit offset dx,dy [px]: {dx:.4f}, {dy:.4f}")

    # At the very end, assemble a FitOutputs dataclass
    # (values shown below are placeholders—use your real variables):
    n_pix_flagged_in_fit = int(n_flagged_in_fit)
    n_pix_total_in_fit = int(n_total_in_fit)
    n_pix_used_in_fit = int(n_used_in_fit)
    detector_id = int(hdr_img.get('DETECTOR')) if hdr_img.get('DETECTOR') is not None else np.nan
    outputs = FitOutputs(
        input_ra_deg=float(ra) if np.isfinite(ra) else np.nan,
        input_dec_deg=float(dec) if np.isfinite(dec) else np.nan,
        fit_radius_px=float(fit_radius_px) if np.isfinite(fit_radius_px) else np.nan,
        box_size_bcg_subtract=box_size_bcg_subtract,
        no_masking=bool(no_masking),

        # FITS-HEADER RELATED STUFF
        obsid=hdr_img.get('OBSID', None),
        bandpass="D"+str(detector_id) if detector_id is not None else np.nan,
        expid=int(hdr_img.get('EXPIDN')) if hdr_img.get('EXPIDN') is not None else np.nan,
        mjd_avg=float(hdr_img.get('MJD-AVG')) if hdr_img.get('MJD-AVG') is not None else np.nan,
        detector_id=detector_id,
        psf_index=int(idx_psf) if 'idx_psf' in locals() else np.nan,
        omega_sr=float(omega_sr) if np.isfinite(omega_sr) else np.nan,
        px_scale_arcsec=float(px_arcsec) if np.isfinite(px_arcsec) else np.nan,

        # FLAGS
        near_cutout_edge=bool((x_fullcut_ml < 5) or (y_fullcut_ml < 5) or ((w_full - x_fullcut_ml) < 5) or ((h_full - y_fullcut_ml) < 5)),
        near_detector_edge=bool(
            ((x_fullimg_ml + det_origin_x) < 10)
            or ((y_fullimg_ml + det_origin_y) < 10)
            or ((det_w - (x_fullimg_ml + det_origin_x)) < 10)
            or ((det_h - (y_fullimg_ml + det_origin_y)) < 10)
        ),
        near_bcg_star=False,
        n_pix_flagged_in_fit=int(n_flagged_in_fit),
        n_pix_used_in_fit=int(n_used_in_fit),
        n_pix_total_in_fit=int(n_total_in_fit),

        # APERTURE PHOTOMETRY AND CENTER-OF-MASS (MODEL-FREE)
        ap_radius_px=float(best_ap['r']) if np.isfinite(best_ap['r']) else np.nan,
        ap_radius_forced=bool(ap_radius_forced) if 'ap_radius_forced' in locals() else np.nan,
        ap_flux_MJysr=float(best_ap['flux_MJysr']) if np.isfinite(best_ap['flux_MJysr']) else np.nan,
        ap_flux_MJysr_err=float(best_ap['sig_MJysr']) if np.isfinite(best_ap['sig_MJysr']) else np.nan,
        ap_snr=float(best_ap['snr']) if np.isfinite(best_ap['snr']) else np.nan,
        ap_centroid_err_px=float(centroid_err_px) if np.isfinite(centroid_err_px) else np.nan,
        ap_flux_uJy=float(ap_flux_uJy) if np.isfinite(ap_flux_uJy) else np.nan,
        ap_flux_uJy_err=float(ap_flux_uJy_err) if np.isfinite(ap_flux_uJy_err) else np.nan,
        ap_xcen_cutout=float(ap_xcen_cutout) if ap_xcen_cutout is not None and np.isfinite(ap_xcen_cutout) else np.nan,
        ap_ycen_cutout=float(ap_ycen_cutout) if ap_ycen_cutout is not None and np.isfinite(ap_ycen_cutout) else np.nan,
        ap_xcen_fullim=float(ap_xcen_fullimg) if ap_xcen_fullimg is not None and np.isfinite(ap_xcen_fullimg) else np.nan,
        ap_ycen_fullim=float(ap_ycen_fullimg) if ap_ycen_fullimg is not None and np.isfinite(ap_ycen_fullimg) else np.nan,

        # CENTER OF MASS OUTPUTS
        com_xcen_cutout=float(com_xcen_cutout) if com_xcen_cutout is not None and np.isfinite(com_xcen_cutout) else np.nan,
        com_ycen_cutout=float(com_ycen_cutout) if com_ycen_cutout is not None and np.isfinite(com_ycen_cutout) else np.nan,
        com_xcen_fullim=float(com_xcen_fullimg) if com_xcen_fullimg is not None and np.isfinite(com_xcen_fullimg) else np.nan,
        com_ycen_fullim=float(com_ycen_fullimg) if com_ycen_fullimg is not None and np.isfinite(com_ycen_fullimg) else np.nan,
        com_ra_deg=float(com_ra_deg) if 'com_ra_deg' in locals() and com_ra_deg is not None and np.isfinite(com_ra_deg) else np.nan,
        com_dec_deg=float(com_dec_deg) if 'com_dec_deg' in locals() and com_dec_deg is not None and np.isfinite(com_dec_deg) else np.nan,
        com_sep_as=float(com_sep_as) if 'com_sep_as' in locals() and com_sep_as is not None and np.isfinite(com_sep_as) else np.nan,
        com_wv_um=float(com_wv_um) if 'com_wv_um' in locals() and com_wv_um is not None and np.isfinite(com_wv_um) else np.nan,
        com_wv_width_um=float(com_wv_width_um) if 'com_wv_width_um' in locals() and com_wv_width_um is not None and np.isfinite(com_wv_width_um) else np.nan,

        # PSF-SCIPY OUTPUTS
        psf_scipy_method_used=psf_scipy_method_used if 'psf_scipy_method_used' in locals() else np.nan,
        psf_scipy_status=str(psf_scipy_status) if 'psf_scipy_status' in locals() else np.nan,
        psf_scipy_flux_MJysr=float(psf_scipy_flux_MJysr) if 'psf_scipy_flux_MJysr' in locals() and psf_scipy_flux_MJysr is not None and np.isfinite(psf_scipy_flux_MJysr) else np.nan,
        psf_scipy_flux_MJysr_err=float(psf_scipy_flux_MJysr_err) if 'psf_scipy_flux_MJysr_err' in locals() and psf_scipy_flux_MJysr_err is not None and np.isfinite(psf_scipy_flux_MJysr_err) else np.nan,
        psf_scipy_snr=float(psf_scipy_snr) if 'psf_scipy_snr' in locals() and psf_scipy_snr is not None and np.isfinite(psf_scipy_snr) else np.nan,
        psf_scipy_flux_uJy=float(psf_scipy_flux_uJy) if 'psf_scipy_flux_uJy' in locals() and psf_scipy_flux_uJy is not None and np.isfinite(psf_scipy_flux_uJy) else np.nan,
        psf_scipy_flux_uJy_err=float(psf_scipy_flux_uJy_err) if 'psf_scipy_flux_uJy_err' in locals() and psf_scipy_flux_uJy_err is not None and np.isfinite(psf_scipy_flux_uJy_err) else np.nan,
        psf_scipy_dx=float(psf_scipy_dx) if 'psf_scipy_dx' in locals() and psf_scipy_dx is not None and np.isfinite(psf_scipy_dx) else np.nan,
        psf_scipy_dy=float(psf_scipy_dy) if 'psf_scipy_dy' in locals() and psf_scipy_dy is not None and np.isfinite(psf_scipy_dy) else np.nan,
        psf_scipy_xcen_cutout=float(psf_scipy_xcen_cutout) if 'psf_scipy_xcen_cutout' in locals() and psf_scipy_xcen_cutout is not None and np.isfinite(psf_scipy_xcen_cutout) else np.nan,
        psf_scipy_ycen_cutout=float(psf_scipy_ycen_cutout) if 'psf_scipy_ycen_cutout' in locals() and psf_scipy_ycen_cutout is not None and np.isfinite(psf_scipy_ycen_cutout) else np.nan,
        psf_scipy_xcen_fullim=float(psf_scipy_xcen_fullim) if 'psf_scipy_xcen_fullim' in locals() and psf_scipy_xcen_fullim is not None and np.isfinite(psf_scipy_xcen_fullim) else np.nan,
        psf_scipy_ycen_fullim=float(psf_scipy_ycen_fullim) if 'psf_scipy_ycen_fullim' in locals() and psf_scipy_ycen_fullim is not None and np.isfinite(psf_scipy_ycen_fullim) else np.nan,
        psf_scipy_ra_deg=float(psf_scipy_ra_deg) if 'psf_scipy_ra_deg' in locals() and psf_scipy_ra_deg is not None and np.isfinite(psf_scipy_ra_deg) else np.nan,
        psf_scipy_ra_err_mas=float(psf_scipy_ra_err_mas) if 'psf_scipy_ra_err_mas' in locals() and psf_scipy_ra_err_mas is not None and np.isfinite(psf_scipy_ra_err_mas) else np.nan,
        psf_scipy_dec_deg=float(psf_scipy_dec_deg) if 'psf_scipy_dec_deg' in locals() and psf_scipy_dec_deg is not None and np.isfinite(psf_scipy_dec_deg) else np.nan,
        psf_scipy_dec_err_mas=float(psf_scipy_dec_err_mas) if 'psf_scipy_dec_err_mas' in locals() and psf_scipy_dec_err_mas is not None and np.isfinite(psf_scipy_dec_err_mas) else np.nan,
        psf_scipy_ra_dec_cov_mas2=psf_scipy_ra_dec_cov_mas2 if 'psf_scipy_ra_dec_cov_mas2' in locals() else np.nan,
        psf_scipy_sep_as=float(psf_scipy_sep_as) if 'psf_scipy_sep_as' in locals() and psf_scipy_sep_as is not None and np.isfinite(psf_scipy_sep_as) else np.nan,
        psf_scipy_wv_um=float(psf_scipy_wv_um) if 'psf_scipy_wv_um' in locals() and psf_scipy_wv_um is not None and np.isfinite(psf_scipy_wv_um) else np.nan,
        psf_scipy_wv_um_err=float(psf_scipy_wv_um_err) if 'psf_scipy_wv_um_err' in locals() and psf_scipy_wv_um_err is not None and np.isfinite(psf_scipy_wv_um_err) else np.nan,
        psf_scipy_wv_width_um=float(psf_scipy_wv_width_um) if 'psf_scipy_wv_width_um' in locals() and psf_scipy_wv_width_um is not None and np.isfinite(psf_scipy_wv_width_um) else np.nan,
        psf_scipy_wv_width_um_err=float(psf_scipy_wv_width_um_err) if 'psf_scipy_wv_width_um_err' in locals() and psf_scipy_wv_width_um_err is not None and np.isfinite(psf_scipy_wv_width_um_err) else np.nan,
        psf_scipy_chi2=float(psf_scipy_chi2) if 'psf_scipy_chi2' in locals() and psf_scipy_chi2 is not None and np.isfinite(psf_scipy_chi2) else np.nan,
        psf_scipy_dof=int(psf_scipy_dof) if 'psf_scipy_dof' in locals() and psf_scipy_dof is not None else np.nan,

        # PSF-ULTRANEST OUTPUTS
        psf_un_flux_MJysr=float(psf_un_flux_MJysr) if 'psf_un_flux_MJysr' in locals() and psf_un_flux_MJysr is not None and np.isfinite(psf_un_flux_MJysr) else np.nan,
        psf_un_flux_MJysr_err=float(psf_un_flux_MJysr_err) if 'psf_un_flux_MJysr_err' in locals() and psf_un_flux_MJysr_err is not None and np.isfinite(psf_un_flux_MJysr_err) else np.nan,
        psf_un_snr=float(psf_un_snr) if 'psf_un_snr' in locals() and psf_un_snr is not None and np.isfinite(psf_un_snr) else np.nan,
        psf_un_flux_uJy=float(psf_un_flux_uJy) if 'psf_un_flux_uJy' in locals() and psf_un_flux_uJy is not None and np.isfinite(psf_un_flux_uJy) else np.nan,
        psf_un_flux_uJy_err=float(psf_un_flux_uJy_err) if 'psf_un_flux_uJy_err' in locals() and psf_un_flux_uJy_err is not None and np.isfinite(psf_un_flux_uJy_err) else np.nan,
        psf_un_dx=float(psf_un_dx) if 'psf_un_dx' in locals() and psf_un_dx is not None and np.isfinite(psf_un_dx) else np.nan,
        psf_un_dy=float(psf_un_dy) if 'psf_un_dy' in locals() and psf_un_dy is not None and np.isfinite(psf_un_dy) else np.nan,
        psf_un_xcen_cutout=float(psf_un_xcen_cutout) if 'psf_un_xcen_cutout' in locals() and psf_un_xcen_cutout is not None and np.isfinite(psf_un_xcen_cutout) else np.nan,
        psf_un_ycen_cutout=float(psf_un_ycen_cutout) if 'psf_un_ycen_cutout' in locals() and psf_un_ycen_cutout is not None and np.isfinite(psf_un_ycen_cutout) else np.nan,
        psf_un_xcen_fullim=float(psf_un_xcen_fullim) if 'psf_un_xcen_fullim' in locals() and psf_un_xcen_fullim is not None and np.isfinite(psf_un_xcen_fullim) else np.nan,
        psf_un_ycen_fullim=float(psf_un_ycen_fullim) if 'psf_un_ycen_fullim' in locals() and psf_un_ycen_fullim is not None and np.isfinite(psf_un_ycen_fullim) else np.nan,
        psf_un_xcen_err=float(psf_un_xcen_err) if 'psf_un_xcen_err' in locals() and psf_un_xcen_err is not None and np.isfinite(psf_un_xcen_err) else np.nan,
        psf_un_ycen_err=float(psf_un_ycen_err) if 'psf_un_ycen_err' in locals() and psf_un_ycen_err is not None and np.isfinite(psf_un_ycen_err) else np.nan,
        psf_un_ra_deg=float(psf_un_ra_deg) if 'psf_un_ra_deg' in locals() and psf_un_ra_deg is not None and np.isfinite(psf_un_ra_deg) else np.nan,
        psf_un_ra_err_mas=float(psf_un_ra_err_mas) if 'psf_un_ra_err_mas' in locals() and psf_un_ra_err_mas is not None and np.isfinite(psf_un_ra_err_mas) else np.nan,
        psf_un_dec_deg=float(psf_un_dec_deg) if 'psf_un_dec_deg' in locals() and psf_un_dec_deg is not None and np.isfinite(psf_un_dec_deg) else np.nan,
        psf_un_dec_err_mas=float(psf_un_dec_err_mas) if 'psf_un_dec_err_mas' in locals() and psf_un_dec_err_mas is not None and np.isfinite(psf_un_dec_err_mas) else np.nan,
        psf_un_sep_as=float(psf_un_sep_as) if 'psf_un_sep_as' in locals() and psf_un_sep_as is not None and np.isfinite(psf_un_sep_as) else np.nan,
        psf_un_ra_dec_cov_mas2=psf_un_ra_dec_cov_mas2 if 'psf_un_ra_dec_cov_mas2' in locals() else np.nan,
        psf_un_wv_um=float(psf_un_wv_um) if 'psf_un_wv_um' in locals() and psf_un_wv_um is not None and np.isfinite(psf_un_wv_um) else np.nan,
        psf_un_wv_um_err=float(psf_un_wv_um_err) if 'psf_un_wv_um_err' in locals() and psf_un_wv_um_err is not None and np.isfinite(psf_un_wv_um_err) else np.nan,
        psf_un_wv_width_um=float(psf_un_wv_width_um) if 'psf_un_wv_width_um' in locals() and psf_un_wv_width_um is not None and np.isfinite(psf_un_wv_width_um) else np.nan,
        psf_un_wv_width_um_err=float(psf_un_wv_width_um_err) if 'psf_un_wv_width_um_err' in locals() and psf_un_wv_width_um_err is not None and np.isfinite(psf_un_wv_width_um_err) else np.nan,
        psf_un_chi2=float(psf_un_chi2) if 'psf_un_chi2' in locals() and psf_un_chi2 is not None and np.isfinite(psf_un_chi2) else np.nan,
        psf_un_dof=int(psf_un_dof) if 'psf_un_dof' in locals() and psf_un_dof is not None else np.nan,
    )

    # Return as a plain dict (JSON-serializable)
    result_dict = asdict(outputs)

    if save_results:
        out_path = Path(results_path) if results_path else Path(f"{fits_path}.results.json")
        with open(out_path, "w") as f:
            _json.dump(result_dict, f, indent=2)
        if debug:
            print(f"[INFO] Results saved to {out_path}")

    return result_dict


def _float_or_none(value: str) -> Optional[float]:
    if isinstance(value, str) and value.lower() in {"none", "auto", "null"}:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPHEREx single-image PSF + aperture fit")
    parser.add_argument("--fits", required=True, help="Path to SPHEREx spectral FITS")
    parser.add_argument("--ra", type=float, required=True, help="Target RA in deg")
    parser.add_argument("--dec", type=float, required=True, help="Target Dec in deg")
    parser.add_argument("--cutout", type=int, nargs=2, default=[15, 15], help="Cutout size (ny nx)")
    parser.add_argument("--fit-radius", type=float, default=4.0, help="Fit radius in pixels")
    parser.add_argument(
        "--aperture-px",
        type=_float_or_none,
        default=3.0,
        help="Forced aperture radius in pixels. Use 'none' or 'auto' to scan radii and maximize S/N (default: 3.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose prints + diagnostics")
    parser.add_argument("--no-show", action="store_true", help="Do not show figures (for batch)")
    parser.add_argument("--save-figs", action="store_true", help="Save figures instead of showing")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Do not write any figures to disk (overrides --save-figs).",
    )
    parser.add_argument("--figs-dir", default=None, help="Directory to save figures")
    parser.add_argument("--posterior-keep", type=int, default=200, help="Keep N posterior samples")
    parser.add_argument("--scipy-only", action="store_true", help="Skip UltraNest sampler; use SciPy PSF fit only")
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write the full result dictionary to a JSON file.",
    )
    parser.add_argument(
        "--results-path",
        default=None,
        help="JSON file path for --save-results output (default: <fits>.results.json).",
    )
    parser.add_argument(
        "--max-pix-offset",
        dest="max_pix_offset",
        type=float,
        default=None,
        help="Max absolute pixel offset for dx and dy (pixels). If 0, dx and dy are fixed to 0.",
    )
    parser.add_argument(
        "--no-masking",
        action="store_true",
        help="Do not mask any pixels during the PSF fitting step (ignore flag-based masking). Does not affect BCG/background-subtraction masking.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_figures:
        args.save_figs = False

    results = analyze_file(
        fits_path=args.fits,
        ra=args.ra,
        dec=args.dec,
        cutout_size=tuple(args.cutout),
        fit_radius_px=args.fit_radius,
        ap_radius_px=args.aperture_px,
        debug=args.debug,
        show_figs=(not args.no_show),
        save_figs=args.save_figs,
        figs_dir=args.figs_dir,
        ultranest_quiet=True,
        posterior_keep=args.posterior_keep,
        use_ultranest=(not args.scipy_only),
        save_results=args.save_results,
        results_path=args.results_path,
        max_pix_offset=args.max_pix_offset,
        no_masking=args.no_masking,
    )

    # Emit a full JSON summary for the batch tool (include every field in `results`).
    def _np_to_jsonable(obj):
        import numpy as _np
        # Convert numpy arrays to lists
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        # Convert numpy scalar types to native Python scalars
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        # Convert numpy booleans
        if isinstance(obj, _np.bool_):
            return bool(obj)
        # Fallback: let json try the default behavior
        return str(obj)

    print("[summary]", _json.dumps(results, allow_nan=True, default=_np_to_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
