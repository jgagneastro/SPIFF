import numpy as np
from astropy.io import fits

from spiff import lv2


def _sample_hdul(nx=80, ny=64):
    primary = fits.PrimaryHDU()
    detector_hdus = [
        fits.ImageHDU(np.zeros((ny, nx), dtype=np.float32), name="IMAGE"),
        fits.ImageHDU(np.zeros((ny, nx), dtype=np.int32), name="FLAGS"),
        fits.ImageHDU(np.ones((ny, nx), dtype=np.float32), name="VARIANCE"),
        fits.ImageHDU(np.zeros((ny, nx), dtype=np.float32), name="ZODI"),
    ]
    detector_hdus[0].header["CRPIX1"] = 40.5
    detector_hdus[0].header["CRPIX2"] = 32.5
    psf = fits.ImageHDU(np.ones((3, 7, 7), dtype=np.float32), name="PSF")
    columns = [
        fits.Column(name="X", format="1J", array=[[1]]),
        fits.Column(name="Y", format="1J", array=[[1]]),
        fits.Column(name="VALUES", format="2E", array=[[1.0, 2.0]]),
    ]
    wave = fits.BinTableHDU.from_columns(columns, name="WCS-WAVE")
    return fits.HDUList([primary, *detector_hdus, psf, wave])


def test_lv2_defaults_to_20_pixel_s3_cutouts():
    args = lv2.build_parser().parse_args(
        [
            "--ra", "10",
            "--dec", "20",
            "--reference-crd-epoch-yr", "2016",
            "--reference-pmra-masyr", "0",
            "--reference-pmdec-masyr", "0",
        ]
    )

    assert args.downloader == "s3"
    assert args.s3_cutout is True
    assert args.s3_cutout_size_px == lv2.DEFAULT_S3_CUTOUT_SIZE_PX == 20


def test_cutout_crops_every_detector_plane_and_preserves_auxiliary_hdus(tmp_path):
    source_path = tmp_path / "source.fits"
    out_path = tmp_path / "cutout.fits"
    _sample_hdul().writeto(source_path)

    with fits.open(source_path) as hdul:
        bounds = lv2._write_fits_cutout(
            hdul,
            str(out_path),
            xpix=79,
            ypix=63,
        )

    assert bounds == (60, 80, 44, 64)
    with fits.open(out_path) as cutout:
        assert [cutout[idx].data.shape for idx in range(1, 5)] == [(20, 20)] * 4
        assert cutout[5].data.shape == (3, 7, 7)
        assert cutout[6].data["VALUES"][0].tolist() == [1.0, 2.0]
        assert cutout[1].header["SPXORX0"] == 60
        assert cutout[1].header["SPXORY0"] == 44
        assert cutout[1].header["SPXNX"] == 80
        assert cutout[1].header["SPXNY"] == 64
        assert cutout[1].header["CRPIX1"] == -19.5
        assert cutout[1].header["CRPIX2"] == -11.5
