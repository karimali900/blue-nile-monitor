"""Water detection from optical and SAR bands.

S2   -> NDWI = (green - nir) / (green + nir)
S1   -> calm open water is a very dark SAR target; on calibrated sigma0 dB we
        apply fixed dual-polarization thresholds (Otsu is unusable: the scene
        histogram is unimodal and dominated by the DN fill floor)
L8/9 -> MNDWI = (green - swir16) / (green + swir16)
"""
from __future__ import annotations

import numpy as np

from .raster import remove_small_regions

NDWI_THRESHOLD = 0.10
MNDWI_THRESHOLD = 0.0
SAR_DB_FLOOR = -60.0
SAR_WATER_VV_DB = -10.0
SAR_WATER_VH_DB = -12.0

# Sentinel-2 SCL classes
SCL_WATER = {6}
SCL_CLOUD = {3, 7, 8, 9, 10}


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(denom) & (denom != 0), (green - nir) / np.where(denom == 0, np.nan, denom), np.nan)


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    denom = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(denom) & (denom != 0), (green - swir) / np.where(denom == 0, np.nan, denom), np.nan)


def to_db(power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.maximum(power, 1e-9))
    return out


def water_mask_optical(
    index: np.ndarray,
    threshold: float,
    cloud_mask: np.ndarray | None = None,
    min_pixels: int = 4,
) -> np.ndarray:
    """Water = index above threshold, cleared of cloud/cloud-shadow pixels."""
    valid = np.isfinite(index)
    mask = valid & (index > threshold)
    if cloud_mask is not None:
        mask &= ~cloud_mask
    return remove_small_regions(mask.astype(bool), min_pixels)


def water_mask_sar(vv_db: np.ndarray, vh_db: np.ndarray | None = None, min_pixels: int = 4) -> np.ndarray:
    """Water from calibrated sigma0 (dB): very dark in both VV and VH.

    Fixed thresholds on calibrated sigma0 (see module docstring). Dark radar
    shadow in rugged terrain can mimic water; the caveat is reported by the
    pipeline rather than silently absorbed by a per-scene Otsu split.
    """
    vv = vv_db.astype("float32")
    vv[vv < SAR_DB_FLOOR] = np.nan
    mask = np.isfinite(vv) & (vv < SAR_WATER_VV_DB)
    if vh_db is not None:
        vh = vh_db.astype("float32")
        vh[vh < SAR_DB_FLOOR] = np.nan
        # Water should also be dark in cross-pol relative to land.
        mask &= np.isfinite(vh) & (vh < SAR_WATER_VH_DB)
    return remove_small_regions(mask, min_pixels)


def sc2_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """Build cloud/cloud-shadow/confident-cover mask from the S2 SCL band."""
    return np.isin(scl.astype(np.int16), list(SCL_CLOUD))


def landsat_qa_cloud_mask(qa: np.ndarray) -> np.ndarray:
    """Cloud mask from Landsat C2 QA_PIXEL: high-confidence cloud or cirrus."""
    qa = qa.astype(np.uint16)
    cloud_conf = (qa >> 3) & 0b11
    return (cloud_conf == 0b11) | ((qa >> 5) & 0b1) == 1


def landsat_toa(
    dn_green: np.ndarray,
    dn_swir: np.ndarray,
    mtl_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Landsat L1TP DN to TOA reflectance using MTL coefficients."""
    import re

    def _get(pattern: str) -> float:
        m = re.search(pattern + r"\s*=\s*([-+0-9.Ee]+)", mtl_text)
        return float(m.group(1)) if m else 1.0

    mult3 = _get(r"REFLECTANCE_MULT_BAND_3")
    add3 = _get(r"REFLECTANCE_ADD_BAND_3")
    mult6 = _get(r"REFLECTANCE_MULT_BAND_6")
    add6 = _get(r"REFLECTANCE_ADD_BAND_6")
    sun_elev = _get(r"SUN_ELEVATION")
    s = float(np.sin(np.deg2rad(sun_elev)))
    with np.errstate(divide="ignore", invalid="ignore"):
        green = (dn_green * mult3 + add3) / s
        swir = (dn_swir * mult6 + add6) / s
    green[~np.isfinite(green)] = np.nan
    swir[~np.isfinite(swir)] = np.nan
    return green.astype("float32"), swir.astype("float32")
