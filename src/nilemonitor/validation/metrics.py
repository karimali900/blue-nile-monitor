"""Accuracy metrics and storage-volume estimation against independent references."""
from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from ..processing import raster as rt


def _warp_nearest(mask: np.ndarray, src_tf: Affine, dst_shape: tuple[int, int], dst_tf: Affine) -> np.ndarray:
    """Reproject a binary mask (nan = invalid) onto the reference grid."""
    out = np.full(dst_shape, np.nan, dtype="float32")
    src = np.where(np.isfinite(mask), mask.astype("float32"), np.nan)
    reproject(
        src,
        out,
        src_transform=src_tf,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=dst_tf,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return out


def confusion_metrics(
    mask: np.ndarray,
    grid_transform: Affine,
    ref: np.ndarray,
    ref_transform: Affine,
) -> dict:
    """Confusion matrix of our water mask vs an independent reference mask.

    ``ref`` is a boolean array on its own grid; both are WGS84. Pixels where
    either array is invalid/NaN are excluded. Water = positive class.
    Returns counts, overall accuracy, precision (user's), recall (producer's),
    F1, IoU and Cohen's kappa.
    """
    ref_shape = ref.shape
    pred = _warp_nearest(mask, grid_transform, ref_shape, ref_transform)
    valid = np.isfinite(pred) & np.isfinite(ref.astype("float32"))
    if valid.sum() == 0:
        return {"error": "no valid overlap pixels"}
    p = pred[valid] > 0.5
    r = ref[valid] > 0.5

    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(np.count_nonzero(~p & ~r))
    n = tp + fp + fn + tn

    oa = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    p_pos = (tp + fp) / n
    p_neg = (tn + fn) / n
    r_pos = (tp + fn) / n
    r_neg = (tn + fp) / n
    p_agree = (tp + tn) / n
    p_chance = p_pos * r_pos + p_neg * r_neg
    kappa = (p_agree - p_chance) / (1 - p_chance) if p_chance < 1 else float("nan")

    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
        "overall_accuracy": round(oa, 4),
        "precision_user": round(precision, 4),
        "recall_producer": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "kappa": round(kappa, 4),
    }


def storage_volume(
    water_mask: np.ndarray,
    grid_transform: Affine,
    dem: np.ndarray,
    dem_transform: Affine,
    crest_level_m: float = 655.0,
) -> dict:
    """Reservoir storage proxy: sum of (crest - bed) over water pixels.

    ``crest_level_m`` is the normal-operation (full supply) water level in
    metres a.s.l. The DEM is warped onto the water-mask grid; only pixels
    where water is detected and DEM is valid contribute. Returns area, mean
    depth, volume (km3 = BCM) and a note that it is an upper-bound proxy when
    the actual lake level is below the crest.
    """
    rows, cols = water_mask.shape
    dem_on_grid = np.full((rows, cols), np.nan, dtype="float32")
    reproject(
        dem,
        dem_on_grid,
        src_transform=dem_transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=grid_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    water = np.isfinite(water_mask) & (water_mask > 0.5)
    depth = np.where(water, np.clip(crest_level_m - dem_on_grid, 0, None), np.nan)
    valid = np.isfinite(depth)
    cell_m2 = rt.pixel_area_m2(grid_transform, "EPSG:4326")
    area_km2 = float(valid.sum() * cell_m2 / 1e6)
    if valid.sum() == 0:
        return {"area_km2": 0.0, "mean_depth_m": 0.0, "volume_km3": 0.0,
                "crest_level_m": crest_level_m, "note": "no water pixels on DEM"}
    volume_km3 = float(np.nansum(depth) * cell_m2 / 1e9)
    return {
        "area_km2": round(area_km2, 2),
        "mean_depth_m": round(float(np.nanmean(depth)), 1),
        "volume_km3": round(volume_km3, 3),
        "crest_level_m": crest_level_m,
        "note": "upper-bound proxy: assumes the lake is at full supply level",
    }
