"""Independent reference data for validation.

* JRC Global Surface Water v1.4 (2021) ``extent`` layer — the version of
  record of the 2016 Nature publication (Pekel et al.), produced from the
  full Landsat archive 1984-2021. Values: 1 = water ever detected.
  Public tiles: https://storage.googleapis.com/global-surface-water/downloads2021/
* Copernicus DEM GLO-30 (COG tiles, ~30 m) — AWS open data.
  Public tiles: https://copernicus-dem-30m.s3.amazonaws.com/
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from ..processing import raster as rt

log = logging.getLogger("nilemonitor")

GSW_BASE = "https://storage.googleapis.com/global-surface-water/downloads2021"
DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_names(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """JRC 10x10-degree tiles covering ``bbox``.

    Tile names encode the left longitude edge and the top latitude edge:
    ``extent_30E_10N`` spans lon 30-40, lat 0-10.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lefts = range(int(math.floor(lon_min / 10)) * 10, int(math.floor(lon_max / 10)) * 10 + 1, 10)
    tops = range(int(math.ceil(lat_max / 10)) * 10, int(math.ceil(lat_min / 10)) * 10 - 1, -10)
    return [(int(l), int(t)) for l in lefts for t in tops if t > 0]


def gsw_extent_masks(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> list[tuple[np.ndarray, Affine, str]]:
    """Reference 'ever water' boolean masks from JRC GSW v1.4 covering ``bbox``.

    Returns a list of (bool mask, transform, tile name); the caller mosaics.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    masks: list[tuple[np.ndarray, Affine, str]] = []
    for lon_left, lat_top in _tile_names(bbox):
        lon_tag = f"{lon_left}E" if lon_left >= 0 else f"{-lon_left}W"
        lat_tag = f"{lat_top}N" if lat_top >= 0 else f"{-lat_top}S"
        name = f"extent_{lon_tag}_{lat_tag}v1_4_2021.tif"
        path = cache_dir / "gsw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GSW_BASE}/extent/{name}"
        if not path.exists():
            log.info("Downloading GSW reference tile %s", name)
            urlretrieve(url, path)
        with rasterio.open(path) as src:
            win = rt._window_from_bounds(*bbox, src.transform).intersection(
                Window(0, 0, src.width, src.height)
            )
            if win.width < 1 or win.height < 1:
                continue
            data = src.read(1, window=win).astype("float32")
            data[data == src.nodata] = np.nan
            tf = src.transform * Affine.translation(win.col_off, win.row_off)
        masks.append((data == 1.0, tf, name))
    return masks


def dem_window(
    bbox: tuple[float, float, float, float],
    cache_dir: Path,
) -> tuple[np.ndarray, Affine] | None:
    """Copernicus DEM GLO-30 elevation (m) over ``bbox`` as one mosaic."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles: list[tuple[np.ndarray, Affine, str]] = []
    for lat_deg in range(int(math.floor(lat_min)), int(math.ceil(lat_max))):
        for lon_deg in range(int(math.floor(lon_min)), int(math.ceil(lon_max))):
            lat_tag = f"N{lat_deg:02d}_00" if lat_deg >= 0 else f"S{-lat_deg:02d}_00"
            lon_tag = f"E{lon_deg:03d}_00" if lon_deg >= 0 else f"W{-lon_deg:03d}_00"
            name = f"Copernicus_DSM_COG_10_{lat_tag}_{lon_tag}_DEM"
            path = cache_dir / "dem" / f"{name}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{DEM_BASE}/{name}/{name}.tif"
            if not path.exists():
                log.info("Downloading Copernicus DEM tile %s (~40 MB)", name)
                urlretrieve(url, path)
            with rasterio.open(path) as src:
                win = rt._window_from_bounds(*bbox, src.transform).intersection(
                    Window(0, 0, src.width, src.height)
                )
                if win.width < 1 or win.height < 1:
                    continue
                data = src.read(1, window=win).astype("float32")
                data[data == src.nodata] = np.nan
                tf = src.transform * Affine.translation(win.col_off, win.row_off)
            tiles.append((data, tf, name))

    if not tiles:
        return None

    def _corners(data: np.ndarray, tf: Affine):
        """(left, bottom, right, top) of the array on its transform."""
        return tf.c, tf.f + data.shape[0] * tf.e, tf.c + data.shape[1] * tf.a, tf.f

    left = min(_corners(d, t)[0] for d, t, _ in tiles)
    bottom = min(_corners(d, t)[1] for d, t, _ in tiles)
    right = max(_corners(d, t)[2] for d, t, _ in tiles)
    top = max(_corners(d, t)[3] for d, t, _ in tiles)
    res = abs(tiles[0][1].a)
    out = np.full((int(round((top - bottom) / res)), int(round((right - left) / res))), np.nan, dtype="float32")
    out_tf = Affine(res, 0, left, 0, -res, top)
    for data, tf, _ in tiles:
        t_left, t_bottom, t_right, t_top = _corners(data, tf)
        c0 = int(round((t_left - left) / res))
        c1 = int(round((t_right - left) / res))
        r0 = int(round((top - t_top) / res))
        r1 = int(round((top - t_bottom) / res))
        out[r0:r1, c0:c1] = np.where(np.isfinite(out[r0:r1, c0:c1]), out[r0:r1, c0:c1], data)
    return out, out_tf
