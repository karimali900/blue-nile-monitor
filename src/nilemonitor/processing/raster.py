"""Low-level raster helpers: windowed COG reads, resampling, area, Otsu."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, reproject
from rasterio.windows import Window, WindowError, from_bounds

EPSG4326 = "EPSG:4326"


def _window_from_bounds(left: float, bottom: float, right: float, top: float, tf: Affine) -> Window:
    """Bounding pixel window of a (possibly rotated) affine for the given bounds."""
    inv = ~tf
    cols = [inv * p for p in ((left, top), (right, top), (left, bottom), (right, bottom))]
    col_off = int(np.floor(min(c[0] for c in cols)))
    row_off = int(np.floor(min(c[1] for c in cols)))
    col_max = int(np.ceil(max(c[0] for c in cols)))
    row_max = int(np.ceil(max(c[1] for c in cols)))
    if col_max <= col_off or row_max <= row_off:
        raise WindowError("Bounds and transform are inconsistent")
    return Window(col_off, row_off, col_max - col_off, row_max - row_off)


def read_window(
    href: str,
    band: int,
    bbox_4326: tuple[float, float, float, float],
    out_shape: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.nearest,
    georef: tuple[Affine, str] | None = None,
) -> tuple[np.ndarray, tuple[Affine, str]] | tuple[None, None]:
    """Read a band window of a COG covering ``bbox_4326`` (lon_min,lat_min,lon_max,lat_max).

    Returns (float32 array, (transform, crs)). Returns (None, None) if the scene
    does not overlap the bbox. ``out_shape`` resamples to (rows, cols) pixels.
    All Earth Search assets are public AWS buckets, so requests are unsigned.

    ``georef`` overrides the dataset's own transform/CRS — needed for Sentinel-1
    measurement GeoTIFFs, which ship un-georeferenced while the STAC item carries
    ``proj:transform`` / ``proj:code``.

    The third meta element is the AOI coverage fraction (0..1): how much of the
    requested bbox the granule actually contains. Scenes with low coverage are
    edge slivers and should be skipped.
    """
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_VIRTUAL_HOSTING="FALSE"):
        with rasterio.open(href) as src:
            tf, crs = georef or (src.transform, src.crs)
            if crs is None:
                return None, None
            bbox_src = transform_bounds(EPSG4326, crs, *bbox_4326, densify_pts=21)
            try:
                expected = _window_from_bounds(*bbox_src, tf)
            except WindowError:
                # bbox window inconsistent with the transform (e.g. AOI mostly
                # outside the granule) — treat as no overlap.
                return None, None
            win = expected.intersection(Window(0, 0, src.width, src.height))
            if win.width < 1 or win.height < 1:
                return None, None
            coverage = (win.width * win.height) / max(1e-9, expected.width * expected.height)

            data = src.read(
                band,
                window=win,
                out_shape=out_shape,
                resampling=resampling,
                boundless=False,
            )
            transform = tf * Affine.translation(win.col_off, win.row_off)
            if out_shape is not None and (out_shape[0], out_shape[1]) != (win.height, win.width):
                sx = win.width / out_shape[1]
                sy = win.height / out_shape[0]
                transform = transform * transform.scale(sx, sy)
            nodata = src.nodata
            if nodata is not None:
                data = data.astype("float32")
                data[data == nodata] = np.nan
            else:
                data = data.astype("float32")
                data[~np.isfinite(data)] = np.nan
            return data, (transform, crs, float(min(coverage, 1.0)))


def pixel_area_m2(transform: Affine, crs=None) -> float:
    a, e = abs(transform.a), abs(transform.e)
    crs_str = str(crs).upper() if crs is not None else ""
    if crs_str.startswith("EPSG:4326"):
        import math

        lat = transform.f
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        m_per_deg_lat = 110_574.0
        return a * m_per_deg_lon * e * m_per_deg_lat
    return a * e


def warp_to_wgs84(
    array: np.ndarray,
    transform: Affine,
    src_crs: str,
    bbox_4326: tuple[float, float, float, float],
    out_res_deg: float,
    dtype: str = "float32",
    nodata: float = np.nan,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, Affine]:
    """Reproject a native-CRS array onto a fixed WGS84 grid over ``bbox_4326``."""
    lon_min, lat_min, lon_max, lat_max = bbox_4326
    cols = max(1, int(round((lon_max - lon_min) / out_res_deg)))
    rows = max(1, int(round((lat_max - lat_min) / out_res_deg)))
    dst_transform = Affine(out_res_deg, 0, lon_min, 0, -out_res_deg, lat_max)
    dst = np.full((rows, cols), nodata, dtype=dtype)
    if np.isnan(nodata):
        dst_fill = np.nan
    else:
        dst_fill = nodata
    src_masked = np.ma.masked_invalid(array)
    reproject(
        src_masked,
        dst,
        src_transform=transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=EPSG4326,
        dst_nodata=dst_fill,
        resampling=resampling,
    )
    return dst, dst_transform


def otsu_threshold(data: np.ndarray, bins: int = 256) -> float | None:
    """Global Otsu threshold on finite values. Returns None if too few valid pixels."""
    valid = data[np.isfinite(data)]
    if valid.size < 50:
        return None
    vmin, vmax = np.nanmin(valid), np.nanmax(valid)
    if vmax - vmin < 1e-9:
        return None
    hist, edges = np.histogram(valid, bins=bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return None
    w = np.cumsum(hist) / total
    mu = np.cumsum(hist * centers) / total
    mu_t = mu[-1]
    between = w * (1 - w) * (mu_t - np.where(w > 0, mu / np.maximum(w, 1e-12), 0)) ** 2
    between[w == 0] = 0
    between[w == 1] = 0
    idx = int(np.argmax(between))
    return float(centers[idx])


def remove_small_regions(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Drop connected components smaller than ``min_pixels`` (speckle/noise filter)."""
    from scipy import ndimage

    if min_pixels <= 1:
        return mask
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_pixels
    return keep[labeled]
