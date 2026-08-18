"""Per-site analysis: fetch scenes, derive water masks, build time series."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from ..auth import CdseAuth
from ..config import Config, Site
from ..download import cdse
from ..download.stac import Scene, search_all_collections
from ..processing import change as chg
from ..processing import raster as rt
from ..processing import water as wtr

import requests  # noqa: E402  (used for download error handling)

COMMON_GRID_DEG = 0.0005  # ~55 m analysis grid for change/maps
MIN_AOI_COVERAGE = 0.5    # skip scenes whose granule covers <50% of the AOI

log = logging.getLogger("nilemonitor")


@dataclass
class SceneReading:
    date: dt.date
    collection: str
    item_id: str
    water_area_km2: float
    ndwi_mean: float | None
    cloud_cover: float | None
    scene_dir: Path | None = None


@dataclass
class SiteAnalysis:
    site: Site
    date_from: dt.date
    date_to: dt.date
    readings: list[SceneReading] = field(default_factory=list)
    scenes_by_collection: dict[str, list] = field(default_factory=dict)
    common_grid: tuple[np.ndarray, Affine] | None = None
    latest_water: dict[str, np.ndarray] = field(default_factory=dict)      # collection -> WGS84 mask
    baseline_water: dict[str, np.ndarray] = field(default_factory=dict)    # collection -> WGS84 mask
    construction: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)  # collection -> (index, thresh)
    sar_stack: dict = field(default_factory=dict)  # {dates, vv, vh} on the common grid (calibrated dB)
    alerts: list = field(default_factory=list)     # construction warnings (see analysis.warnings)
    alerts_pair: str | None = None                 # "baseline__detected" dates of the SAR pair
    preview_pngs: dict = field(default_factory=dict)  # item_id -> {"caption", "collection", "date", "png"}
    out_dir: Path | None = None

    @property
    def series(self) -> pd.DataFrame:
        rows = [
            {
                "date": r.date.isoformat(),
                "collection": r.collection,
                "item_id": r.item_id,
                "water_area_km2": round(r.water_area_km2, 3),
                "ndwi_mean": r.ndwi_mean,
                "cloud_cover": r.cloud_cover,
            }
            for r in self.readings
        ]
        return pd.DataFrame(rows)

    def by_collection(self, collection: str) -> pd.DataFrame:
        return self.series[self.series["collection"] == collection]


def _build_preview(collection: str, band_arrays: dict) -> bytes | None:
    """Render a small PNG preview of a scene from its in-memory bands."""
    from nilemonitor.viz import satellite

    if collection == "sentinel-2-l2a" and band_arrays.get("visual") is not None:
        return satellite.render_rgb_png(band_arrays["visual"])
    if collection == "sentinel-2-l2a" and band_arrays.get("green") is not None and band_arrays.get("nir") is not None:
        g, n = band_arrays["green"], band_arrays["nir"]
        rgb = np.stack([n, g, np.maximum(n, g) * 0.6], axis=-1)
        return satellite.render_rgb_png(rgb)
    if collection == "sentinel-1-grd" and band_arrays.get("vv") is not None:
        db = wtr.to_db(band_arrays["vv"])
        return satellite.render_db_png(db)
    return None


def _band_names(collection: str) -> tuple[str, ...]:
    if collection == "sentinel-2-l2a":
        return "green", "nir", "scl"
    if collection == "sentinel-1-grd":
        return "vv", "vh"
    if collection == "landsat-c2-l2":
        return "green", "swir16", "qa_pixel", "mtl"
    raise ValueError(collection)


def _mask_water(
    collection: str,
    band_arrays: dict[str, np.ndarray],
    min_pixels: int,
) -> np.ndarray | None:
    if collection == "sentinel-2-l2a":
        idx = wtr.ndwi(band_arrays["green"], band_arrays["nir"])
        cloud = wtr.sc2_cloud_mask(band_arrays["scl"]) if band_arrays.get("scl") is not None else None
        return wtr.water_mask_optical(idx, wtr.NDWI_THRESHOLD, cloud, min_pixels)
    if collection == "sentinel-1-grd":
        vv = wtr.to_db(band_arrays["vv"])
        vh = wtr.to_db(band_arrays["vh"]) if band_arrays.get("vh") is not None else None
        return wtr.water_mask_sar(vv, vh, min_pixels)
    if collection == "landsat-c2-l2":
        if band_arrays.get("mtl") is not None:
            green, swir = wtr.landsat_toa(band_arrays["green"], band_arrays["swir16"], band_arrays["mtl"])
        else:
            green, swir = band_arrays["green"], band_arrays["swir16"]
        idx = wtr.mndwi(green, swir)
        cloud = wtr.landsat_qa_cloud_mask(band_arrays["qa_pixel"]) if band_arrays.get("qa_pixel") is not None else None
        return wtr.water_mask_optical(idx, wtr.MNDWI_THRESHOLD, cloud, min_pixels)
    return None


def _read_scene_bands(
    scene,
    bbox_4326: tuple[float, float, float, float],
) -> tuple[dict[str, np.ndarray], tuple[Affine, str]]:
    """Read all required bands of a scene over the site bbox at native resolution."""
    bands: dict[str, np.ndarray] = {}
    transform, crs = None, None
    coverage = 0.0
    primary_shape: tuple[int, int] | None = None
    for name in _band_names(scene.collection):
        href = scene.href(name)
        if not href:
            continue
        if name == "mtl":
            bands[name] = Path(href).read_text(errors="ignore")
            continue
        out_shape = None
        if primary_shape is not None:
            # Resample auxiliary bands (e.g. S2 SCL at 20 m) onto the primary grid.
            out_shape = primary_shape
        georef_override = scene.georef_override()
        georef = (Affine(*georef_override[0]), georef_override[1]) if georef_override else None
        arr, meta = rt.read_window(href, 1, bbox_4326, out_shape=out_shape, georef=georef)
        if arr is None:
            return {}, (transform, crs)
        bands[name] = arr
        if transform is None:
            # Use the first (finest-resolution) band's georef for area/geolocation;
            # later bands (e.g. S2 SCL at 20 m) share the window but differ in pixel size.
            transform, crs = meta[0], meta[1]
            primary_shape = arr.shape
        coverage = max(coverage, meta[2] if len(meta) > 2 else 1.0)
    if coverage < MIN_AOI_COVERAGE:
        log.info("skipping %s: AOI coverage only %.0f%%", getattr(scene, "item_id", "?"), coverage * 100)
        return {}, (transform, crs)
    if scene.collection == "sentinel-2-l2a":
        # True-colour preview (low-res read of the visual asset)
        vhref = scene.href("visual")
        if vhref:
            vbands = [rt.read_window(vhref, i + 1, bbox_4326, out_shape=(360, 360))[0] for i in range(3)]
            if all(v is not None for v in vbands):
                bands["visual"] = np.stack(vbands, axis=-1)
    if scene.collection == "sentinel-1-grd" and transform is not None and scene.georef_override() is not None:
        cdse._calibrate_scene_bands(scene, bands, meta)
    return bands, (transform, crs)


def _load_sar_scenes(
    cfg: Config,
    site: Site,
    date_from: dt.date,
    date_to: dt.date,
) -> list[Scene] | None:
    """Fetch Sentinel-1 GRD via CDSE (credentialed). Returns None if unavailable."""
    auth = CdseAuth(cfg.cache_dir)
    if not auth.configured:
        log.warning(
            "SAR layer skipped: CDSE credentials missing. Add CDSE_USERNAME/CDSE_PASSWORD "
            "or CDSE_CLIENT_ID/CDSE_CLIENT_SECRET to .env to enable Sentinel-1."
        )
        return None

    log.info("Searching CDSE for Sentinel-1 GRD over %s", site.id)
    products = cdse.search_s1_grd(auth, site.bbox_deg, date_from, date_to, max(cfg.sar_max_items * 4, 10))
    if not products:
        log.warning("No S1 GRD products found in window over %s", site.id)
        return None

    # Cheap annotation-only coverage check first, then keep the sar_max_items
    # best-covered scenes (newest-first as tie-break) and only download those.
    candidates: list[tuple[float, dict]] = []
    for product in products:
        try:
            coverage = cdse.scene_aoi_coverage(auth, product, cfg.cache_dir, site.bbox_deg, site.center_deg)
        except (PermissionError, requests.HTTPError, ValueError) as exc:
            log.warning("coverage check failed for %s: %s", product["Name"], exc)
            coverage = None
        if coverage is None or coverage < MIN_AOI_COVERAGE:
            log.info(
                "skipping S1 %s before download: AOI coverage only %s",
                product["Name"],
                "n/a" if coverage is None else f"{coverage * 100:.0f}%",
            )
            continue
        candidates.append((coverage, product))
    candidates.sort(key=lambda t: (-t[0], t[1]["ContentDate"]["Start"]), reverse=False)
    candidates = candidates[: cfg.sar_max_items]
    if not candidates:
        log.warning("No S1 GRD products with >= %.0f%% AOI coverage in window over %s", MIN_AOI_COVERAGE * 100, site.id)
        return None
    log.info("S1 scenes selected for download: %s", ", ".join(p["Name"][:30] for _, p in candidates))

    scenes: list[Scene] = []
    for coverage, product in candidates:
        try:
            bands = cdse.download_measurement_tiffs(auth, product, cfg.cache_dir)
        except (PermissionError, requests.HTTPError) as exc:
            log.warning("S1 download failed for %s: %s", product["Name"], exc)
            continue
        if "vv" not in bands:
            continue
        assets = {
            "vv": str(bands["vv"]),
            **({"vh": str(bands["vh"])} if "vh" in bands else {}),
            **({"cal_vv": str(bands["cal_vv"])} if "cal_vv" in bands else {}),
            **({"cal_vh": str(bands["cal_vh"])} if "cal_vh" in bands else {}),
        }
        properties: dict = {}
        annotation = bands.get("annotation")
        if annotation is not None:
            georef = cdse.georef_from_annotation(Path(annotation), anchor_lonlat=site.center_deg)
            if georef is not None:
                properties["proj:transform"] = tuple(float(x) for x in georef)
                properties["proj:code"] = "EPSG:4326"
        scenes.append(
            Scene(
                collection="sentinel-1-grd",
                item_id=product["Id"],
                datetime=dt.datetime.fromisoformat(
                    product["ContentDate"]["Start"].replace("Z", "+00:00")
                ),
                bbox=site.bbox_deg,
                cloud_cover=None,
                assets=assets,
                properties=properties,
            )
        )
    return scenes


def _load_landsat_scenes(
    cfg: Config,
    site: Site,
    date_from: dt.date,
    date_to: dt.date,
) -> list[Scene] | None:
    """Fetch Landsat C2 L1TP via CDSE (credentialed). Returns None if unavailable."""
    auth = CdseAuth(cfg.cache_dir)
    if not auth.configured:
        log.warning("Landsat layer skipped: CDSE credentials missing.")
        return None

    log.info("Searching CDSE for Landsat over %s", site.id)
    products = cdse.search_landsat(auth, site.bbox_deg, date_from, date_to, cfg.landsat_max_items)
    if not products:
        log.warning("No Landsat products found in window over %s", site.id)
        return None

    scenes: list[Scene] = []
    for product in products:
        try:
            files = cdse.download_landsat_bands(auth, product, cfg.cache_dir)
        except (PermissionError, requests.HTTPError) as exc:
            log.warning("Landsat download failed for %s: %s", product["Name"], exc)
            continue
        if "green" not in files or "swir16" not in files:
            continue
        assets = {
            "green": str(files["green"]),
            "swir16": str(files["swir16"]),
            **({"qa_pixel": str(files["qa_pixel"])} if "qa_pixel" in files else {}),
            **({"mtl": str(files["mtl"])} if "mtl" in files else {}),
        }
        scenes.append(
            Scene(
                collection="landsat-c2-l2",
                item_id=product["Id"],
                datetime=dt.datetime.fromisoformat(
                    product["ContentDate"]["Start"].replace("Z", "+00:00")
                ),
                bbox=site.bbox_deg,
                cloud_cover=None,
                assets=assets,
                properties={},
            )
        )
    return scenes


def run_site_analysis(
    cfg: Config,
    site: Site,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
    min_water_region_m2: float | None = None,
) -> SiteAnalysis:
    date_to = date_to or dt.date.today()
    date_from = date_from or (date_to - dt.timedelta(days=cfg.lookback_days))

    bbox = site.bbox_deg
    scenes = search_all_collections(
        cfg.stac_url, bbox, date_from, date_to, cfg.collection_config
    )

    if cfg.use_sar and "sentinel-1-grd" in scenes:
        sar_scenes = _load_sar_scenes(cfg, site, date_from, date_to)
        if sar_scenes:
            scenes["sentinel-1-grd"] = sar_scenes
        else:
            scenes.pop("sentinel-1-grd", None)

    if "landsat-c2-l2" in scenes:
        landsat_scenes = _load_landsat_scenes(cfg, site, date_from, date_to)
        if landsat_scenes:
            scenes["landsat-c2-l2"] = landsat_scenes
        else:
            scenes.pop("landsat-c2-l2", None)

    analysis = SiteAnalysis(site=site, date_from=date_from, date_to=date_to)
    analysis.scenes_by_collection = scenes

    min_pixels = max(
        2,
        int((min_water_region_m2 or cfg.min_water_region_m2) / (0.0011 * 0.0011 * 111_000 * 111_000)),
    )

    # SAR stacks for the construction index (on the common WGS84 grid)
    sar_dB_stack: dict[str, list[np.ndarray]] = {"vv": [], "vh": []}
    sar_dates: list[dt.date] = []
    water_stack: dict[str, list[np.ndarray]] = {}

    # Baseline water masks per collection
    all_scene_dates: dict[str, list[SceneReading]] = {c: [] for c in scenes}

    for collection, items in scenes.items():
        for scene in items:
            band_arrays, meta = _read_scene_bands(scene, bbox)
            if not band_arrays or meta[0] is None:
                continue
            transform, crs = meta

            native_mask = _mask_water(collection, band_arrays, min_pixels)
            if native_mask is None:
                continue

            area_km2 = float(np.count_nonzero(native_mask) * rt.pixel_area_m2(transform, crs)) / 1e6

            reading = SceneReading(
                date=scene.date,
                collection=collection,
                item_id=scene.item_id,
                water_area_km2=area_km2,
                ndwi_mean=(
                    float(np.nanmean(wtr.ndwi(band_arrays["green"], band_arrays["nir"])))
                    if collection in ("sentinel-2-l2a", "landsat-c2-l2") and "green" in band_arrays
                    else None
                ),
                cloud_cover=scene.cloud_cover,
            )
            analysis.readings.append(reading)
            all_scene_dates[collection].append(scene.date)

            # Satellite preview (thumbnail PNG) for the dashboard/HTML report
            preview = _build_preview(collection, band_arrays)
            if preview:
                analysis.preview_pngs[scene.item_id] = {
                    "caption": f"{collection} · {scene.date.isoformat()}"
                    + (f" · cloud {scene.cloud_cover:.0f}%" if scene.cloud_cover else ""),
                    "collection": collection,
                    "date": scene.date.isoformat(),
                    "png": preview,
                }

            # Warp water mask to common grid
            mask_wgs, grid_transform = rt.warp_to_wgs84(
                native_mask.astype("float32"),
                transform,
                crs,
                bbox,
                COMMON_GRID_DEG,
                resampling=rasterio.enums.Resampling.nearest,
            )
            analysis.common_grid = (mask_wgs, grid_transform)

            water_stack.setdefault(collection, []).append((scene.date, mask_wgs))
            if collection == "sentinel-1-grd":
                for key in ("vv", "vh"):
                    if key in band_arrays:
                        db = wtr.to_db(band_arrays[key])
                        dbwgs, _ = rt.warp_to_wgs84(db, transform, crs, bbox, COMMON_GRID_DEG)
                        sar_dB_stack[key].append(dbwgs)
                sar_dates.append(scene.date)

    # Sort and pick latest/baseline masks per collection
    for collection, stack in water_stack.items():
        stack.sort(key=lambda t: t[0])
        if stack:
            analysis.baseline_water[collection] = stack[0][1]
            analysis.latest_water[collection] = stack[-1][1]

    # Construction index from SAR activity + water flips (needs >= 2 scenes)
    if len(sar_dB_stack["vv"]) >= 2:
        activity = chg.stack_activity(sar_dB_stack["vv"])
        vh_stack = sar_dB_stack["vh"]
        if len(vh_stack) == len(sar_dB_stack["vv"]):
            vh_act = chg.stack_activity(vh_stack)
            activity = np.where(np.isfinite(vh_act), 0.7 * activity + 0.3 * vh_act, activity)
        water_masks = [m for _, m in water_stack.get("sentinel-1-grd", [])]
        flips = chg.water_flip_fraction(water_masks) if water_masks else np.full_like(activity, np.nan)
        idx, thresh = chg.construction_index(activity, flips)
        analysis.construction["sentinel-1-grd"] = (idx, thresh)

        # Construction warnings: region-level alerts from the SAR pair
        from . import warnings as wrn

        order = np.argsort([d.toordinal() for d in sar_dates])
        sar_dates_sorted = [sar_dates[i] for i in order]
        vv_sorted = [sar_dB_stack["vv"][i] for i in order]
        vh_sorted = [sar_dB_stack["vh"][i] for i in order]
        analysis.sar_stack = {
            "dates": [d.isoformat() for d in sar_dates_sorted],
            "vv": vv_sorted,
            "vh": vh_sorted,
        }
        wm_dates = [d for d, _ in water_stack.get("sentinel-1-grd", [])]
        wm_masks = [m for _, m in water_stack.get("sentinel-1-grd", [])]
        wm_by_date = dict(zip(wm_dates, wm_masks))
        wm_aligned = [wm_by_date[d] for d in sar_dates_sorted if d in wm_by_date]
        if len(wm_aligned) == len(sar_dates_sorted):
            analysis.alerts_pair = f"{sar_dates_sorted[0].isoformat()}__{sar_dates_sorted[-1].isoformat()}"
            grid_transform = analysis.common_grid[1] if analysis.common_grid else None
            analysis.alerts = wrn.detect_construction(
                sar_dates_sorted, vv_sorted, vh_sorted, wm_aligned, grid_transform=grid_transform
            )
            log.info("construction warnings: %d alert(s) for pair %s", len(analysis.alerts), analysis.alerts_pair)

    analysis.readings.sort(key=lambda r: (r.date, r.collection))
    return analysis


def save_outputs(cfg: Config, analysis: SiteAnalysis, run_tag: str | None = None) -> Path:
    """Persist CSV, GeoTIFFs and JSON of a SiteAnalysis. Returns the output dir."""
    out = cfg.output_dir / analysis.site.id / (run_tag or analysis.date_from.isoformat() + "__" + analysis.date_to.isoformat())
    out.mkdir(parents=True, exist_ok=True)

    series = analysis.series
    series.to_csv(out / "water_area_timeseries.csv", index=False)

    if analysis.common_grid is not None:
        grid, grid_transform = analysis.common_grid
        height, width = grid.shape
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": grid_transform,
        }
        for collection, mask in analysis.baseline_water.items():
            _write_tiff(out / f"water_baseline_{collection}.tif", np.where(np.isfinite(mask), mask, 0).astype("uint8"), profile)
        for collection, mask in analysis.latest_water.items():
            _write_tiff(out / f"water_latest_{collection}.tif", np.where(np.isfinite(mask), mask, 0).astype("uint8"), profile)
        for collection, (idx, thresh) in analysis.construction.items():
            p = dict(profile)
            p["dtype"] = "float32"
            p["nodata"] = -9999.0
            arr = np.where(np.isfinite(idx), idx, -9999.0).astype("float32")
            _write_tiff(out / f"construction_index_{collection}.tif", arr, p)

    meta = {
        "site": analysis.site.id,
        "site_name": analysis.site.name,
        "lon": analysis.site.lon,
        "lat": analysis.site.lat,
        "date_from": analysis.date_from.isoformat(),
        "date_to": analysis.date_to.isoformat(),
        "n_scenes": len(analysis.readings),
        "collections": {
            c: {"count": len(v), "max_items": cfg.collection_config.get(c, {}).get("max_items")}
            for c, v in analysis.scenes_by_collection.items()
        },
        "water_area_km2_latest": {
            c: _count_km2_deg(m) for c, m in analysis.latest_water.items()
        },
        "water_area_km2_baseline": {
            c: _count_km2_deg(m) for c, m in analysis.baseline_water.items()
        },
        "alerts": {
            "count": len(analysis.alerts),
            "by_severity": {
                s: sum(1 for a in analysis.alerts if a.severity == s)
                for s in ("HIGH", "MEDIUM", "LOW")
            },
            "pair": analysis.alerts_pair,
        },
    }
    (out / "analysis.json").write_text(json.dumps(meta, indent=2))

    if analysis.alerts_pair:
        from .warnings import save_alerts

        bg = None
        if analysis.sar_stack.get("vv"):
            bg = analysis.sar_stack["vv"][-1]
        grid_transform = analysis.common_grid[1] if analysis.common_grid else None
        save_alerts(analysis.alerts, out, bg, grid_transform, pair=analysis.alerts_pair)

    if analysis.preview_pngs:
        scenes_dir = out / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        for item_id, p in analysis.preview_pngs.items():
            (scenes_dir / f"{item_id}.png").write_bytes(p["png"])

    analysis.out_dir = out
    return out


def _count_km2_deg(mask: np.ndarray, deg: float = COMMON_GRID_DEG) -> float:
    """Area (km2) of mask pixels on a degree grid; NaN/outside-granule px excluded."""
    return float(np.count_nonzero(np.isfinite(mask) & (mask > 0)) * (deg * 111_000) ** 2) / 1e6


def _write_tiff(path: Path, arr: np.ndarray, profile: dict) -> None:
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
