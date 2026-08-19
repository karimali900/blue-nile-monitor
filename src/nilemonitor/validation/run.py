"""Orchestrates independent-reference validation and persists results."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from ..analysis.timeseries import SiteAnalysis
from ..processing import raster as rt
from . import metrics as mt
from . import reference as ref

log = logging.getLogger("nilemonitor")

GERD_FSL_M = 655.0  # GERD normal-operation (full supply) level, m a.s.l.


def _mosaic_gsw(masks: list[tuple[np.ndarray, Affine, str]]) -> tuple[np.ndarray, Affine] | None:
    if not masks:
        return None
    left = min(t.c for _, t, _ in masks)
    right = max(t.c + m.shape[1] * t.a for m, t, _ in masks)
    top = max(t.f for _, t, _ in masks)
    bottom = min(t.f + m.shape[0] * t.e for m, t, _ in masks)
    res = abs(masks[0][1].a)
    rows = int(round((top - bottom) / res))
    cols = int(round((right - left) / res))
    out = np.zeros((rows, cols), dtype=bool)
    out_tf = Affine(res, 0, left, 0, -res, top)
    for m, t, _ in masks:
        win = rt._window_from_bounds(left, bottom, right, top, t)
        r0, r1 = int(win.row_off), int(win.row_off + win.height)
        c0, c1 = int(win.col_off), int(win.col_off + win.width)
        out[r0:r1, c0:c1] |= m
    return out, out_tf


def run_validation(analysis: SiteAnalysis, out: Path, cache_dir: Path) -> dict:
    """Validate latest water masks vs JRC GSW and estimate reservoir storage.

    Writes ``validation.json`` and ``validation_map.png`` into ``out`` and
    appends a ``validation`` key to ``analysis.json``. Returns the result dict.
    """
    if analysis.common_grid is None or not analysis.latest_water:
        log.info("No water masks to validate — skipping")
        return {"skipped": True}

    bbox = (
        analysis.site.lon - 0.27,
        analysis.site.lat - 0.27,
        analysis.site.lon + 0.27,
        analysis.site.lat + 0.27,
    )
    gsw = _mosaic_gsw(ref.gsw_extent_masks(bbox, cache_dir))
    dem = ref.dem_window(bbox, cache_dir)

    result: dict = {
        "reference": {
            "gsw": "JRC Global Surface Water v1.4 (2021) maximum extent, Landsat 1984-2021",
            "dem": "Copernicus DEM GLO-30 (2021, ~30 m)",
            "gerd_full_supply_level_m": GERD_FSL_M,
            "bbox": list(bbox),
        },
        "collections": {},
    }

    grid, grid_tf = analysis.common_grid
    for collection, mask in analysis.latest_water.items():
        entry: dict = {}
        if gsw is not None:
            ref_mask, ref_tf = gsw
            entry["gsw_metrics"] = mt.confusion_metrics(mask, grid_tf, ref_mask, ref_tf)
        if dem is not None:
            entry["storage"] = mt.storage_volume(mask, grid_tf, dem[0], dem[1], GERD_FSL_M)
        if entry:
            result["collections"][collection] = entry

    (out / "validation.json").write_text(json.dumps(result, indent=2))

    ap = out / "analysis.json"
    if ap.exists():
        try:
            meta = json.loads(ap.read_text())
            meta["validation"] = result
            ap.write_text(json.dumps(meta, indent=2))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not update analysis.json with validation: %s", e)

    from ..viz.maps import render_validation_map

    render_validation_map(analysis, gsw, out)

    log.info("Validation written to %s (collections: %s)", out / "validation.json", list(result["collections"]))
    return result
