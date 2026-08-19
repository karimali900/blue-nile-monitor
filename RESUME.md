# RESUME — blue-nile-monitor

Status as of 2026-08-18 22:20 UTC. Pipeline runs green; results in `output/gerd/full/`.

## Run
```
conda run -n nilemonitor python scripts/run_pipeline.py --site gerd --since 2026-07-27 --tag full
```
(~15 min when it downloads a new S1 scene; ~6 min when cached.)

## Latest verified outputs (output/gerd/full/)
- CSV: S1 water 615.6 km² (Aug 2) → 602.0 km² (Aug 14); S2 45.5 km² (Aug 6) → 34.4 km² (Aug 16). S1 is an upper bound (canyon radar shadow caveat in report).
- Construction index (2-scene SAR std + water flips): 211,263 px ≈ 650.7 km² high-activity; tif + map written; Otsu-derived threshold.
- Landsat: none in window (16-day revisit + USGS Collection 2 agreement not accepted yet → downloads 403 DAT-ZIP-608).
- Construction warnings (`analysis/warnings.py`): VV/VH gain detector on each SAR pair; median-filtered, clustered, shadow/water-fill excluded; severity HIGH/MEDIUM/LOW; written to alerts.json + alerts_history.json + alerts_map.png; report section + dashboard panel. Aug 2→Aug 14 pair: 0 detections (verified correct; unit tests in tests/test_warnings.py).
- Dashboard: `conda run -n nilemonitor streamlit run dashboard/app.py` (port 8501).
- **Independent validation** (`validation/` module): water masks vs **JRC GSW v1.4 (2021) max-extent reference** (tiles from storage.googleapis.com/global-surface-water/downloads2021 — name = left-lon + TOP-lat, e.g. extent_30E_20N spans lat 10-20/lon 30-40) + **Copernicus DEM GLO-30** (AWS copernicus-dem-30m; file is inside a folder prefix, not at bucket root). Writes validation.json (OA/precision/recall/F1/IoU/kappa + storage volume at GERD FSL 655 m) + validation_map.png; report sections (md+html); dashboard "Validation" tab (EN/AR).
- Validation numbers (real run 2026-08-18): S2 OA 0.929 / prec 0.603 / recall 0.123 (S2 NDWI misses turbid reservoir water — honest diagnostic), S1 OA 0.920 / prec 0.474 / recall 0.855 (SAR shadow inflates FP, as caveated). Storage proxy: S2 4.88 km³ over 52 km² (mean depth 94 m), S1 34.49 km³ over 463 km² (74 m) — AOI slice of the full reservoir at FSL, upper-bound labeled.
- Cached validation data: data/cache/gsw/extent_30E_20Nv1_4_2021.tif (~13 MB) + 4 DEM tiles in data/cache/dem/ (~160 MB). Disk ~2.3 GB free.
- Tests: tests/test_validation.py (confusion math by hand, nodata exclusion, volume math) + test_warnings.py — 8 pass.

## Key code facts
- S1 georef: annotation `geolocationGridPointList` → degree-2 poly fit in NORMALIZED (pixel,line) space, linearized at AOI center (normalization is mandatory — raw 0..25000 px is ill-conditioned). Sanity-checked before use.
- S1 calibration: `sigma0 = DN / sigmaNought(line)`, per-line median of 631 values, `np.interp` over calibration lines; `factors[:, None]` broadcast.
- Coverage prefilter: annotation-only `scene_aoi_coverage`, keep top `sar_max_items` by coverage (newest tie-break), then download.
- SAR water: calibrated `vv < -10 dB AND vh < -12 dB` (Otsu unusable: degenerate at fill floor).
- `analysis.json` areas count `isfinite & > 0` (NaN outside granule must not count).
- Windows: `_window_from_bounds` handles rotated affines; `read_window` returns (None, None) on WindowError.
- `validation/reference.py`: GSW tile naming = (left_lon, TOP_lat) — NOT center. DEM tile is a bucket folder (`..._DEM/Copernicus_DSM_COG_..._DEM.tif`). Mosaic blit uses exact corner placement (`_corners` returns left/bottom/right/top in that order — swapping top/bottom breaks the mosaic).
- Metrics rounding: OA/precision/recall/F1/IoU/kappa rounded to 4 dp, area to 2 dp, volume to 3 dp — tests assert the rounded values.
- Disk is 98-99% full: each S1 scene ≈ 1.7 GB; keep `sar_max_items: 2` in config/sites.yaml.

## Blocked / future
- Landsat: user must accept USGS Collection 2 agreement in Copernicus Browser first; then widen window to 2026-07-01.
- More SAR scenes (for a stronger construction index): free disk, raise sar_max_items.
- `sar_max_items` search now fetches max_items*4 products and filters by coverage — do not revert to newest-first acceptance.
- Validation next steps: GERD water level from altimetry (e.g. ICESat-2 ATL13 or Sentinel-3 SRAL via CDSE) to replace the FSL assumption; GSW "monthly history" 2021-2024 for a time-matched reference; turbidity-robust S2 index (NDWI fails on Blue Nile sediment — recall 0.12).