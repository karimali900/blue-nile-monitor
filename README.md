# 🛰 Nile Monitor

Satellite + open-source-intelligence (OSINT) monitoring of **human-made change on
the Ethiopian Nile** — dam construction and reservoir filling on the Blue Nile /
upper Nile, centred on the **Grand Ethiopian Renaissance Dam (GERD)**.

The pipeline fuses three free satellite sources into water-extent time series,
construction/activity change maps, georeferenced GeoTIFFs and a written report,
then pairs the quantitative picture with public reporting about the dams.

```
┌───────────────────────────────┐
│  Data sources (all free)      │
│  • Sentinel-2 L2A  (10 m)     │  NDWI water detection  — anonymous, no key
│  • Sentinel-1 GRD  (SAR)      │  cloud-free water + construction index
│  • Landsat 8/9 C2  (30 m)     │  MNDWI, historical baseline
│  via Earth Search STAC +      │
│  Copernicus Data Space        │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│  Per site analysis            │
│  • water masks per scene      │
│  • water area time series     │
│  • granule-coverage filter    │
│  • construction activity idx  │
└──────────────┬────────────────┘
               ▼
┌───────────────────────────────┐
│  Outputs                      │
│  • CSV time series            │
│  • GeoTIFFs (water, change)   │
│  • PNG change + activity maps │
│  • interactive folium HTML    │
│  • OSINT news digest          │
│  • Markdown report            │
└──────────────┬────────────────┘
               ▼
        Streamlit dashboard
```

## Quickstart

```bash
conda create -n nilemonitor python=3.12 -y
conda activate nilemonitor
pip install -r requirements.txt

cp .env.example .env        # add CDSE credentials (only needed for S1 + Landsat)

# 1. See which scenes are available over a site:
python scripts/list_scenes.py --site gerd --days 90

# 2. Run the full pipeline (analysis → maps → OSINT → report):
python scripts/run_pipeline.py --site gerd --since 2026-05-17

# 3. View everything in the dashboard:
streamlit run dashboard/app.py
```

Results land in `output/<site>/<run_tag>/`:

| File | Content |
|---|---|
| `water_area_timeseries.csv` | water area (km²) per scene/date/collection |
| `water_latest_*.tif` / `water_baseline_*.tif` | water masks on a common WGS84 grid |
| `construction_index_*.tif` | SAR construction/activity index raster |
| `water_change_map.png` | latest vs baseline water overlay |
| `construction_map.png` | activity heatmap |
| `interactive_map.html` | folium map (water + activity heat layer) |
| `osint_news.json` | deduplicated open-source news items |
| `report.md` | merged satellite + OSINT report |
| `analysis.json` | run metadata |

## Credentials

- **Sentinel-2**: anonymous (Earth Search STAC / AWS public COGs) — works out of the box.
- **Sentinel-1 + Landsat**: the public AWS buckets for these are *requester-pays*,
  so the pipeline fetches them through the **Copernicus Data Space** catalogue
  using your free CDSE account. Register at <https://dataspace.copernicus.eu>,
  then fill `.env`:

  ```
  CDSE_USERNAME=...
  CDSE_PASSWORD=...
  ```

  or machine-to-machine credentials (`CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`).
  S1 downloads only the VV/VH measurement GeoTIFFs (~1.7 GB per scene, cached
  in `data/cache/s1/`). Landsat downloads B3/B6/QA_PIXEL/MTL only (~130 MB/scene).

## Configuration (`config/sites.yaml`)

- **Sites**: lat/lon, analysis radius, upstream reservoir reach. `gerd` is
  preconfigured at 11.215 N / 35.093 E (Benishangul-Gumuz, ~11 km from the
  Ethiopia–Sudan border). Add more sites (e.g. Beles, Tekeze, Kessem) as needed.
- **Pipeline**: lookback window, per-collection scene caps and cloud limits,
  STAC endpoint, output dirs, minimum water-region size.
- **OSINT**: news queries, keyword allow/deny lists, custom RSS feeds
  (e.g. Nile Basin Initiative outlets).

## Methodology

- **Sentinel-2**: NDWI `(green−NIR)/(green+NIR)` on L2A surface reflectance,
  cloud-masked with the SCL scene-classification band.
- **Landsat**: MNDWI `(green−SWIR1)/(green+SWIR1)` on TOA reflectance derived
  from CDSE L1TP products (MTL coefficients), QA_PIXEL cloud mask.
- **Sentinel-1**: VV/VH backscatter (dB). Calm open water is a very dark SAR
  target; an Otsu split separates water from land, and the temporal standard
  deviation of backscatter + water-state flips form a **construction/activity
  index** — a proxy for dam infill, reservoir filling, land clearing and new
  infrastructure.
- Scenes whose granule covers <50% of the site AOI are skipped (edge slivers).

## Limitations & next steps

- Water area is cloud-limited for optical data; the SAR layer is the
  cloud-independent complement (needs CDSE credentials).
- The activity index is relative (normalised to the 95th percentile of
  backscatter variability); validate hotspots against full-resolution imagery.
- Future: automated weekly runs (cron), DEM-based reservoir elevation estimates,
  SAR backscatter temporal composite (RTC) products, and integration of
  Copernicus global water products for cross-validation.

> Figures are estimates for monitoring purposes. Validate critical claims
> against full-resolution imagery before operational decisions.