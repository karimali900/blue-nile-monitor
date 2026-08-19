"""Construction-change detection on Sentinel-1 stacks: region-level alerts.

New construction (dams, bridges, earthworks) introduces strong, stable radar
scatterers. Over a short SAR repeat (12 days) that shows up as a large sigma0
gain in VV (and typically VH) between two scenes, either on quiet land
(excavation / structures) or over calm water (bridges, coffer dams). The
detector normalizes each scene by its own median (removes inter-scene offsets),
median-filters the gain image (speckle is per-pixel), clusters survivors, and
emits one alert per cluster, ranked by severity.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

COMMON_GRID_DEG = 0.0005

LAND_VV_GAIN_DB = 3.5   # land-type alert: minimum VV sigma0 gain (dB)
LAND_VH_GAIN_DB = 1.5   # land-type alert: minimum VH sigma0 gain (dB)
WATER_VV_GAIN_DB = 6.0  # structure-over-water alert: minimum VV gain (dB)
MIN_PX_LAND = 4         # speckle is per-pixel; require a real region
MIN_PX_WATER = 3
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass
class ConstructionAlert:
    id: str
    kind: str                    # 'land' | 'water'
    date_baseline: str
    date_detected: str
    lon: float
    lat: float
    area_km2: float
    pixels: int
    vv_gain_db: float            # cluster mean gain
    vv_gain_db_max: float
    vh_gain_db: float | None
    severity: str                # HIGH / MEDIUM / LOW
    description: str = ""


def _nan_median_filter(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter ignoring NaN; NaNs are restored afterwards."""
    valid = np.isfinite(arr)
    if not valid.any():
        return arr
    filled = np.where(valid, arr, -999.0)
    out = ndimage.median_filter(filled, size=size)
    return np.where(valid, out, np.nan)


def detect_construction(
    dates: list[dt.date],
    vv_stack: list[np.ndarray],
    vh_stack: list[np.ndarray],
    water_masks: list[np.ndarray],
    grid_deg: float = COMMON_GRID_DEG,
    grid_transform=None,
) -> list[ConstructionAlert]:
    """Return region-level construction alerts from a >=2 scene SAR stack.

    ``vv_stack``/``vh_stack`` are calibrated sigma0 dB images on a common grid
    (sorted ascending by ``dates``); ``water_masks`` are the SAR water masks
    per date (used to separate land vs water construction and to exclude radar
    shadow, which is always classified as water). ``grid_deg`` is the common
    grid pixel size in degrees; ``grid_transform`` (rasterio Affine, north-up)
    maps pixels to lon/lat when given.
    """
    if len(vv_stack) < 2 or len(vh_stack) < 2:
        return []
    if not all(np.isfinite(a).any() for a in vv_stack):
        return []

    vv0, vv1 = vv_stack[0], vv_stack[-1]
    vh0, vh1 = vh_stack[0], vh_stack[-1]
    w0, w1 = water_masks[0], water_masks[-1]
    d0, d1 = dates[0].isoformat(), dates[-1].isoformat()

    def _to_lonlat(x: float, y: float) -> tuple[float, float]:
        if grid_transform is not None:
            t = grid_transform
            return t.c + x * t.a, t.f + y * t.e
        return x * grid_deg, -y * grid_deg

    # Remove systematic inter-scene offset (instrument gain, incidence angle):
    # compare each scene to its own median over the overlap.
    ov = np.isfinite(vv0) & np.isfinite(vv1) & np.isfinite(vh0) & np.isfinite(vh1)
    vv0n = vv0 - np.nanmedian(vv0[ov])
    vv1n = vv1 - np.nanmedian(vv1[ov])
    vh0n = vh0 - np.nanmedian(vh0[ov])
    vh1n = vh1 - np.nanmedian(vh1[ov])
    gain_vv = _nan_median_filter(vv1n - vv0n)
    gain_vh = _nan_median_filter(vh1n - vh0n)

    # land-type: gain in both polarizations on land that stayed land
    land = ~(w0.astype(bool) | w1.astype(bool))
    cand_land = (
        ov & land
        & (gain_vv > LAND_VV_GAIN_DB)
        & (gain_vh > LAND_VH_GAIN_DB)
    )
    # water-type: strong VV gain over baseline water (bridge, coffer dam, pier)
    cand_water = ov & w0.astype(bool) & (gain_vv > WATER_VV_GAIN_DB)

    alerts: list[ConstructionAlert] = []
    for cond, kind, min_px in ((cand_land, "land", MIN_PX_LAND), (cand_water, "water", MIN_PX_WATER)):
        labels, n = ndimage.label(cond, structure=np.ones((3, 3), dtype=int))
        if n == 0:
            continue
        counts = np.bincount(labels.ravel())[1:]
        valid_idx = np.where(counts >= min_px)[0] + 1
        if len(valid_idx) == 0:
            continue
        ys, xs = np.indices(cond.shape)
        label_idxs = valid_idx
        for li in label_idxs:
            px = counts[li - 1]
            mean_gain = float(ndimage.mean(gain_vv, labels, li))
            max_gain = float(np.nanmax(gain_vv[labels == li]))
            if not np.isfinite(mean_gain) or mean_gain < (LAND_VV_GAIN_DB if kind == "land" else WATER_VV_GAIN_DB):
                continue
            lon, lat = _to_lonlat(
                float(ndimage.mean(xs, labels, li)),
                float(ndimage.mean(ys, labels, li)),
            )
            vh_g = float(ndimage.mean(gain_vh, labels, li)) if kind == "land" else None
            area = px * (grid_deg * 111_000) ** 2 / 1e6
            severity = _severity(kind, area, mean_gain)
            if kind == "water":
                desc = (
                    f"Strong radar return over water on {d1}: possible bridge, "
                    f"coffer dam or pier construction (VV +{mean_gain:.1f} dB over {area:.3f} km²)."
                )
            else:
                desc = (
                    f"Strong backscatter gain on previously quiet land between "
                    f"{d0} and {d1} (+{mean_gain:.1f} dB VV, +{vh_g or 0.0:.1f} dB VH over {area:.3f} km²): "
                    f"possible excavation, roads or structures."
                )
            alerts.append(
                ConstructionAlert(
                    id=f"al-{d1}-{kind}-{len(alerts) + 1:03d}",
                    kind=kind,
                    date_baseline=d0,
                    date_detected=d1,
                    lon=round(lon, 5),
                    lat=round(lat, 5),
                    area_km2=round(area, 4),
                    pixels=int(px),
                    vv_gain_db=round(mean_gain, 2),
                    vv_gain_db_max=round(max_gain, 2),
                    vh_gain_db=round(vh_g, 2) if vh_g is not None else None,
                    severity=severity,
                    description=desc,
                )
            )
    alerts.sort(key=lambda a: (SEVERITY_ORDER[a.severity], a.area_km2), reverse=True)
    return alerts


def _severity(kind: str, area_km2: float, gain_db: float) -> str:
    if kind == "water" and area_km2 >= 0.01:
        return "HIGH"
    if area_km2 >= 0.05 and gain_db >= 6.0:
        return "HIGH"
    if area_km2 >= 0.02 or gain_db >= 5.0:
        return "MEDIUM"
    return "LOW"


def save_alerts(
    alerts: list[ConstructionAlert],
    out_dir: Path,
    background_db: np.ndarray | None,
    grid_transform,
    pair: str | None = None,
) -> Path:
    """Persist alerts.json, merge into alerts_history.json and write alerts_map.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if pair is None and alerts:
        pair = f"{alerts[0].date_baseline}__{alerts[0].date_detected}"
    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pair": pair,
        "n_alerts": len(alerts),
        "alerts": [asdict(a) for a in alerts],
    }
    path = out_dir / "alerts.json"
    path.write_text(json.dumps(payload, indent=2))

    history_path = out_dir / "alerts_history.json"
    history: dict = {}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            history = {}
    if pair:
        base, det = pair.split("__", 1)
        history[pair] = {
            "baseline": base,
            "detected": det,
            "n_alerts": len(alerts),
            "high": sum(1 for a in alerts if a.severity == "HIGH"),
            "medium": sum(1 for a in alerts if a.severity == "MEDIUM"),
            "low": sum(1 for a in alerts if a.severity == "LOW"),
            "latest": [asdict(a) for a in alerts],
        }
    history_path.write_text(json.dumps(history, indent=2))

    if background_db is not None and grid_transform is not None:
        _write_alerts_map(alerts, out_dir, background_db, grid_transform)
    return path


def _write_alerts_map(alerts, out_dir: Path, background_db: np.ndarray, grid_transform) -> None:
    """Simple overview PNG: VV dB background + alert markers coloured by severity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    t = grid_transform
    if abs(t.a) != abs(t.e):
        return
    a, b, c, d, e, f = t.a, t.b, t.c, t.d, t.e, t.f
    if abs(b) > 1e-9 or abs(d) > 1e-9:
        return  # rotated grid: skip the quick map
    rows, cols = background_db.shape
    extent = (c, c + a * cols, f + e * rows, f)

    cmap = LinearSegmentedColormap.from_list("db", ["#0b1220", "#274060", "#7d98b8", "#d8e2ee"])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(background_db, extent=extent, cmap=cmap, vmin=-25, vmax=-2, interpolation="nearest")
    colors = {"HIGH": "#e11d48", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
    for alert in alerts:
        ax.plot(
            alert.lon, alert.lat, "o",
            markersize=8 + 6 * (alert.severity == "HIGH"),
            color=colors[alert.severity],
            markeredgecolor="white", markeredgewidth=1,
            label=f"{alert.severity} {alert.kind} {alert.area_km2:.3f} km²",
        )
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Construction warnings — VV backscatter background (sigma0 dB)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "alerts_map.png", dpi=110)
    plt.close(fig)
