"""Static (matplotlib) and interactive (folium) map outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from shapely.geometry import shape

from ..analysis.timeseries import COMMON_GRID_DEG, SiteAnalysis


def render_water_change_map(analysis: SiteAnalysis, out: Path, title: str | None = None) -> Path:
    """Latest vs baseline water extent overlay (green=gained, red=lost)."""
    grid, transform = analysis.common_grid
    h, w = grid.shape
    east = transform.c
    north = transform.f
    extent = (east, east + w * transform.a, north + h * transform.e, north)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#111111")
    ax.set_axis_off()

    collections = [c for c in analysis.latest_water if c in analysis.baseline_water]
    for collection in collections:
        latest = analysis.latest_water[collection].astype(bool)
        baseline = analysis.baseline_water[collection].astype(bool)
        if latest.shape != baseline.shape:
            continue
        gained = latest & ~baseline
        lost = ~latest & baseline
        ax.imshow(latest, extent=extent, cmap="Blues", vmin=0, vmax=1, alpha=0.85)
        ax.imshow(gained, extent=extent, cmap=matplotlib.colors.ListedColormap(["none", "#00ff00"]), vmin=0, vmax=1, alpha=0.7)
        ax.imshow(lost, extent=extent, cmap=matplotlib.colors.ListedColormap(["none", "#ff3333"]), vmin=0, vmax=1, alpha=0.7)
        break

    ax.scatter([analysis.site.lon], [analysis.site.lat], marker="*", c="#ffdd00", s=220, edgecolors="black", zorder=5, label="Site")
    ax.legend(loc="lower right", frameon=True)
    ax.set_title(title or f"Water extent change — {analysis.site.name}\n{analysis.date_from} → {analysis.date_to}")

    path = out / "water_change_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#111111")
    plt.close(fig)
    return path


def render_construction_map(analysis: SiteAnalysis, out: Path, title: str | None = None) -> Path:
    if not analysis.construction:
        return out / "construction_map.png"
    idx, thresh = analysis.construction["sentinel-1-grd"]
    grid, transform = analysis.common_grid
    h, w = grid.shape
    east = transform.c
    north = transform.f
    extent = (east, east + w * transform.a, north + h * transform.e, north)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#101010")
    ax.set_axis_off()
    valid = np.isfinite(idx)
    img = np.zeros((h, w, 3))
    img[valid] = plt.cm.inferno((idx[valid] - np.nanmin(idx)) / max(1e-9, np.nanmax(idx) - np.nanmin(idx)))[:, :3]
    ax.imshow(img, extent=extent)
    ax.scatter([analysis.site.lon], [analysis.site.lat], marker="*", c="#ffdd00", s=220, edgecolors="black", zorder=5)
    ax.set_title(title or f"SAR construction/activity index — {analysis.site.name}\n{analysis.date_from} → {analysis.date_to}")

    path = out / "construction_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#101010")
    plt.close(fig)
    return path


def render_validation_map(analysis: SiteAnalysis, gsw_ref, out: Path, title: str | None = None) -> Path:
    """Our latest water vs the JRC GSW 'ever water' reference (green=ours, red=ours not in ref)."""
    path = out / "validation_map.png"
    if gsw_ref is None:
        return path
    ref_mask, ref_tf = gsw_ref
    grid, transform = analysis.common_grid
    h, w = grid.shape
    east = transform.c
    north = transform.f
    extent = (east, east + w * transform.a, north + h * transform.e, north)

    from nilemonitor.validation.metrics import _warp_nearest

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#111111")
    ax.set_axis_off()

    pred = _warp_nearest(np.where(np.isfinite(grid), 0.0, np.nan), transform, ref_mask.shape, ref_tf)
    valid_ref = np.isfinite(pred)
    ax.imshow(np.where(valid_ref, ref_mask, False), extent=(ref_tf.c, ref_tf.c + ref_mask.shape[1] * ref_tf.a,
                                                            ref_tf.f + ref_mask.shape[0] * ref_tf.e, ref_tf.f),
              cmap=matplotlib.colors.ListedColormap(["none", "#2563eb"]), vmin=0, vmax=1, alpha=0.85)

    for collection in analysis.latest_water:
        mask = analysis.latest_water[collection]
        pred_mask = _warp_nearest(np.where(np.isfinite(mask), mask, np.nan), transform, ref_mask.shape, ref_tf)
        valid = np.isfinite(pred_mask)
        detected = valid & (pred_mask > 0.5)
        not_in_ref = detected & ~ref_mask
        ax.imshow(detected, extent=(ref_tf.c, ref_tf.c + ref_mask.shape[1] * ref_tf.a,
                                    ref_tf.f + ref_mask.shape[0] * ref_tf.e, ref_tf.f),
                  cmap=matplotlib.colors.ListedColormap(["none", "#22c55e"]), vmin=0, vmax=1, alpha=0.95)
        ax.imshow(not_in_ref, extent=(ref_tf.c, ref_tf.c + ref_mask.shape[1] * ref_tf.a,
                                      ref_tf.f + ref_mask.shape[0] * ref_tf.e, ref_tf.f),
                  cmap=matplotlib.colors.ListedColormap(["none", "#ef4444"]), vmin=0, vmax=1, alpha=0.9)
        break

    ax.scatter([analysis.site.lon], [analysis.site.lat], marker="*", c="#ffdd00", s=220, edgecolors="black", zorder=5, label="Site")
    ax.set_title(title or f"Independent validation — {analysis.site.name}\n"
                          f"blue: JRC GSW reference (1984–2021) · green: detected · red: not in reference")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#111111")
    plt.close(fig)
    return path


def render_interactive_map(analysis: SiteAnalysis, out: Path) -> Path:
    """folium HTML map: water latest + construction heat layer + site marker."""
    import folium
    from folium.plugins import HeatMap

    m = folium.Map(location=[analysis.site.lat, analysis.site.lon], zoom_start=11,
                   tiles="Esri WorldImagery")
    folium.Marker(
        [analysis.site.lat, analysis.site.lon],
        popup=analysis.site.name,
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    grid, transform = analysis.common_grid
    for collection, mask in analysis.latest_water.items():
        feats = polygons_from_grid(mask, transform)
        if feats:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": feats},
                name=f"water latest ({collection})",
                style_function=lambda f: {"color": "#1f6fff", "weight": 1, "fillColor": "#1f6fff", "fillOpacity": 0.55},
            ).add_to(m)

    if analysis.construction:
        idx, _ = analysis.construction["sentinel-1-grd"]
        valid = np.isfinite(idx)
        pts = np.where(valid)
        heat = [[float(analysis.site.lat) + (r - grid.shape[0] / 2) * transform.e,
                 float(analysis.site.lon) + (c - grid.shape[1] / 2) * transform.a,
                 float(np.clip(idx[r, c], 0, 1))] for r, c in zip(pts[0], pts[1])]
        if heat:
            HeatMap(heat, name="construction activity", radius=6, blur=6, max_zoom=12).add_to(m)

    folium.LayerControl().add_to(m)
    path = out / "interactive_map.html"
    m.save(str(path))
    return path


def polygons_from_grid(mask: np.ndarray, transform):
    features = []
    if mask is None or not np.any(mask):
        return features
    px_km2 = (abs(transform.a) * 111_000) ** 2 / 1e6
    clean = np.where(np.isfinite(mask), mask, 0).astype("uint8")
    for geom, value in rasterio.features.shapes(clean, transform=transform):
        if value != 1:
            continue
        area = shape(geom).area * px_km2
        if area < 0.05:
            continue
        features.append({"type": "Feature", "properties": {"area_km2": round(area, 3)}, "geometry": geom})
    return features
