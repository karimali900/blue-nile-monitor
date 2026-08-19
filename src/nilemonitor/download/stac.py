"""Search Earth Search STAC (public AWS open data) for analysis-ready COGs.

Every item we return is a Cloud-Optimized GeoTIFF asset, so we never download
whole scenes — individual bands are read with HTTP range requests by rasterio.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable

from pystac_client import Client

COLLECTIONS = ("sentinel-2-l2a", "sentinel-1-grd", "landsat-c2-l2")

# Which asset keys map to which physical band / use
ASSET_PRIORITY = {
    "sentinel-2-l2a": ["green", "nir", "scl", "visual"],
    "sentinel-1-grd": ["vv", "vh", "visual"],
    "landsat-c2-l2": ["green", "swir16", "qa_pixel", "visual"],
}


@dataclass
class Scene:
    collection: str
    item_id: str
    datetime: dt.datetime
    bbox: tuple[float, float, float, float]
    cloud_cover: float | None
    assets: dict[str, str]  # asset key -> COG href
    properties: dict[str, Any]

    @property
    def date(self) -> dt.date:
        return self.datetime.date()

    def href(self, key: str) -> str | None:
        return self.assets.get(key)

    def georef_override(self) -> tuple[tuple[float, float, float, float, float, float], str] | None:
        """STAC-item georeferencing for assets that ship un-georeferenced (Sentinel-1).

        Returns ((a, b, c, d, e, f) affine, crs string) or None.
        """
        tf = self.properties.get("proj:transform")
        code = self.properties.get("proj:code")
        if tf and code:
            return tuple(tf), code
        return None


def _get_client(stac_url: str) -> Client:
    return Client.open(stac_url)


def search_scenes(
    stac_url: str,
    collection: str,
    bbox: tuple[float, float, float, float],
    date_from: dt.date,
    date_to: dt.date,
    max_items: int = 24,
    max_cloud_cover: float | None = None,
) -> list[Scene]:
    """Search a single collection for COG scenes intersecting `bbox`.

    Returns scenes sorted newest-first.
    """
    if collection not in COLLECTIONS:
        raise ValueError(f"Unsupported collection {collection!r}; choose from {COLLECTIONS}")

    client = _get_client(stac_url)
    query = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=(f"{date_from.isoformat()}T00:00:00Z", f"{date_to.isoformat()}T23:59:59Z"),
        max_items=max_items,
    )
    items = query.items()

    scenes: list[Scene] = []
    for item in items:
        cloud = item.properties.get("eo:cloud_cover")
        if max_cloud_cover is not None and cloud is not None and cloud > max_cloud_cover:
            continue
        assets = {}
        for key in ASSET_PRIORITY.get(collection, []):
            if key in item.assets:
                assets[key] = item.assets[key].href
        scenes.append(
            Scene(
                collection=collection,
                item_id=item.id,
                datetime=item.datetime,
                bbox=tuple(item.bbox),
                cloud_cover=cloud,
                assets=assets,
                properties=item.properties,
            )
        )
    scenes.sort(key=lambda s: s.datetime, reverse=True)
    scenes = _dedupe_by_date(scenes)
    return scenes[:max_items]


def _dedupe_by_date(scenes: list[Scene]) -> list[Scene]:
    """Adjacent granule tiles of the same date overlap; keep the least cloudy one."""
    by_date: dict[dt.date, Scene] = {}
    for scene in scenes:
        cur = by_date.get(scene.date)
        if cur is None:
            by_date[scene.date] = scene
            continue
        cur_cloud = cur.cloud_cover if cur.cloud_cover is not None else 999.0
        new_cloud = scene.cloud_cover if scene.cloud_cover is not None else 999.0
        if new_cloud < cur_cloud:
            by_date[scene.date] = scene
    return sorted(by_date.values(), key=lambda s: s.datetime, reverse=True)


def search_all_collections(
    stac_url: str,
    bbox: tuple[float, float, float, float],
    date_from: dt.date,
    date_to: dt.date,
    collection_config: dict[str, dict[str, Any]],
) -> dict[str, list[Scene]]:
    """Search every configured collection and bucket the results by collection id."""
    out: dict[str, list[Scene]] = {}
    for collection, cfg in collection_config.items():
        out[collection] = search_scenes(
            stac_url,
            collection,
            bbox,
            date_from,
            date_to,
            max_items=cfg.get("max_items", 24),
            max_cloud_cover=cfg.get("max_cloud_cover"),
        )
    return out
