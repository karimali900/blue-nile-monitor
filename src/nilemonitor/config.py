"""Configuration loading: sites, pipeline settings, and environment secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Site:
    id: str
    name: str
    country: str
    river: str
    lon: float
    lat: float
    analysis_radius_km: float
    reservoir_upstream_km: float
    min_reservoir_elevation_m: float = 0.0

    @property
    def center_deg(self) -> tuple[float, float]:
        """(lon, lat) of the analysis center."""
        return (self.lon, self.lat)

    @property
    def bbox_deg(self) -> tuple[float, float, float, float]:
        """Bounding box (lon_min, lat_min, lon_max, lat_max) for the analysis area."""
        # Approximate: 1 deg lat ~ 111 km. Expand to cover the analysis circle.
        half_deg = self.analysis_radius_km / 111.0
        lon_half = half_deg / max(0.1, abs(__import__("math").cos(__import__("math").radians(self.lat))))
        return (
            self.lon - lon_half,
            self.lat - half_deg,
            self.lon + lon_half,
            self.lat + half_deg,
        )


@dataclass
class Config:
    sites: list[Site] = field(default_factory=list)
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    output_dir: Path = PROJECT_ROOT / "output"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    lookback_days: int = 365
    collection_config: dict[str, Any] = field(default_factory=dict)
    min_water_region_m2: float = 250_000.0
    osint_config: dict[str, Any] = field(default_factory=dict)
    use_sar: bool = True
    sar_max_items: int = 6
    landsat_max_items: int = 6

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "sites.yaml"
        with open(config_path) as fh:
            raw = yaml.safe_load(fh)

        sites = [Site(**s) for s in raw.get("sites", [])]
        pipe = raw.get("pipeline", {})
        osint = raw.get("osint", {})

        output_dir = Path(os.environ.get("NILE_OUTPUT_DIR", pipe.get("output_dir", "output")))
        cache_dir = Path(os.environ.get("NILE_CACHE_DIR", pipe.get("cache_dir", "data/cache")))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir

        return cls(
            sites=sites,
            stac_url=pipe.get("stac_url", cls.stac_url),
            output_dir=output_dir,
            cache_dir=cache_dir,
            lookback_days=pipe.get("default_lookback_days", 365),
            collection_config=pipe.get("collection_config", {}),
            min_water_region_m2=pipe.get("min_water_region_m2", 250_000.0),
            osint_config=osint,
            use_sar=pipe.get("use_sar", True),
            sar_max_items=int(pipe.get("sar_max_items", 6)),
            landsat_max_items=int(pipe.get("landsat_max_items", 6)),
        )

    def site(self, site_id: str) -> Site:
        for s in self.sites:
            if s.id == site_id:
                return s
        raise KeyError(f"Unknown site: {site_id}")


def load_config(config_path: str | Path | None = None) -> Config:
    return Config.load(config_path)
