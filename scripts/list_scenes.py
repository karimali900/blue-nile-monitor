#!/usr/bin/env python3
"""Query the STAC catalog for available scenes over a site (no analysis).

Usage:
  python scripts/list_scenes.py [--site gerd] [--collection sentinel-2-l2a] [--days 90]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nilemonitor.config import load_config  # noqa: E402
from nilemonitor.download.stac import search_all_collections  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="gerd")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    cfg = load_config()
    site = cfg.site(args.site)
    date_to = dt.date.today()
    date_from = date_to - dt.timedelta(days=args.days)

    collections = (
        {args.collection: cfg.collection_config[args.collection]}
        if args.collection
        else cfg.collection_config
    )
    scenes = search_all_collections(cfg.stac_url, site.bbox_deg, date_from, date_to, collections)
    for collection, items in scenes.items():
        print(f"\n== {collection}: {len(items)} scenes ==")
        for s in items[:10]:
            cloud = f"{s.cloud_cover:.1f}%" if s.cloud_cover is not None else "n/a"
            print(f"  {s.date}  {s.item_id[:60]:60s} cloud={cloud}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())