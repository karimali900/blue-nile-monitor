#!/usr/bin/env python3
"""Run the full monitoring pipeline for one or more sites.

Usage:
  python scripts/run_pipeline.py [--site gerd] [--since 2025-08-01] [--until 2026-08-17] [--no-osint] [--tag myrun]
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nilemonitor.config import load_config  # noqa: E402
from nilemonitor.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default=None, help="Site id from config/sites.yaml (default: all)")
    parser.add_argument("--since", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--until", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--no-osint", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    sites = [args.site] if args.site else [s.id for s in cfg.sites]
    for site_id in sites:
        out = run_pipeline(cfg, site_id, args.since, args.until, include_osint=not args.no_osint, run_tag=args.tag)
        print(f"OK {site_id}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())