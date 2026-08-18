#!/usr/bin/env python3
"""Run the monitoring pipeline for one or all sites and print a summary.

Usage:
    python scripts/run_analysis.py [--site gerd] [--from 2025-01-01] [--to 2026-08-01]
                                   [--no-osint] [--config config/sites.yaml] [--tag myrun]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nilemonitor.config import load_config  # noqa: E402
from nilemonitor.pipeline import run_pipeline  # noqa: E402


def parse_date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Nile satellite monitoring pipeline")
    ap.add_argument("--site", default=None, help="site id from config/sites.yaml (default: all)")
    ap.add_argument("--from", dest="date_from", default=None, help="start date YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", default=None, help="end date YYYY-MM-DD")
    ap.add_argument("--no-osint", action="store_true", help="skip the OSINT layer")
    ap.add_argument("--config", default=None, help="path to sites.yaml")
    ap.add_argument("--tag", default=None, help="run tag for the output directory")
    args = ap.parse_args()

    cfg = load_config(args.config)
    date_from = parse_date(args.date_from)
    date_to = parse_date(args.date_to)

    sites = [cfg.site(args.site)] if args.site else cfg.sites
    if not sites:
        print("No sites configured.", file=sys.stderr)
        return 1

    for site in sites:
        print(f"\n=== {site.name} ===")
        out = run_pipeline(
            cfg, site.id, date_from, date_to,
            include_osint=not args.no_osint, run_tag=args.tag,
        )
        print(f"Outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())