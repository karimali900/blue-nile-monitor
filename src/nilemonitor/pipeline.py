"""End-to-end pipeline: satellite analysis -> maps -> OSINT -> report."""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from .analysis.timeseries import run_site_analysis, save_outputs
from .config import Config
from .osint.gather import gather_news
from .report import build_report
from .report_html import build_report_html
from .viz.maps import (
    render_construction_map,
    render_interactive_map,
    render_water_change_map,
)

log = logging.getLogger("nilemonitor")


def run_pipeline(
    cfg: Config,
    site_id: str,
    date_from: dt.date | None = None,
    date_to: dt.date | None = None,
    include_osint: bool | None = None,
    run_tag: str | None = None,
) -> Path:
    """Run the full pipeline for one site and return the output directory."""
    site = cfg.site(site_id)
    log.info("Analysing %s (%s) over %s → %s", site.name, site.id, date_from, date_to)
    analysis = run_site_analysis(cfg, site, date_from, date_to)
    out = save_outputs(cfg, analysis, run_tag)

    log.info("Running independent validation (JRC GSW + Copernicus DEM)")
    from .validation.run import run_validation

    run_validation(analysis, out, cfg.cache_dir)

    log.info("Rendering maps for %s", site.id)
    render_water_change_map(analysis, out)
    render_construction_map(analysis, out)
    render_interactive_map(analysis, out)

    osint = None
    use_osint = cfg.osint_config.get("enabled", True) if include_osint is None else include_osint
    if use_osint:
        log.info("Gathering open-source reporting for %s", site.id)
        osint = gather_news(cfg)

    report_path = build_report(analysis, osint, out)
    log.info("Report written: %s", report_path)
    html_path = build_report_html(analysis, osint, out)
    log.info("HTML report written: %s", html_path)
    return out