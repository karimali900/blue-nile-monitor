"""Markdown report generation merging satellite metrics, maps and OSINT."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .analysis.timeseries import SiteAnalysis
from .osint.gather import OsintResult, save_osint


def _fmt_delta(delta_km2: float) -> str:
    sign = "+" if delta_km2 >= 0 else "−"
    return f"{sign}{abs(delta_km2):,.2f} km²"


def build_report(
    analysis: SiteAnalysis,
    osint: OsintResult | None,
    out_dir: Path,
) -> Path:
    """Write report.md into out_dir and return its path."""
    site = analysis.site
    series = analysis.series

    lines: list[str] = []
    lines.append(f"# {site.name} — monitoring report")
    lines.append("")
    lines.append(
        f"- **River:** {site.river} ({site.country})"
    )
    lines.append(f"- **Site coordinates:** {site.lat:.4f} N, {site.lon:.4f} E")
    lines.append(f"- **Analysis window:** {analysis.date_from} → {analysis.date_to}")
    lines.append(f"- **Generated:** {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")
    lines.append(f"- **Scenes analysed:** {len(analysis.readings)}")
    lines.append("")

    # Per-collection summary
    lines.append("## Satellite water-extent summary")
    lines.append("")
    lines.append("| Collection | Scenes | Water (baseline) | Water (latest) | Δ |")
    lines.append("|---|---|---|---|---|")
    for collection, df in series.groupby("collection"):
        df = df.sort_values("date")
        base = df.iloc[0]["water_area_km2"]
        last = df.iloc[-1]["water_area_km2"]
        lines.append(
            f"| {collection} | {len(df)} | {base:,.2f} km² | {last:,.2f} km² | {_fmt_delta(last - base)} |"
        )
    lines.append("")
    if "sentinel-1-grd" in series["collection"].values:
        lines.append(
            "> **SAR caveat:** Sentinel-1 water areas over rugged terrain include radar "
            "shadow (canyon walls / steep slopes), which is indistinguishable from calm "
            "water in a single image. Treat the SAR water extent as an upper bound; "
            "multi-date filtering will refine it as the archive grows."
        )
        lines.append("")

    # Peak / largest step
    if len(series) >= 2:
        piv = series.pivot_table(index="date", columns="collection", values="water_area_km2")
        for collection in piv.columns:
            col = piv[collection].dropna()
            if len(col) < 2:
                continue
            growth = col.diff()
            peak_date = col.idxmax()
            biggest = growth.abs().idxmax()
            lines.append(
                f"- **{collection}**: peak water extent {col.max():,.2f} km² on {peak_date}; "
                f"largest single-observation change {growth.loc[biggest]:+.2f} km² on {biggest}."
            )
        lines.append("")

    # Full table
    lines.append("## Full observation series")
    lines.append("")
    lines.append("| Date | Collection | Water area (km²) | NDWI mean | Cloud (%) |")
    lines.append("|---|---|---|---|---|")
    for _, row in series.sort_values(["date", "collection"]).iterrows():
        lines.append(
            f"| {row['date']} | {row['collection']} | {row['water_area_km2']:,.2f} "
            f"| {row['ndwi_mean']:.3f} | {row['cloud_cover'] if pd.notna(row['cloud_cover']) else ''} |"
        )
    lines.append("")

    # Construction index
    if analysis.construction:
        idx, thresh = analysis.construction["sentinel-1-grd"]
        import numpy as np

        valid = np.isfinite(idx)
        if not valid.any():
            lines.append("## Construction / activity index (Sentinel-1 SAR)")
            lines.append("")
            lines.append(
                "- Not computed: the construction index needs **≥ 2 SAR scenes** in the "
                "analysis window; only one scene had sufficient AOI coverage this run."
            )
            lines.append("")
        else:
            high = int(np.nansum(idx[valid] >= thresh * 0.9))
            km2 = high * (0.0005 * 111_000) ** 2 / 1e6
            lines.append("## Construction / activity index (Sentinel-1 SAR)")
            lines.append("")
            lines.append(
                f"- High-activity pixels (≥ 90% of the Otsu-derived reference): **{high:,}** "
                f"(≈ {km2:,.1f} km² of sustained backscatter change within the analysis area)."
            )
            lines.append(
                "- Interpretation: persistent SAR backscatter change plus water-state flips are "
                "consistent with dam construction, reservoir filling and land clearing."
            )
            lines.append("")

    # Construction warnings
    if analysis.alerts_pair:
        lines.append("## Construction warnings")
        lines.append("")
        if not analysis.alerts:
            lines.append(
                f"- **No construction detections** in the SAR pair "
                f"{analysis.alerts_pair.replace('__', ' → ')}: no VV/VH backscatter "
                "gains above the detector thresholds. Radar shadow and reservoir "
                "water-fill pixels are excluded from detection."
            )
            lines.append("")
        else:
            lines.append(
                f"- {len(analysis.alerts)} detection(s) from the SAR pair "
                f"{analysis.alerts_pair.replace('__', ' → ')}. "
                "Detections are VV/VH backscatter gains (median-filtered, clustered); radar "
                "shadow and reservoir water-fill pixels are excluded. Verify before acting."
            )
            lines.append("")
            lines.append("| # | Severity | Type | Detected | Area (km²) | VV gain (dB) | Location |")
            lines.append("|---|---|---|---|---|---|---|")
            for i, a in enumerate(analysis.alerts, 1):
                lines.append(
                    f"| {i} | {a.severity} | {a.kind} | {a.date_detected} | {a.area_km2:,.3f} | "
                    f"+{a.vv_gain_db:.1f} | {a.lat:.4f} N, {a.lon:.4f} E |"
                )
            lines.append("")
            lines.append(
                "Detected change is automatic and unverified: a sustained VV gain can also "
                "come from wetting, burning or surface roughness changes. Treat every alert "
                "as a **candidate** pending ground truth or high-resolution imagery."
            )
            lines.append("")

    # Independent validation
    vp = out_dir / "validation.json"
    if vp.exists():
        try:
            import json

            val = json.loads(vp.read_text())
            colls = val.get("collections", {})
            if colls:
                lines.append("## Independent validation")
                lines.append("")
                lines.append(
                    "Water masks are compared against the **JRC Global Surface Water "
                    "v1.4 (2021) maximum-extent reference** (Landsat archive 1984–2021, "
                    "Pekel et al., Nature 2016) on a shared 30 m grid; water is the "
                    "positive class. Reference coverage ends in 2021, so pixels the "
                    "pipeline detects that the reference lacks are expected where water "
                    "appeared after 2021 (e.g. the ongoing reservoir fill)."
                )
                lines.append("")
                lines.append("| Collection | OA | Precision | Recall | F1 | IoU | Kappa |")
                lines.append("|---|---|---|---|---|---|---|")
                for collection, entry in colls.items():
                    m = entry.get("gsw_metrics") or {}
                    if "error" in m:
                        lines.append(f"| {collection} | — | — | — | — | — | — |")
                        continue
                    lines.append(
                        f"| {collection} | {m.get('overall_accuracy'):.3f} | "
                        f"{m.get('precision_user'):.3f} | {m.get('recall_producer'):.3f} | "
                        f"{m.get('f1'):.3f} | {m.get('iou'):.3f} | {m.get('kappa'):.3f} |"
                    )
                lines.append("")
                for collection, entry in colls.items():
                    s = entry.get("storage")
                    if not s or "note" not in s:
                        continue
                    lines.append(
                        f"- **{collection}** reservoir storage proxy (Copernicus DEM GLO-30, "
                        f"water level at the GERD full-supply elevation {s['crest_level_m']:.0f} m a.s.l.): "
                        f"**{s['volume_km3']:.2f} km³** over {s['area_km2']:,.1f} km² "
                        f"(mean depth {s['mean_depth_m']:.0f} m). {s['note']}."
                    )
                lines.append("")
                if (out_dir / "validation_map.png").exists():
                    lines.append("![Independent validation](validation_map.png)")
                    lines.append("")
        except (ValueError, KeyError, OSError):
            lines.append("## Independent validation")
            lines.append("")
            lines.append("- Validation data present but unreadable.")
            lines.append("")

    # Maps
    lines.append("## Maps")
    lines.append("")
    lines.append("![Water extent change](water_change_map.png)")
    lines.append("")
    if (out_dir / "construction_map.png").exists():
        lines.append("![Construction activity](construction_map.png)")
        lines.append("")
    if (out_dir / "interactive_map.html").exists():
        lines.append("- [Interactive map](interactive_map.html)")
        lines.append("")

    # OSINT
    lines.append("## Open-source reporting")
    lines.append("")
    if osint and osint.items:
        for item in osint.items[:12]:
            pub = item.published.strftime("%Y-%m-%d") if item.published else "n/a"
            lines.append(f"- **{item.title}** — {item.source} ({pub})")
            if item.url:
                lines.append(f"  {item.url}")
        lines.append("")
    else:
        lines.append("No qualifying open-source items fetched in this run.")
        lines.append("")

    if osint and osint.errors:
        lines.append("### Fetch warnings")
        lines.append("")
        for err in osint.errors[:5]:
            lines.append(f"- `{err}`")
        lines.append("")

    lines.append("---")
    lines.append(
        "*Automatically generated by nilemonitor. Water masks: Sentinel-2 NDWI / "
        "Landsat MNDWI / Sentinel-1 SAR backscatter. Figures are estimates; validate "
        "critical claims against full-resolution imagery.*"
    )
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    if osint:
        save_osint(osint, out_dir)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path