"""Self-contained HTML report with satellite imagery, charts, maps and alerts."""
from __future__ import annotations

import base64
import datetime as dt
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.timeseries import SiteAnalysis
from .osint.gather import OsintResult


def _b64(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def _series_chart_png(series: pd.DataFrame) -> str | None:
    if series.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for collection, df in series.groupby("collection"):
        df = df.sort_values("date")
        ax.plot(pd.to_datetime(df["date"]), df["water_area_km2"], marker="o", ms=5, label=collection)
    ax.set_ylabel("Water area (km²)")
    ax.set_title("Water-extent time series")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _severity_color(sev: str) -> str:
    return {"HIGH": "#e11d48", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(sev, "#64748b")


def build_report_html(
    analysis: SiteAnalysis,
    osint: OsintResult | None,
    out_dir: Path,
) -> Path:
    """Write report.html (self-contained, embedded imagery) into out_dir."""
    site = analysis.site
    series = analysis.series.copy()
    series["date"] = pd.to_datetime(series["date"])

    chart = _series_chart_png(series)

    # Per-collection summary rows
    summary_rows = ""
    for collection, df in series.groupby("collection"):
        df = df.sort_values("date")
        base, last = df.iloc[0]["water_area_km2"], df.iloc[-1]["water_area_km2"]
        delta = last - base
        arrow = "&#9650;" if delta >= 0 else "&#9660;"
        color = "#e11d48" if delta > 0 else "#16a34a"
        summary_rows += (
            f"<tr><td>{collection}</td><td>{len(df)}</td>"
            f"<td>{base:,.2f} km²</td><td><b>{last:,.2f} km²</b></td>"
            f"<td style='color:{color}'>{arrow} {abs(delta):,.2f} km²</td></tr>"
        )

    series_rows = "".join(
        f"<tr><td>{r['date'].date()}</td><td>{r['collection']}</td>"
        f"<td>{r['water_area_km2']:,.2f}</td>"
        f"<td>{r['ndwi_mean']:.3f}</td>"
        f"<td>{r['cloud_cover']:.1f}</td></tr>"
        for _, r in series.sort_values(["date", "collection"]).iterrows()
        if pd.notna(r["water_area_km2"])
    )

    # Satellite imagery gallery
    gallery = ""
    for item_id, p in sorted(analysis.preview_pngs.items(), key=lambda kv: kv[1]["date"]):
        png = base64.b64encode(p["png"]).decode()
        gallery += (
            f"<figure><img src='data:image/png;base64,{png}' alt='{p['caption']}'>"
            f"<figcaption>{p['caption']}<br><small>{item_id[:20]}</small></figcaption></figure>"
        )

    # Maps
    maps_html = ""
    for label, fname in (
        ("Water change", "water_change_map.png"),
        ("Construction activity", "construction_map.png"),
        ("Construction warnings", "alerts_map.png"),
    ):
        b = _b64(out_dir / fname)
        if b:
            maps_html += (
                f"<section class='card'><h2>{label}</h2>"
                f"<img src='data:image/png;base64,{b}' class='map'></section>"
            )

    # Warnings
    warnings_html = ""
    if analysis.alerts_pair:
        if not analysis.alerts:
            warnings_html = (
                "<section class='card'><h2>Construction warnings</h2>"
                "<p class='ok'>No construction detections in the SAR pair "
                f"{analysis.alerts_pair.replace('__', ' &rarr; ')}. "
                "Radar shadow and reservoir water-fill pixels are excluded from detection.</p></section>"
            )
        else:
            rows = "".join(
                f"<tr><td><span class='sev' style='background:{_severity_color(a.severity)}'>{a.severity}</span></td>"
                f"<td>{a.kind}</td><td>{a.date_detected}</td><td>{a.area_km2:,.3f}</td>"
                f"<td>+{a.vv_gain_db:.1f} dB</td>"
                f"<td>{a.lat:.4f} N, {a.lon:.4f} E</td></tr>"
                for a in analysis.alerts
            )
            warnings_html = (
                "<section class='card'><h2>Construction warnings</h2>"
                f"<p>{len(analysis.alerts)} unverified candidate detection(s) from the SAR pair "
                f"{analysis.alerts_pair.replace('__', ' &rarr; ')}. Verify before acting.</p>"
                "<table><tr><th>Severity</th><th>Type</th><th>Detected</th><th>Area (km²)</th>"
                "<th>VV gain</th><th>Location</th></tr>" + rows + "</table></section>"
            )

    # OSINT
    osint_html = ""
    if osint and osint.items:
        items = "".join(
            f"<li><b>{i.title}</b> — {i.source} ({i.published.strftime('%Y-%m-%d') if i.published else 'n/a'})"
            + (f"<br><a href='{i.url}'>{i.url}</a>" if i.url else "")
            + "</li>"
            for i in osint.items[:12]
        )
        osint_html = f"<section class='card'><h2>Open-source reporting</h2><ul>{items}</ul></section>"

    # Independent validation
    validation_html = ""
    vp = out_dir / "validation.json"
    if vp.exists():
        try:
            import json as _json

            val = _json.loads(vp.read_text())
            colls = val.get("collections", {})
            if colls:
                rows = ""
                for collection, entry in colls.items():
                    m = entry.get("gsw_metrics") or {}
                    if "error" in m:
                        rows += f"<tr><td>{collection}</td><td colspan='6'>—</td></tr>"
                        continue
                    rows += (
                        f"<tr><td>{collection}</td><td>{m.get('overall_accuracy'):.3f}</td>"
                        f"<td>{m.get('precision_user'):.3f}</td><td>{m.get('recall_producer'):.3f}</td>"
                        f"<td>{m.get('f1'):.3f}</td><td>{m.get('iou'):.3f}</td><td>{m.get('kappa'):.3f}</td></tr>"
                    )
                storage_p = ""
                for collection, entry in colls.items():
                    s = entry.get("storage")
                    if s and "note" in s:
                        storage_p += (
                            f"<p><b>{collection}</b> reservoir storage proxy (Copernicus DEM GLO-30, "
                            f"water level at GERD full-supply elevation {s['crest_level_m']:.0f} m a.s.l.): "
                            f"<b>{s['volume_km3']:.2f} km&sup3;</b> over {s['area_km2']:,.1f} km&sup2; "
                            f"(mean depth {s['mean_depth_m']:.0f} m). {s['note']}.</p>"
                        )
                val_img = _b64(out_dir / "validation_map.png")
                img = (
                    f"<img src='data:image/png;base64,{val_img}' class='map' style='margin-top:12px'>"
                    if val_img
                    else ""
                )
                validation_html = (
                    "<section class='card'><h2>Independent validation</h2>"
                    "<p>Water masks vs the <b>JRC Global Surface Water v1.4 (2021) maximum-extent "
                    "reference</b> (Landsat 1984&ndash;2021, Pekel et al., Nature 2016) on a shared "
                    "30 m grid. Pixels detected by the pipeline but absent from the reference are "
                    "expected where water appeared after 2021 (ongoing reservoir fill).</p>"
                    "<table><tr><th>Collection</th><th>OA</th><th>Precision</th><th>Recall</th>"
                    "<th>F1</th><th>IoU</th><th>Kappa</th></tr>" + rows + "</table>"
                    + storage_p + img + "</section>"
                )
        except (ValueError, KeyError, OSError) as e:
            validation_html = (
                "<section class='card'><h2>Independent validation</h2>"
                f"<p class='ok'>Could not load validation data: {e}</p></section>"
            )

    # Construction index summary
    constr_html = ""
    if analysis.construction:
        idx, thresh = analysis.construction["sentinel-1-grd"]
        valid = np.isfinite(idx)
        high = int(np.nansum(idx[valid] >= thresh * 0.9)) if valid.any() else 0
        km2 = high * (0.0005 * 111_000) ** 2 / 1e6
        constr_html = (
            "<section class='card'><h2>Construction / activity index</h2>"
            f"<p>{high:,} high-activity pixels (&asymp; {km2:,.1f} km² of sustained SAR change "
            "within the analysis area) — consistent with construction, reservoir filling "
            "and land clearing.</p></section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{site.name} — Monitoring report</title>
<style>
  :root {{ --ink:#0f172a; --muted:#64748b; --brand1:#0f2a4a; --brand2:#0e7490; --line:#e2e8f0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
         color:var(--ink); background:#f1f5f9; line-height:1.55; }}
  header {{ background:linear-gradient(120deg,#0b2a4a 0%,#0e7490 65%,#14b8a6 100%);
           color:#fff; padding:34px 6vw 30px; }}
  header h1 {{ margin:0 0 6px; font-size:26px; letter-spacing:.3px; }}
  header p {{ margin:2px 0; color:#cbe8f2; font-size:14px; }}
  .chips {{ margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }}
  .chip {{ background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25);
          padding:6px 14px; border-radius:999px; font-size:13px; }}
  main {{ max-width:1080px; margin:26px auto 60px; padding:0 6vw; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 18px;
         box-shadow:0 1px 3px rgba(15,23,42,.06); }}
  .kpi .k {{ font-size:12px; text-transform:uppercase; letter-spacing:.8px; color:var(--muted); }}
  .kpi .v {{ font-size:22px; font-weight:700; margin-top:2px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px 22px;
          margin:18px 0; box-shadow:0 1px 3px rgba(15,23,42,.06); }}
  .card h2 {{ margin:0 0 12px; font-size:17px; color:#0f2a4a; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th {{ text-align:left; color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); }}
  td {{ border-bottom:1px solid var(--line); padding:7px 4px; }}
  .sev {{ color:#fff; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
  .ok {{ color:#16a34a; font-weight:600; }}
  .map {{ width:100%; border-radius:8px; border:1px solid var(--line); }}
  figure {{ margin:0; border:1px solid var(--line); border-radius:10px; overflow:hidden;
           background:#000; }}
  figure img {{ width:100%; height:180px; object-fit:cover; display:block; }}
  figcaption {{ background:#0f172a; color:#e2e8f0; font-size:12px; padding:8px 10px; }}
  ul {{ padding-left:18px; }}
  li {{ margin:8px 0; font-size:14px; }}
  a {{ color:#0e7490; }}
  footer {{ color:var(--muted); font-size:12px; text-align:center; padding:20px; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; }}
</style>
</head>
<body>
<header>
  <h1>&#128752; {site.name} — monitoring report</h1>
  <p>{site.river} · {site.country} &nbsp;|&nbsp; {site.lat:.4f} N, {site.lon:.4f} E</p>
  <p>Analysis window: {analysis.date_from} &rarr; {analysis.date_to}
     &nbsp;·&nbsp; Generated: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC
     &nbsp;·&nbsp; Scenes analysed: {len(analysis.readings)}</p>
  <div class="chips">
    <span class="chip">S2 water (latest): {series[series['collection']=='sentinel-2-l2a']['water_area_km2'].iloc[-1]:,.1f} km²</span>
    <span class="chip">S1 water (latest): {series[series['collection']=='sentinel-1-grd']['water_area_km2'].iloc[-1]:,.1f} km²</span>
    <span class="chip">Warnings: {len(analysis.alerts)}</span>
    <span class="chip">OSINT items: {len(osint.items) if osint and osint.items else 0}</span>
  </div>
</header>
<main>
  <div class="grid">
    <div class="kpi"><div class="k">Sentinel-2 water</div><div class="v">{series[series['collection']=='sentinel-2-l2a']['water_area_km2'].iloc[-1]:,.1f} km²</div></div>
    <div class="kpi"><div class="k">Sentinel-1 water</div><div class="v">{series[series['collection']=='sentinel-1-grd']['water_area_km2'].iloc[-1]:,.1f} km²</div></div>
    <div class="kpi"><div class="k">Construction warnings</div><div class="v">{len(analysis.alerts)}</div></div>
    <div class="kpi"><div class="k">Scenes analysed</div><div class="v">{len(analysis.readings)}</div></div>
  </div>

  <section class="card"><h2>Water-extent time series</h2>
    <img src="data:image/png;base64,{chart}" class="map" alt="time series"></section>

  <section class="card"><h2>Satellite water-extent summary</h2>
    <table><tr><th>Collection</th><th>Scenes</th><th>Water (baseline)</th>
    <th>Water (latest)</th><th>&Delta;</th></tr>{summary_rows}</table></section>

  <section class="card"><h2>Full observation series</h2>
    <table><tr><th>Date</th><th>Collection</th><th>Water area (km²)</th>
    <th>NDWI mean</th><th>Cloud (%)</th></tr>{series_rows}</table></section>

  {warnings_html}
  {constr_html}
  {validation_html}

  <section class="card"><h2>Satellite imagery</h2>
    <div class="gallery">{gallery}</div></section>

  {maps_html}
  {osint_html}
</main>
<footer>
  Automatically generated by nilemonitor — Sentinel-2 NDWI, Sentinel-1 SAR backscatter and
  Landsat MNDWI. Figures are estimates; validate critical claims against full-resolution imagery.
</footer>
</body>
</html>"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"
    path.write_text(html)
    return path
