"""nilemonitor dashboard — Streamlit app (English / العربية).

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nilemonitor.config import load_config  # noqa: E402
from nilemonitor.pipeline import run_pipeline  # noqa: E402

COLLECTION_NAMES = {
    "sentinel-2-l2a": "Sentinel-2",
    "sentinel-1-grd": "Sentinel-1",
    "landsat-c2-l2": "Landsat",
}

L10N = {
    "en": {
        "lang_label": "Language",
        "site": "Site",
        "lookback": "Lookback (days)",
        "run": "Run analysis now",
        "hero_sub": "Satellite water-extent and construction-change monitoring — Sentinel-1 SAR, "
                    "Sentinel-2 NDWI, Landsat MNDWI + open-source reporting.",
        "window": "Window",
        "metric_s2": "Sentinel-2 water (latest)",
        "metric_s1": "Sentinel-1 water (latest)",
        "metric_alerts": "Construction warnings",
        "metric_scenes": "Scenes analysed",
        "vs_baseline": "vs baseline",
        "tab_overview": "Overview",
        "tab_imagery": "Satellite imagery",
        "tab_alerts": "Warnings & construction",
        "tab_validation": "Validation",
        "tab_report": "Report",
        "chart_title": "Water extent over time",
        "chart_sub": "Water area within the analysis radius (km²)",
        "sum_header": "Latest summary",
        "col_base": "baseline",
        "col_latest": "latest",
        "map_header": "Map",
        "full_table": "Full observation table",
        "img_header": "Satellite imagery",
        "img_caption": "Scene previews over the analysis area — Sentinel-2 true colour, Sentinel-1 radar backscatter.",
        "no_previews": "No scene previews in this run — run the analysis to generate them.",
        "alerts_note": "unverified detection(s) — SAR pair. Every alert is a candidate; verify with high-resolution imagery before acting.",
        "alerts_header": "Alert locations (unverified candidates)",
        "alerts_caption": "Warnings over SAR backscatter",
        "no_alerts": "No construction detections in this SAR pair — the detector is watching every 12-day repeat.",
        "constr_header": "Construction / activity index",
        "constr_caption": "Construction activity index",
        "dl_html": "Download HTML report",
        "no_report": "No report for this run yet.",
        "no_analysis": "No analysis found yet — press **Run analysis now** in the sidebar.",
        "done": "Analysis complete.",
        "spinner": "Fetching scenes, computing water masks, gathering news…",
        "no_map": "No interactive map for this run yet.",
        "severity": "Severity",
        "kind": "Type",
        "detected": "Detected",
        "area": "Area (km²)",
        "gain": "VV gain (dB)",
        "lat": "Lat",
        "lon": "Lon",
        "description": "Description",
        "collection": "Collection",
        "date": "Date",
        "water_area": "Water area (km²)",
        "ndwi": "NDWI mean",
        "cloud": "Cloud (%)",
        "watch_header": "Monitoring reference",
        "watch_construction": "New construction",
        "watch_none": "none detected — watching every 12-day SAR repeat",
        "watch_active": "unverified detection(s) — verify before acting",
        "watch_reservoir": "GERD reservoir (retained water)",
        "watch_storage_proxy": "area-based storage proxy",
        "watch_filling": "reservoir filling → less flow downstream",
        "watch_releasing": "reservoir releasing → more flow downstream",
        "watch_stable": "storage stable → expected flow unchanged",
        "watch_flow": "Flow to Egypt & Sudan",
        "watch_flow_note": "derived from the reservoir storage trend (no river gauges)",
        "watch_course": "Nile course & tributaries",
        "watch_course_note": "monitored via S2/S1 water series and change maps",
        "watch_osint": "Open-source reporting",
        "watch_s2_latest": "S2 water (latest)",
        "watch_s1_latest": "S1 water (latest)",
        "val_header": "Independent validation",
        "val_intro": "Water masks vs the JRC Global Surface Water v1.4 (2021) maximum-extent reference (Landsat 1984–2021, Pekel et al., Nature 2016) on a shared 30 m grid. Water is the positive class.",
        "val_note": "Pixels detected by the pipeline but absent from the reference are expected where water appeared after 2021 (ongoing reservoir fill).",
        "val_metrics": "Accuracy metrics vs GSW reference",
        "val_precision": "Precision",
        "val_storage": "Reservoir storage proxy (Copernicus DEM GLO-30)",
        "val_storage_note": "volume assumes the lake is at the GERD full-supply elevation — upper bound",
        "val_vol": "Volume (km³)",
        "val_area_ref": "Water area (km²)",
        "val_depth": "Mean depth (m)",
        "val_crest": "Crest level",
        "val_no_data": "No validation data for this run yet.",
    },
    "ar": {
        "lang_label": "اللغة",
        "site": "الموقع",
        "lookback": "الفترة الزمنية (يوم)",
        "run": "تشغيل التحليل الآن",
        "hero_sub": "مراقبة مساحات المياه والتغيرات الإنشائية عبر الأقمار الصناعية — سينتينل-1، سينتينل-2، لاندسات + تقارير المصادر المفتوحة.",
        "window": "الفترة",
        "metric_s2": "مياه سينتينل-2 (الأحدث)",
        "metric_s1": "مياه سينتينل-1 (الأحدث)",
        "metric_alerts": "تحذيرات الإنشاءات",
        "metric_scenes": "المشاهد المحللة",
        "vs_baseline": "مقارنة بالأساس",
        "tab_overview": "نظرة عامة",
        "tab_imagery": "صور الأقمار الصناعية",
        "tab_alerts": "التحذيرات والإنشاءات",
        "tab_validation": "التحقق العلمي",
        "tab_report": "التقرير",
        "chart_title": "تطور مساحة المياه عبر الزمن",
        "chart_sub": "مساحة المياه ضمن نطاق التحليل (كم²)",
        "sum_header": "الملخص الأحدث",
        "col_base": "الأساس",
        "col_latest": "الأحدث",
        "map_header": "الخريطة",
        "full_table": "جدول المشاهدات الكامل",
        "img_header": "صور الأقمار الصناعية",
        "img_caption": "مشاهد من منطقة التحليل — سينتينل-2 بالألوان الحقيقية، سينتينل-1 بالرادار.",
        "no_previews": "لا توجد صور لهذه الجولة — شغّل التحليل لإنشائها.",
        "alerts_note": "كشف غير مؤكد — زوج رادار سينتينل-1. كل تحذير مرشح للفحص؛ تحقق من الصور عالية الدقة قبل اتخاذ أي إجراء.",
        "alerts_header": "مواقع التحذيرات (مرشحة للفحص)",
        "alerts_caption": "التحذيرات فوق خلفية الرادار",
        "no_alerts": "لا توجد كشوفات إنشائية في هذا الزوج الراداري — المراقب يعمل كل 12 يوماً.",
        "constr_header": "مؤشر الإنشاءات / النشاط",
        "constr_caption": "مؤشر نشاط الإنشاءات",
        "dl_html": "تحميل التقرير (HTML)",
        "no_report": "لا يوجد تقرير لهذه الجولة بعد.",
        "no_analysis": "لا يوجد تحليل بعد — اضغط **تشغيل التحليل الآن** في القائمة الجانبية.",
        "done": "اكتمل التحليل.",
        "spinner": "جاري جلب المشاهد وحساب خرائط المياه وجمع الأخبار…",
        "no_map": "لا توجد خريطة تفاعلية لهذه الجولة بعد.",
        "severity": "الخطورة",
        "kind": "النوع",
        "detected": "تاريخ الكشف",
        "area": "المساحة (كم²)",
        "gain": "كسب VV (dB)",
        "lat": "خط العرض",
        "lon": "خط الطول",
        "description": "الوصف",
        "collection": "المجموعة",
        "date": "التاريخ",
        "water_area": "مساحة المياه (كم²)",
        "ndwi": "متوسط NDWI",
        "cloud": "الغيوم (%)",
        "watch_header": "لوحة المراقبة المرجعية",
        "watch_construction": "إنشاءات جديدة",
        "watch_none": "لا شيء مكتشف — المراقبة كل 12 يوماً عبر الرادار",
        "watch_active": "كشف غير مؤكد — تحقق قبل اتخاذ الإجراء",
        "watch_reservoir": "خزان سد النهضة (المياه المحتجزة)",
        "watch_storage_proxy": "مؤشر تخزين يعتمد على المساحة",
        "watch_filling": "الخزان يمتلئ ← تدفق أقل لأسفل النهر",
        "watch_releasing": "الخزان يفرّغ ← تدفق أكبر لأسفل النهر",
        "watch_stable": "التخزين مستقر ← التدفق المتوقع دون تغيير",
        "watch_flow": "التدفق إلى مصر والسودان",
        "watch_flow_note": "مشتق من اتجاه تخزين الخزان (لا توجد مقاييس نهرية)",
        "watch_course": "مجرى النيل وروافده",
        "watch_course_note": "تُراقب عبر سلاسل مياه سينتينل-1/2 وخرائط التغيير",
        "watch_osint": "تقارير المصادر المفتوحة",
        "watch_s2_latest": "مياه سينتينل-2 (الأحدث)",
        "watch_s1_latest": "مياه سينتينل-1 (الأحدث)",
        "val_header": "التحقق العلمي المستقل",
        "val_intro": "مقارنة خرائط المياه مع مرجع جي آر سي للمياه السطحية العالمية (الإصدار 1.4 لعام 2021، أقصى مدى، لاندسات 1984–2021 — بيكل وآخرون، نيتشر 2016) على شبكة 30 مترًا موحّدة. المياه هي الفئة الإيجابية.",
        "val_note": "البكسلات التي تكتشفها الخوارزمية ولا توجد في المرجع متوقّعة حيث ظهرت المياه بعد 2021 (استمرار امتلاء الخزان).",
        "val_metrics": "مقاييس الدقة مقابل مرجع جي آر سي",
        "val_precision": "الدقة الإيجابية",
        "val_storage": "تقدير سعة التخزين (كوبرنيكوس ديم)",
        "val_storage_note": "يفترض الحجم أن منسوب البحيرة عند منسوب الامتلاء الكامل لسد النهضة — حد أعلى",
        "val_vol": "الحجم (كم³)",
        "val_area_ref": "مساحة المياه (كم²)",
        "val_depth": "متوسط العمق (م)",
        "val_crest": "منسوب القمة",
        "val_no_data": "لا توجد بيانات تحقق لهذه الجولة بعد.",
    },
}

st.set_page_config(page_title="Nile Monitor", layout="wide", page_icon="🛰")

lang = st.sidebar.selectbox(L10N["en"]["lang_label"], ["English", "العربية"], index=0)
is_ar = lang == "العربية"
t = L10N["ar" if is_ar else "en"]
dir_css = """
<style>
  html, body, [data-testid="stAppViewContainer"] { direction: rtl; }
  [data-testid="stAppViewContainer"] { text-align: right; }
</style>
""" if is_ar else ""

CSS = dir_css + """
<style>
  .stApp { background:#f1f5f9; }
  section[data-testid="stSidebar"] { background:#0f172a; }
  section[data-testid="stSidebar"] * { color:#e2e8f0; }
  .hero { background:linear-gradient(120deg,#0b2a4a 0%,#0e7490 65%,#14b8a6 100%);
          border-radius:16px; padding:26px 30px; margin-bottom:18px; color:#fff; }
  .hero h1 { margin:0; font-size:28px; letter-spacing:.3px; }
  .hero p { margin:4px 0 0; color:#cbe8f2; }
  div[data-testid="stMetric"] { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
          padding:12px 16px; box-shadow:0 1px 3px rgba(15,23,42,.06); }
  div[data-testid="stMetricLabel"] { color:#64748b; }
  .stTabs [data-baseweb="tab-list"] { gap:6px; }
  .stTabs [data-baseweb="tab"] { border-radius:999px; padding:6px 18px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

cfg = load_config()
site_ids = [s.id for s in cfg.sites]

with st.sidebar:
    st.markdown("## 🛰 Nile Monitor")
    site_id = st.selectbox(t["site"], site_ids, format_func=lambda i: cfg.site(i).name)
    site = cfg.site(site_id)
    st.markdown(f"**{site.river} — {site.country}**")
    st.markdown(f"{site.lat:.4f} N, {site.lon:.4f} E")
    days = st.slider(t["lookback"], 30, 3650, 365)
    run_now = st.button(t["run"], type="primary", use_container_width=True)

date_to = dt.date.today()
date_from = date_to - dt.timedelta(days=days)

st.markdown(
    f"""
    <div class="hero">
      <h1>🛰 {site.name}</h1>
      <p>{t['hero_sub']} {t['window']}: {date_from} → {date_to}.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

out_dir = cfg.output_dir / site_id

if run_now:
    with st.spinner(t["spinner"]):
        out_dir = run_pipeline(cfg, site_id, date_from, date_to)
    st.success(t["done"])

csv_path = out_dir / "water_area_timeseries.csv"
if not csv_path.exists():
    st.info(t["no_analysis"])
    st.stop()

series = pd.read_csv(csv_path)
series["date"] = pd.to_datetime(series["date"])

alerts = []
alerts_path = out_dir / "alerts.json"
if alerts_path.exists():
    alerts = json.loads(alerts_path.read_text()).get("alerts", [])


def latest_of(col: str) -> float | None:
    sub = series[series["collection"] == col].sort_values("date")
    return float(sub["water_area_km2"].iloc[-1]) if not sub.empty else None


def baseline_of(col: str) -> float | None:
    sub = series[series["collection"] == col].sort_values("date")
    return float(sub["water_area_km2"].iloc[0]) if not sub.empty else None


s2_last, s2_base = latest_of("sentinel-2-l2a"), baseline_of("sentinel-2-l2a")
s1_last, s1_base = latest_of("sentinel-1-grd"), baseline_of("sentinel-1-grd")

# ---- Monitoring reference panel (status board, always visible) ----
with st.sidebar:
    st.divider()
    st.markdown(f"### 📋 {t['watch_header']}")

    st.markdown(f"**{t['watch_construction']}**")
    if alerts:
        high = sum(1 for a in alerts if a["severity"] == "HIGH")
        med = sum(1 for a in alerts if a["severity"] == "MEDIUM")
        st.markdown(
            f":red[**{len(alerts)} {t['watch_active']}**]"
            f"<br><small>HIGH: {high} · MEDIUM: {med}</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f":green[**{t['watch_none']}**]")

    st.markdown(f"**{t['watch_reservoir']}**")
    if s2_last is not None and s2_base is not None:
        delta = s2_last - s2_base
        arrow = ":red[▲]" if delta > 0 else (":green[▼]" if delta < 0 else ":gray[—]")
        st.markdown(
            f"**{s2_last:,.1f} km²** {arrow} {delta:+,.1f} km²"
            f"<br><small>{t['watch_storage_proxy']}</small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("—")

    st.markdown(f"**{t['watch_flow']}**")
    if s2_base and s2_last:
        rel = (s2_last - s2_base) / abs(s2_base) * 100
        if rel > 0.5:
            flow = f":blue[▲ {t['watch_filling']}]"
        elif rel < -0.5:
            flow = f":blue[▼ {t['watch_releasing']}]"
        else:
            flow = f":gray[— {t['watch_stable']}]"
        st.markdown(f"{flow}<br><small>{t['watch_flow_note']}</small>", unsafe_allow_html=True)
    else:
        st.markdown("—")

    st.markdown(f"**{t['watch_course']}**")
    st.markdown(f":violet[🛰 {t['watch_course_note']}]")
    if s1_last is not None:
        st.markdown(f"<small>{t['watch_s1_latest']}: **{s1_last:,.1f} km²**</small>", unsafe_allow_html=True)

    osint_items: list = []
    _np = out_dir / "osint_news.json"
    if _np.exists():
        try:
            osint_items = json.loads(_np.read_text()).get("items", [])[:3]
        except (json.JSONDecodeError, OSError):
            osint_items = []
    if osint_items:
        st.markdown(f"**{t['watch_osint']}**")
        for it in osint_items:
            st.markdown(f"<small>• {it.get('title', '')[:80]}</small>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric(t["metric_s2"], f"{s2_last:,.1f} km²", f"{s2_last - s2_base:+,.1f} {t['vs_baseline']}" if s2_base else None)
c2.metric(t["metric_s1"], f"{s1_last:,.1f} km²", f"{s1_last - s1_base:+,.1f} {t['vs_baseline']}" if s1_base else None)
c3.metric(t["metric_alerts"], f"{len(alerts)}", None)
c4.metric(t["metric_scenes"], f"{len(series)}", None)

tab_overview, tab_imagery, tab_alerts, tab_validation, tab_report = st.tabs(
    [t["tab_overview"], t["tab_imagery"], t["tab_alerts"], t["tab_validation"], t["tab_report"]]
)

with tab_overview:
    left, right = st.columns([3, 2])
    with left:
        st.subheader(t["chart_title"])
        fig = px.line(
            series, x="date", y="water_area_km2",
            color=pd.Series(series["collection"].map(lambda c: COLLECTION_NAMES.get(c, c))),
            markers=True, title=t["chart_sub"],
            color_discrete_map={"Sentinel-2": "#0e7490", "Sentinel-1": "#f59e0b", "Landsat": "#16a34a"},
            template="plotly_white",
        )
        fig.update_layout(height=420, legend_title=t["collection"], yaxis_title=f"km²")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(t["sum_header"])
        latest = series.sort_values("date").groupby("collection").tail(1)
        baseline = series.sort_values("date").groupby("collection").head(1)
        merged = baseline.merge(latest, on="collection", suffixes=("_base", "_latest"))
        merged["collection"] = merged["collection"].map(lambda c: COLLECTION_NAMES.get(c, c))
        merged["Δ km²"] = (merged["water_area_km2_latest"] - merged["water_area_km2_base"]).round(2)
        st.dataframe(
            merged[[
                t["collection"], f"water_area_km2_{'base'}", f"water_area_km2_{'latest'}", "Δ km²"
            ]].rename(columns={
                f"water_area_km2_base": t["col_base"] + " km²",
                f"water_area_km2_latest": t["col_latest"] + " km²",
            }),
            hide_index=True, use_container_width=True,
        )
        st.subheader(t["map_header"])
        map_html = out_dir / "interactive_map.html"
        if map_html.exists():
            import streamlit.components.v1 as components

            components.html(map_html.read_text(), height=520, scrolling=True)
        else:
            st.warning(t["no_map"])
        with st.expander(t["full_table"]):
            tab = series.copy()
            tab["collection"] = tab["collection"].map(lambda c: COLLECTION_NAMES.get(c, c))
            st.dataframe(
                tab.rename(columns={
                    "date": t["date"], "water_area_km2": t["water_area"],
                    "ndwi_mean": t["ndwi"], "cloud_cover": t["cloud"], "collection": t["collection"],
                }),
                hide_index=True, use_container_width=True,
            )

with tab_imagery:
    st.subheader(t["img_header"])
    st.caption(t["img_caption"])
    scenes_dir = out_dir / "scenes"
    items = sorted(scenes_dir.glob("*.png")) if scenes_dir.exists() else []
    if items:
        cols = st.columns(3)
        for i, img in enumerate(items):
            with cols[i % 3]:
                st.image(str(img), use_container_width=True, caption=img.stem[:34])
    else:
        st.info(t["no_previews"])

with tab_alerts:
    if alerts:
        st.success(f"**{len(alerts)} {t['alerts_note']}**")
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(
            df_alerts[[
                "severity", "kind", "date_detected", "area_km2", "vv_gain_db", "lat", "lon", "description"
            ]].rename(columns={
                "severity": t["severity"], "kind": t["kind"], "date_detected": t["detected"],
                "area_km2": t["area"], "vv_gain_db": t["gain"],
                "lat": t["lat"], "lon": t["lon"], "description": t["description"],
            }),
            hide_index=True, use_container_width=True,
        )
        fig = px.scatter(
            df_alerts, x="lon", y="lat", color="severity", size="area_km2",
            hover_data=["kind", "vv_gain_db", "date_detected"],
            title=t["alerts_header"],
            color_discrete_map={"HIGH": "#e11d48", "MEDIUM": "#f59e0b", "LOW": "#22c55e"},
            template="plotly_white",
        )
        fig.update_layout(height=420, xaxis_title=t["lon"], yaxis_title=t["lat"])
        st.plotly_chart(fig, use_container_width=True)
        if (out_dir / "alerts_map.png").exists():
            st.image(str(out_dir / "alerts_map.png"), caption=t["alerts_caption"], use_container_width=True)
    else:
        st.info(t["no_alerts"])

    st.subheader(t["constr_header"])
    if (out_dir / "construction_map.png").exists():
        st.image(str(out_dir / "construction_map.png"), caption=t["constr_caption"], use_container_width=True)

with tab_validation:
    vp = out_dir / "validation.json"
    if vp.exists():
        try:
            val = json.loads(vp.read_text())
            colls = val.get("collections", {})
            if colls:
                st.markdown(f"### {t['val_header']}")
                st.caption(f"{t['val_intro']} {t['val_note']}")
                rows = []
                for coll, entry in colls.items():
                    m = entry.get("gsw_metrics") or {}
                    rows.append({
                        t["collection"]: coll,
                        "OA": m.get("overall_accuracy", float("nan")),
                        t["val_precision"]: m.get("precision_user", float("nan")),
                        "Recall": m.get("recall_producer", float("nan")),
                        "F1": m.get("f1", float("nan")),
                        "IoU": m.get("iou", float("nan")),
                        "Kappa": m.get("kappa", float("nan")),
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                for coll, entry in colls.items():
                    s = entry.get("storage")
                    if s and "note" in s:
                        v1, v2, v3, v4 = st.columns(4)
                        v1.metric(t["val_vol"], f"{s['volume_km3']:.2f}")
                        v2.metric(t["val_area_ref"], f"{s['area_km2']:,.1f}")
                        v3.metric(t["val_depth"], f"{s['mean_depth_m']:.0f} m")
                        v4.metric(t["val_crest"], f"{s['crest_level_m']:.0f} m a.s.l.")
                        st.caption(f"{coll}: {t['val_storage_note']}")
                        break
                if (out_dir / "validation_map.png").exists():
                    st.image(str(out_dir / "validation_map.png"),
                             caption="GSW reference (blue) · detected (green) · not in reference (red)",
                             use_container_width=True)
            else:
                st.info(t["val_no_data"])
        except (ValueError, KeyError, OSError):
            st.info(t["val_no_data"])
    else:
        st.info(t["val_no_data"])

with tab_report:
    html_path = out_dir / "report.html"
    md_path = out_dir / "report.md"
    if html_path.exists():
        st.download_button(t["dl_html"], html_path.read_text(), file_name="report.html", mime="text/html")
        import streamlit.components.v1 as components

        components.html(html_path.read_text(), height=1400, scrolling=True)
    elif md_path.exists():
        with st.expander(t["tab_report"]):
            st.markdown(md_path.read_text())
    else:
        st.info(t["no_report"])
