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
        "site_gerd": "Grand Ethiopian Renaissance Dam (GERD)",
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
        "sig_by": "Prepared by",
        "done": "Analysis ready",
        "spinner": "Analysis processing now...",
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
        "site_gerd": "سد النهضة الإثيوبي الكبير (GERD)",
        "lookback": "الفترة الزمنية (يوم)",
        "run": "تشغيل التحليل الآن",
        "hero_sub": "مراقبة مساحات المياه والتغيرات الإنشائية عبر الأقمار الصناعية للهضبة الاثيوبية وسد النهضة — سينتينل-1، سينتينل-2، لاندسات + تقارير المصادر المفتوحة.",
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
        "sig_by": "إعداد",
        "done": "التحليل جاهز",
        "spinner": "التحليل قيد المعالجة الآن...",
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

import os as _os

_os.environ.setdefault("GDAL_HTTP_TIMEOUT", "20")
_os.environ.setdefault("GDAL_HTTP_RETRY_COUNT", "1")

lang = st.sidebar.selectbox(L10N["en"]["lang_label"], ["English", "العربية"], index=0)
is_ar = lang == "العربية"
t = L10N["ar" if is_ar else "en"]
page_title = "مراقبة النيل" if is_ar else "Nile Monitor"
dir_css = """
<style>
  html, body, [data-testid="stAppViewContainer"] { direction: rtl; }
  [data-testid="stAppViewContainer"] { text-align: right; }
</style>
""" if is_ar else ""

CSS = dir_css + """
<style>
  html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stDataFrame, .stPlotlyChart {
    font-family:'Segoe UI', 'Inter', system-ui, -apple-system, 'Noto Sans Arabic', sans-serif;
    font-size:16px; color:#062a3f; }
  .stApp { background:#dcebf8; }
  section[data-testid="stSidebar"] { background:#071a2c; }
  section[data-testid="stSidebar"] * { color:#dbeafe; }
  .hero { background:linear-gradient(120deg,#050d1c 0%,#0a2c4a 60%,#0d3d5c 100%);
          border-radius:16px; padding:26px 30px; margin-bottom:18px; color:#ffffff;
          box-shadow:0 4px 14px rgba(5,13,28,.45); }
  .hero h1 { margin:0; font-size:28px; letter-spacing:.3px; color:#ffffff;
          font-weight:800; text-shadow:0 2px 6px rgba(0,0,0,.55); }
  .hero p { margin:6px 0 0; color:#eaf6fd; font-size:15px; font-weight:600;
          text-shadow:0 1px 3px rgba(0,0,0,.5); }
  div[data-testid="stMetric"] { background:linear-gradient(135deg,#082c4a 0%,#0e4d6e 100%);
          border:1px solid #062a3f; border-radius:12px; padding:14px 18px;
          box-shadow:0 2px 6px rgba(5,13,28,.25); }
  div[data-testid="stMetricLabel"] { color:#bcdcf0; font-weight:700; font-size:13px;
          white-space:normal; line-height:1.35; }
  div[data-testid="stMetricValue"] { color:#ffffff; font-weight:800; font-size:20px; }
  div[data-testid="stMetricDelta"] { color:#fbbf24; font-weight:700; }
  [data-testid="stDataFrame"] { background:#dff1ff; border-radius:10px; border:1px solid #9ec7ea;
          width:100% !important; }
  .stDataFrame { width:100% !important; }
  .stPlotlyChart, [data-testid="stPlotlyChart"], .js-plotly-plot { direction:ltr; width:100%; }
  .js-plotly-plot .plot-container, .js-plotly-plot .svg-container { width:100% !important; }
  .stPlotlyChart iframe, [data-testid="stPlotlyChart"] iframe { width:100% !important; }
  [data-testid="stDataFrame"] thead th {
    background:#082c4a !important; color:#ffffff !important; font-weight:700; }
  [data-testid="stDataFrame"] tbody td, [data-testid="stDataFrame"] tbody tr {
    background:#dff1ff !important; color:#062a3f; font-weight:500; }
  [data-testid="stExpander"], [data-testid="stExpander"] details { background:#dff1ff; border-radius:10px;
          border:1px solid #9ec7ea; }
  [data-testid="stExpander"] summary { color:#062a3f; font-weight:700; }
  .stTabs [data-baseweb="tab-list"] { gap:6px; }
  .stTabs [data-baseweb="tab"], .stTabs [role="tab"] {
          border-radius:999px; padding:6px 18px; font-weight:700;
          color:#062a3f !important; background:#e8f4ff !important;
          border:1px solid #9ec7ea; }
  .stTabs [data-baseweb="tab"][aria-selected="true"], .stTabs [role="tab"][aria-selected="true"] {
          background:#082c4a !important; color:#ffffff !important; border-color:#082c4a; }
  h1, h2, h3, h4 { color:#062a3f; font-weight:800; }
  .stMarkdown strong { color:#062a3f; }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#1d4e6b; font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

cfg = load_config()
site_ids = [s.id for s in cfg.sites]


def _col_last(series: pd.DataFrame, col: str) -> float | None:
    sub = series[series["collection"] == col].sort_values("date")
    return float(sub["water_area_km2"].iloc[-1]) if not sub.empty else None


def _col_base(series: pd.DataFrame, col: str) -> float | None:
    sub = series[series["collection"] == col].sort_values("date")
    return float(sub["water_area_km2"].iloc[0]) if not sub.empty else None


with st.sidebar:
    st.markdown(f"## 🛰 {page_title}")
    site_id = st.selectbox(t["site"], site_ids, format_func=lambda i: t.get(f"site_{i}", cfg.site(i).name))
    site = cfg.site(site_id)
    out_dir = cfg.output_dir / site_id

    # ---- Monitoring reference panel (status board) ----
    st.divider()
    st.markdown(f"### 📋 {t['watch_header']}")
    panel_csv = out_dir / "water_area_timeseries.csv"
    if panel_csv.exists():
        panel_series = pd.read_csv(panel_csv)
        panel_series["date"] = pd.to_datetime(panel_series["date"])
        panel_alerts: list = []
        panel_alerts_path = out_dir / "alerts.json"
        if panel_alerts_path.exists():
            try:
                panel_alerts = json.loads(panel_alerts_path.read_text()).get("alerts", [])
            except (json.JSONDecodeError, OSError):
                panel_alerts = []
        p_s2_last = _col_last(panel_series, "sentinel-2-l2a")
        p_s2_base = _col_base(panel_series, "sentinel-2-l2a")
        p_s1_last = _col_last(panel_series, "sentinel-1-grd")

        st.markdown(f"**{t['watch_construction']}**")
        if panel_alerts:
            high = sum(1 for a in panel_alerts if a["severity"] == "HIGH")
            med = sum(1 for a in panel_alerts if a["severity"] == "MEDIUM")
            st.markdown(
                f":red[**{len(panel_alerts)} {t['watch_active']}**]"
                f"<br><small>HIGH: {high} · MEDIUM: {med}</small>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f":green[**{t['watch_none']}**]")

        st.markdown(f"**{t['watch_reservoir']}**")
        if p_s2_last is not None and p_s2_base is not None:
            delta = p_s2_last - p_s2_base
            arrow = ":red[▲]" if delta > 0 else (":green[▼]" if delta < 0 else ":gray[—]")
            st.markdown(
                f"**{p_s2_last:,.1f} km²** {arrow} {delta:+,.1f} km²"
                f"<br><small>{t['watch_storage_proxy']}</small>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("—")

        st.markdown(f"**{t['watch_flow']}**")
        if p_s2_base and p_s2_last:
            rel = (p_s2_last - p_s2_base) / abs(p_s2_base) * 100
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
        if p_s1_last is not None:
            st.markdown(f"<small>{t['watch_s1_latest']}: **{p_s1_last:,.1f} km²**</small>", unsafe_allow_html=True)

        panel_osint: list = []
        panel_osint_path = out_dir / "osint_news.json"
        if panel_osint_path.exists():
            try:
                panel_osint = json.loads(panel_osint_path.read_text()).get("items", [])[:3]
            except (json.JSONDecodeError, OSError):
                panel_osint = []
        if panel_osint:
            st.markdown(f"**{t['watch_osint']}**")
            for it in panel_osint:
                st.markdown(f"<small>• {it.get('title', '')[:80]}</small>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#e2f7e6;border:1px solid #7cc98a;border-radius:8px;"
                    f"padding:10px 12px;color:#0b5e2e;font-weight:700;font-size:15px;'>"
                    f"⚠ {t['no_analysis']}</div>",
                    unsafe_allow_html=True)

    # ---- Controls ----
    st.divider()
    st.markdown(f"**{site.river} — {site.country}**")
    st.markdown(f"{site.lat:.4f} N, {site.lon:.4f} E")
    days = st.slider(t["lookback"], 30, 3650, 365)
    run_now = st.button(t["run"], type="primary", use_container_width=True)

date_to = dt.date.today()
date_from = date_to - dt.timedelta(days=days)

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex;align-items:center;gap:18px;">
        <img src="/app/static/GERD.jpg" alt="GERD"
             style="width:110px;height:auto;border-radius:12px;border:2px solid #ffffff;
                    box-shadow:0 3px 10px rgba(0,0,0,.5);"/>
        <div>
          <h1 style="margin:0;">🛰 {t.get(f"site_{site_id}", site.name)}</h1>
          <p style="margin:6px 0 0;">{t['hero_sub']} {t['window']}: {date_from} → {date_to}.</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_now:
    with st.spinner(t["spinner"]):
        out_dir = run_pipeline(cfg, site_id, date_from, date_to)
    st.success(t["done"])

csv_path = out_dir / "water_area_timeseries.csv"
if not csv_path.exists():
    st.markdown(f"<div style='background:#e2f7e6;border:1px solid #7cc98a;border-radius:10px;"
                f"padding:14px 18px;color:#0b5e2e;font-weight:700;font-size:17px;'>"
                f"⚠ {t['no_analysis']}</div>",
                unsafe_allow_html=True)
    st.stop()

series = pd.read_csv(csv_path)
series["date"] = pd.to_datetime(series["date"])

alerts = []
alerts_path = out_dir / "alerts.json"
if alerts_path.exists():
    alerts = json.loads(alerts_path.read_text()).get("alerts", [])

s2_last, s2_base = _col_last(series, "sentinel-2-l2a"), _col_base(series, "sentinel-2-l2a")
s1_last, s1_base = _col_last(series, "sentinel-1-grd"), _col_base(series, "sentinel-1-grd")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    t["metric_s2"],
    f"{s2_last:,.1f} km²" if s2_last is not None else "—",
    f"{s2_last - s2_base:+,.1f} {t['vs_baseline']}" if (s2_last is not None and s2_base is not None) else None,
)
c2.metric(
    t["metric_s1"],
    f"{s1_last:,.1f} km²" if s1_last is not None else "—",
    f"{s1_last - s1_base:+,.1f} {t['vs_baseline']}" if (s1_last is not None and s1_base is not None) else None,
)
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
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.18, x=0),
                          yaxis_title=f"km²",
                          paper_bgcolor="#dcebf8", plot_bgcolor="#dcebf8",
                          font=dict(family="Segoe UI, sans-serif"),
                          title=dict(text=t["chart_sub"], x=0.5,
                                     font=dict(size=17, color="#0b7a3b", family="Segoe UI, sans-serif")))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(t["sum_header"])
        latest = series.sort_values("date").groupby("collection").tail(1)
        baseline = series.sort_values("date").groupby("collection").head(1)
        merged = baseline.merge(latest, on="collection", suffixes=("_base", "_latest"))
        merged["collection"] = merged["collection"].map(lambda c: COLLECTION_NAMES.get(c, c))
        merged["Δ km²"] = (merged["water_area_km2_latest"] - merged["water_area_km2_base"]).round(2)
        st.dataframe(
            merged.rename(columns={"collection": t["collection"]})[[
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

        components.html(map_html.read_text(), height=620, scrolling=True)
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
        fig.update_layout(height=420, xaxis_title=t["lon"], yaxis_title=t["lat"],
                          paper_bgcolor="#dcebf8", plot_bgcolor="#dcebf8", font=dict(family="Segoe UI, sans-serif"))
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

st.markdown(
    f"""
    <div style="margin-top:26px;padding-top:12px;border-top:2px solid #9ec7ea;
                text-align:center;color:#0b5e2e;font-weight:700;font-size:14px;">
      ✍ {t['sig_by']} Karim Ali — 90.karim@gmail.com
    </div>
    """,
    unsafe_allow_html=True,
)
