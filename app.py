"""
app.py  –  Main Streamlit Dashboard: HIV/AIDS Global Analytics
Course  : Exploratory Data Analysis  |  Instructor: Ali Hassan Sherazi
Dataset : aidsinfo.unaids.org.csv   |  UNAIDS HIV/AIDS Global Estimates 2025
"""

import streamlit as st
import pandas as pd
import numpy as np

from filters import load_data, apply_filters
from charts  import (pie_chart, histogram, line_chart, bar_chart,
                     scatter_plot, box_plot, heatmap, area_chart,
                     count_plot, violin_plot, pair_plot)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔴 HIV/AIDS Analytics Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    background-color: #0D1117;
    color: #E6EDF3;
    font-family: 'Lato', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

/* Header banner */
.banner {
    background: linear-gradient(135deg, #1A0A0A 0%, #6B1414 60%, #C0392B 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 20px;
    border: 1px solid #C0392B33;
}
.banner h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    color: #E6EDF3;
    margin: 0;
    letter-spacing: 1px;
}
.banner p { color: #E08080; margin: 6px 0 0; font-size: 0.95rem; }

/* KPI cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi {
    background: #161B22;
    border: 1px solid #30363D;
    border-top: 3px solid #C0392B;
    border-radius: 10px;
    padding: 18px 22px;
    flex: 1;
    min-width: 140px;
    transition: transform .2s;
}
.kpi:hover { transform: translateY(-3px); border-top-color: #E74C3C; }
.kpi .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: #C0392B;
    display: block;
}
.kpi .lbl { font-size: 0.75rem; color: #8B949E; text-transform: uppercase; letter-spacing: 1px; }

/* Section headers */
.sec-head {
    font-family: 'Playfair Display', serif;
    color: #C0392B;
    font-size: 1.2rem;
    border-left: 4px solid #C0392B;
    padding-left: 12px;
    margin: 28px 0 14px;
}

/* Chart containers */
.chart-wrap {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    transition: box-shadow .2s;
}
.chart-wrap:hover { box-shadow: 0 0 0 1px #C0392B55; }

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #6B1414, #C0392B);
    color: #E6EDF3;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    transition: opacity .2s;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.85; }

/* Streamlit widget labels */
label { color: #C0392B !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    """
    WHY @st.cache_data: Caches the loaded DataFrame so Streamlit does not
    re-read the 2.5M-row CSV on every filter interaction. Makes the dashboard
    fast and responsive.
    """
    return load_data()

df = get_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔴 Filters")
    st.markdown("---")

    # Reset button
    # WHY session_state: Streamlit re-runs the script on every interaction.
    # session_state["reset"] is the correct pattern for resetting widgets.
    if st.button("🔄 Reset All Filters"):
        st.session_state["reset"] = True
        st.rerun()

    reset = st.session_state.get("reset", False)

    # 1. Time Period Range Slider
    yr_min = int(df["Time_Period"].min())
    yr_max = int(df["Time_Period"].max())
    time_range = st.slider(
        "📅 Time Period Range",
        yr_min, yr_max,
        (yr_min, yr_max) if reset else (yr_min, yr_max),
    )

    # 2. Area (Country/Region) Multi-Select
    all_areas = sorted(df["Area"].unique())
    areas = st.multiselect(
        "🌍 Area (Country / Region)",
        all_areas,
        default=[] if reset else [],
    )

    # 3. Indicator Multi-Select
    all_indicators = sorted(df["Indicator"].unique())
    indicators = st.multiselect(
        "📊 HIV/AIDS Indicator",
        all_indicators,
        default=[] if reset else [],
    )

    # 4. Unit Multi-Select
    all_units = sorted(df["Unit"].unique())
    units = st.multiselect(
        "📐 Unit Type",
        all_units,
        default=[] if reset else [],
    )

    # 5. Source Multi-Select
    all_sources = sorted(df["Source"].unique())
    sources = st.multiselect(
        "🏥 Data Source",
        all_sources,
        default=[] if reset else [],
    )

    # 6. Area Level Multi-Select
    level_map = {1: "Global", 2: "Regional", 3: "National"}
    all_levels = sorted(df["Area_Level"].unique())
    level_labels = {v: k for k, v in level_map.items()}
    area_level_display = [level_map.get(l, str(l)) for l in all_levels]
    sel_level_labels = st.multiselect(
        "🗺️ Area Level",
        area_level_display,
        default=[] if reset else [],
    )
    area_levels = [level_labels.get(l, l) for l in sel_level_labels]

    # 7. Subgroup Multi-Select (top 30 for usability)
    top_subgroups = df["Subgroup"].value_counts().head(30).index.tolist()
    subgroups = st.multiselect(
        "👥 Subgroup",
        top_subgroups,
        default=[] if reset else [],
    )

    # 8. Data Value Range Slider
    val_min = float(df["Data_Value"].quantile(0.01))
    val_max = float(df["Data_Value"].quantile(0.99))
    value_range = st.slider(
        "📈 Data Value Range",
        val_min, val_max,
        (val_min, val_max),
        format="%.2f",
    )

    st.session_state["reset"] = False
    st.markdown("---")
    st.caption("All charts update simultaneously when filters change.")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
fdf = apply_filters(
    df, areas, indicators, units, sources,
    area_levels, subgroups, time_range, value_range,
)

# ── BANNER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div style="text-align:center; margin-bottom:16px;">
    <div style="font-size:0.68rem; letter-spacing:4px; text-transform:uppercase; color:#E6EDF3; opacity:0.75;">Created By</div>
    <div style="font-family:'Playfair Display',serif; font-size:1.9rem; color:#E6EDF3; font-weight:700; letter-spacing:2px;">Alisha Saif</div>
    <div style="width:60px; height:2px; background:linear-gradient(90deg,transparent,#E6EDF3,transparent); margin:6px auto 0;"></div>
  </div>
  <h1 style="text-align:center;">🔴 HIV/AIDS Global Analytics Dashboard</h1>
  <p style="text-align:center;">2,509,456 records across 228 areas · 50 indicators · 1990–2024
  &nbsp;·&nbsp; Exploratory Data Analysis &nbsp;·&nbsp; Instructor: Ali Hassan Sherazi</p>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────────────────────────
total        = len(fdf)
total_areas  = fdf["Area"].nunique()      if total else 0
total_indic  = fdf["Indicator"].nunique() if total else 0
avg_val      = fdf["Data_Value"].mean()   if total else 0
max_val      = fdf["Data_Value"].max()    if total else 0
total_src    = fdf["Source"].nunique()    if total else 0
min_val      = fdf["Data_Value"].min()    if total else 0

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><span class="val">{total:,}</span><span class="lbl">Total Records</span></div>
  <div class="kpi"><span class="val">{total_areas:,}</span><span class="lbl">Total Areas</span></div>
  <div class="kpi"><span class="val">{total_indic}</span><span class="lbl">Total Indicators</span></div>
  <div class="kpi"><span class="val">{avg_val:,.2f}</span><span class="lbl">Avg Data Value</span></div>
  <div class="kpi"><span class="val">{max_val:,.0f}</span><span class="lbl">Max Data Value</span></div>
  <div class="kpi"><span class="val">{min_val:,.2f}</span><span class="lbl">Min Data Value</span></div>
  <div class="kpi"><span class="val">{total_src}</span><span class="lbl">Data Sources</span></div>
</div>
""", unsafe_allow_html=True)

if total == 0:
    st.warning("No records match the current filters. Please adjust the sidebar filters.")
    st.stop()

# ── SECTION 1: Distribution Charts ───────────────────────────────────────────
st.markdown('<div class="sec-head">📊 Distribution Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(histogram(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(pie_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 2: Categorical Analysis ──────────────────────────────────────────
st.markdown('<div class="sec-head">🌍 Regional & Source Analysis</div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(bar_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(count_plot(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 3: Time Series ────────────────────────────────────────────────────
st.markdown('<div class="sec-head">📈 Trends Over Time</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(line_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c6:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(area_chart(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 4: Relationship & Spread ─────────────────────────────────────────
st.markdown('<div class="sec-head">🔬 Relationship & Spread Analysis</div>', unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(scatter_plot(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c8:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(box_plot(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 5: Advanced ───────────────────────────────────────────────────────
st.markdown('<div class="sec-head">🎻 Advanced Visualisations</div>', unsafe_allow_html=True)
c9, c10 = st.columns(2)
with c9:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(violin_plot(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c10:
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(heatmap(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 6: BONUS Pair Plot (full width) ───────────────────────────────────
st.markdown('<div class="sec-head">✨ Bonus: Pair Plot</div>', unsafe_allow_html=True)
with st.expander("Show Pair Plot (may take a moment to render)", expanded=False):
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(pair_plot(fdf), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RAW DATA PREVIEW ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">🗃️ Filtered Data Preview</div>', unsafe_allow_html=True)
with st.expander("Show filtered data table"):
    st.dataframe(fdf.head(200), use_container_width=True)

st.caption("Dashboard by Alisha Saif  ·  Course: Exploratory Data Analysis  ·  Instructor: Ali Hassan Sherazi")
