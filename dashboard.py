"""
dashboard.py — Real-time Violations Dashboard
────────────────────────────────────────────────────────────
Run with:  streamlit run dashboard.py

Fixes applied:
  • DASH-7: True timer-based auto-refresh via streamlit-autorefresh package
            The old session_state approach only fired on user interaction,
            meaning the dashboard never refreshed when left idle
"""

import streamlit as st
import pandas as pd
import os
import yaml
from PIL import Image
from datetime import datetime

# DASH-7: proper auto-refresh — fires even with no user interaction
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

LOG_CSV   = CFG["paths"]["log_csv"]
VIO_DIR   = CFG["paths"]["violations_dir"]
REFRESH_S = int(CFG.get("dashboard", {}).get("refresh_seconds", 3))

st.set_page_config(
    page_title="Traffic Violation Dashboard",
    page_icon="🚦",
    layout="wide",
)

# DASH-7: st_autorefresh triggers a true rerun every N milliseconds
st_autorefresh(interval=REFRESH_S * 1000, key="auto_refresh")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
EMPTY_DF = pd.DataFrame(columns=[
    "Timestamp", "Vehicle_ID", "Violation",
    "License_Plate", "Speed_kmh", "Image",
])

@st.cache_data(ttl=REFRESH_S)
def load_data() -> pd.DataFrame:
    if not os.path.exists(LOG_CSV):
        return EMPTY_DF.copy()
    try:
        if os.path.getsize(LOG_CSV) == 0:
            return EMPTY_DF.copy()
        df = pd.read_csv(LOG_CSV)
        if df.empty:
            return EMPTY_DF.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.sort_values("Timestamp", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not load violation log: {e}")
        return EMPTY_DF.copy()

df = load_data()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("🚦 Smart Traffic Violation Dashboard")
st.caption(
    f"Last updated: {datetime.now().strftime('%H:%M:%S')}  "
    f"•  Auto-refreshes every {REFRESH_S}s"
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def kpi(col, label, value, color="#fff"):
    col.markdown(
        f"""
        <div style='background:#1e1e2e;padding:16px;border-radius:10px;text-align:center'>
            <div style='font-size:28px;font-weight:bold;color:{color}'>{value}</div>
            <div style='font-size:13px;color:#aaa'>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

total     = len(df)
red_light = int((df["Violation"] == "RED LIGHT").sum()) if not df.empty else 0
speeding  = int((df["Violation"] == "SPEEDING").sum())  if not df.empty else 0
wrong_way = int((df["Violation"] == "WRONG WAY").sum()) if not df.empty else 0
no_helmet = int((df["Violation"] == "NO HELMET").sum()) if not df.empty else 0

kpi(col1, "Total Violations", total,     "#e74c3c")
kpi(col2, "🔴 Red Light",     red_light, "#e74c3c")
kpi(col3, "💨 Speeding",      speeding,  "#f39c12")
kpi(col4, "⬆ Wrong Way",      wrong_way, "#9b59b6")
kpi(col5, "⛑ No Helmet",      no_helmet, "#3498db")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# CHARTS — always uses full df, not filtered
# ─────────────────────────────────────────────────────────────
if not df.empty:
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Violations by Type")
        type_counts = df["Violation"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        st.bar_chart(type_counts.set_index("Type"))

    with chart_col2:
        st.subheader("Violations by Hour")
        hour_df = df.dropna(subset=["Timestamp"]).copy()
        hour_df["Hour"] = hour_df["Timestamp"].dt.hour
        hourly = hour_df.groupby("Hour").size().reset_index(name="Count")
        if not hourly.empty:
            st.bar_chart(hourly.set_index("Hour"))

    st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FILTER + TABLE
# ─────────────────────────────────────────────────────────────
st.subheader("📋 Violation Log")

filter_col1, filter_col2 = st.columns([2, 1])
with filter_col1:
    violation_types = ["All"] + (df["Violation"].unique().tolist() if not df.empty else [])
    selected_type   = st.selectbox("Filter by violation type", violation_types)
with filter_col2:
    search_plate = st.text_input("Search by plate", "")

filtered = df.copy()
if selected_type != "All":
    filtered = filtered[filtered["Violation"] == selected_type]
if search_plate:
    filtered = filtered[
        filtered["License_Plate"].astype(str).str.contains(search_plate.upper(), na=False)
    ]

if filtered.empty:
    st.info("No violations match the current filter.")
else:
    display_cols = [c for c in
                    ["Timestamp", "Vehicle_ID", "Violation", "License_Plate", "Speed_kmh"]
                    if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=300)

st.download_button(
    "⬇ Download CSV",
    filtered.to_csv(index=False).encode(),
    file_name="violations_export.csv",
    mime="text/csv",
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# EVIDENCE PHOTO VIEWER
# ─────────────────────────────────────────────────────────────
st.subheader("📸 Evidence Photos")

if df.empty:
    st.info("No violations recorded yet. The system is running...")
elif filtered.empty:
    st.info("No photos match the current filter.")
else:
    recent = filtered.head(12)
    cols   = st.columns(4)
    for i, (_, row) in enumerate(recent.iterrows()):
        img_path = os.path.join(VIO_DIR, str(row.get("Image", "")))
        with cols[i % 4]:
            st.markdown(
                f"**ID {row['Vehicle_ID']}** — {row['Violation']}"
                f"  \n`{row['License_Plate']}`"
                f"  \n<small>{row['Timestamp']}</small>",
                unsafe_allow_html=True,
            )
            if os.path.exists(img_path):
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.warning("Image not found")
            st.markdown("---")