"""
app.py — AI Traffic Vision — Streamlit Dashboard.

Professional dark-themed dashboard for real-time traffic analysis.
All metrics are derived from actual YOLO detection + ByteTrack output.
No hardcoded or fake data.
"""

import os
import sys
import tempfile
import time

import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Traffic Vision",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Project root on sys.path so local modules resolve correctly
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_PATH, OUTPUT_VIDEO, STATS_CSV, DATA_DIR, OUTPUT_DIR,
    DEFAULT_CONF_THRESHOLD, DEFAULT_LINE_Y_RATIO, DENSITY_MAX_VEHICLES,
    DENSITY_LOW_MAX, DENSITY_MEDIUM_MAX,
    CONGESTION_LOW_MAX, CONGESTION_MODERATE_MAX, CONGESTION_HIGH_MAX,
)
from pipeline import VideoProcessor

# ---------------------------------------------------------------------------
# CSS — dark professional theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
    }
    .main-header h1 { color: #58a6ff; margin: 0; font-size: 2rem; }
    .main-header p  { color: #8b949e; margin: 4px 0 0 0; font-size: 0.95rem; }

    /* ── KPI cards ── */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #58a6ff; line-height: 1.1; }
    .kpi-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-top: 4px; }

    /* ── Section headers ── */
    .section-header {
        color: #c9d1d9; font-size: 1.05rem; font-weight: 600;
        border-bottom: 1px solid #30363d; padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }

    /* ── Metric badges ── */
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.05em;
    }
    .badge-low      { background:#1a3a2a; color:#3fb950; border:1px solid #238636; }
    .badge-medium   { background:#2d2a1a; color:#d29922; border:1px solid #9e6a03; }
    .badge-high     { background:#3a1a1a; color:#f78166; border:1px solid #da3633; }
    .badge-severe   { background:#2d0d0d; color:#ff7b72; border:1px solid #b62324; }
    .badge-moderate { background:#2a2040; color:#a371f7; border:1px solid #6e40c9; }

    /* ── Info box ── */
    .info-box {
        background:#161b22; border:1px solid #30363d; border-radius:8px;
        padding:14px 18px; margin:8px 0; font-size:0.88rem; color:#8b949e;
    }
    .info-box strong { color:#c9d1d9; }

    /* ── Plotly chart wrapper ── */
    .chart-container { border:1px solid #30363d; border-radius:10px; padding:4px; }

    /* ── Progress ── */
    div[data-testid="stProgress"] > div { background-color: #58a6ff !important; }

    /* ── Buttons ── */
    .stDownloadButton button {
        background:#1f6feb; color:white; border:none; border-radius:6px;
        padding:8px 20px; font-weight:600;
    }
    .stDownloadButton button:hover { background:#388bfd; }
    div[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def level_badge(level: str) -> str:
    cls_map = {
        "LOW": "badge-low", "MEDIUM": "badge-medium",
        "HIGH": "badge-high", "SEVERE": "badge-severe",
        "MODERATE": "badge-moderate",
    }
    css = cls_map.get(level.upper(), "badge-medium")
    return f'<span class="badge {css}">{level}</span>'


def kpi_card(value, label: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def plotly_dark_layout(fig, title=""):
    fig.update_layout(
        title=title,
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        title_font_color="#58a6ff",
        xaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        yaxis=dict(gridcolor="#30363d", zerolinecolor="#30363d"),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🚦 AI Traffic Vision")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload an MP4 (or compatible) traffic video to analyse.",
    )

    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.95, value=DEFAULT_CONF_THRESHOLD, step=0.05,
        help="Minimum confidence for a detection to be kept.",
    )

    st.markdown("### 📊 Analytics Settings")
    density_max = st.slider(
        "Max Vehicles (density calibration)",
        min_value=5, max_value=60, value=DENSITY_MAX_VEHICLES, step=1,
        help="Vehicle count that equals 100% density. Adjust per camera FOV.",
    )

    st.markdown("### 🚧 Counting Line")
    line_y_ratio = st.slider(
        "Line Position (fraction of frame height)",
        min_value=0.20, max_value=0.85, value=DEFAULT_LINE_Y_RATIO, step=0.05,
        help="0.5 = middle of frame. Move up/down to suit the road layout.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;'>"
        "Model: YOLOv8n + ByteTrack<br>"
        "Classes: car · motorcycle · bus · truck<br>"
        "Density & congestion are <em>estimates</em>."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main page header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🚦 AI Traffic Vision</h1>
    <p>Real-Time Traffic Monitoring & Analytics — Powered by YOLOv8 + ByteTrack</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "processed":    False,
        "summary":      None,
        "df":           None,
        "output_path":  None,
        "stats_path":   None,
        "error":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if uploaded_file is None:
    st.info("👈 Upload a traffic video in the sidebar to get started.")

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>1. Upload</strong><br>
            Upload any MP4 traffic video captured from a road-side or overhead camera.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>2. Analyse</strong><br>
            YOLOv8 detects vehicles frame-by-frame. ByteTrack assigns consistent IDs
            and counts crossings of a virtual line.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-box">
            <strong>3. Export</strong><br>
            Download the annotated video and a per-frame CSV with density,
            congestion and vehicle counts.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">About the ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Detection</strong> — YOLOv8n (Nano) runs on every frame and filters COCO
        detections to four vehicle classes: <em>car, motorcycle, bus, truck</em>.<br><br>
        <strong>Tracking</strong> — ByteTrack associates detections across frames using
        a Kalman filter and IoU matching, producing stable per-vehicle IDs.<br><br>
        <strong>Counting</strong> — A configurable virtual horizontal line is drawn across
        the frame. A vehicle is counted <em>once</em> when its tracked centre crosses the line.<br><br>
        <strong>Density</strong> — Estimated as <code>(vehicles in frame / calibration max) × 100</code>.
        Classified LOW / MEDIUM / HIGH.<br><br>
        <strong>Congestion</strong> — A weighted combination of density score and normalised
        vehicle count. Classified LOW / MODERATE / HIGH / SEVERE.<br><br>
        <em>⚠️ Speed is not estimated — doing so reliably requires camera calibration data
        (focal length, mount height, angle) that is not available from a generic video.</em>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Process button
# ---------------------------------------------------------------------------

process_col, _ = st.columns([1, 3])
with process_col:
    run_btn = st.button("▶  Process Video", type="primary", use_container_width=True)

if run_btn:
    st.session_state.processed = False
    st.session_state.error     = None

    # Save upload to a temp file
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Output paths (inside project output / data dirs)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)
    out_video = os.path.join(OUTPUT_DIR, "processed.mp4")
    out_csv   = os.path.join(DATA_DIR,   "traffic_data.csv")

    # Progress UI
    progress_bar  = st.progress(0, text="Initialising…")
    status_text   = st.empty()

    def on_progress(cur, total):
        pct = int(cur / total * 100) if total else 0
        progress_bar.progress(pct / 100, text=f"Processing frame {cur} / {total}  ({pct}%)")

    try:
        proc = VideoProcessor(
            conf_threshold       = conf_threshold,
            line_y_ratio         = line_y_ratio,
            density_max_vehicles = density_max,
        )
        summary = proc.process(tmp_path, out_video, out_csv, progress_callback=on_progress)

        progress_bar.progress(1.0, text="✅ Processing complete!")
        status_text.success(f"Processed {summary['frames_processed']} frames in {summary['processing_time_s']}s")

        st.session_state.processed   = True
        st.session_state.summary     = summary
        st.session_state.output_path = out_video
        st.session_state.stats_path  = out_csv
        if os.path.exists(out_csv):
            st.session_state.df = pd.read_csv(out_csv)

    except Exception as exc:
        st.session_state.error = str(exc)
        progress_bar.empty()
        status_text.empty()

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Error state
# ---------------------------------------------------------------------------

if st.session_state.error:
    st.error(f"❌ Processing failed:\n\n{st.session_state.error}")
    st.stop()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if not st.session_state.processed:
    st.stop()

summary = st.session_state.summary
df      = st.session_state.df
counts  = summary.get("counts", {})


# ── KPI cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Vehicle Counts (Line Crossings)</div>', unsafe_allow_html=True)
c0, c1, c2, c3, c4 = st.columns(5)
with c0: st.markdown(kpi_card(summary["total_crossed"], "Total Vehicles"), unsafe_allow_html=True)
with c1: st.markdown(kpi_card(counts.get("car",         0), "🚗 Cars"),         unsafe_allow_html=True)
with c2: st.markdown(kpi_card(counts.get("motorcycle",  0), "🏍 Motorcycles"),   unsafe_allow_html=True)
with c3: st.markdown(kpi_card(counts.get("bus",         0), "🚌 Buses"),         unsafe_allow_html=True)
with c4: st.markdown(kpi_card(counts.get("truck",       0), "🚛 Trucks"),        unsafe_allow_html=True)


# ── Processed video ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Processed Video</div>', unsafe_allow_html=True)
out_path = st.session_state.output_path
if out_path and os.path.exists(out_path):
    with open(out_path, "rb") as vf:
        video_bytes = vf.read()
    st.video(video_bytes)
else:
    st.warning("Processed video not found. Check output/ directory.")


# ── Analytics summary ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">Traffic Analytics</div>', unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)

with a1:
    st.metric("Avg Density Score", f"{summary['avg_density']}%")
with a2:
    st.metric("Peak Density",      f"{summary['peak_density']}%")
with a3:
    st.metric("Avg Congestion",    f"{summary['avg_congestion']}")
with a4:
    st.metric("Peak Congestion",   f"{summary['peak_congestion']}")

if df is not None and not df.empty:
    last = df.iloc[-1]
    col_d, col_c = st.columns(2)
    with col_d:
        dl = last["density_level"]
        st.markdown(
            f"**Final Density Level:** {level_badge(dl)}",
            unsafe_allow_html=True,
        )
    with col_c:
        cl = last["congestion_level"]
        st.markdown(
            f"**Final Congestion Level:** {level_badge(cl)}",
            unsafe_allow_html=True,
        )

st.markdown("""
<div class="info-box">
    ⚠️ <strong>Disclaimer:</strong> Density and congestion scores are <em>estimates</em>
    based on computer-vision measurements.  They are not scientifically calibrated values.
    Accuracy depends on camera angle, video quality, detection accuracy, and the
    "Max Vehicles" calibration parameter.
</div>""", unsafe_allow_html=True)


# ── Charts ─────────────────────────────────────────────────────────────────
if df is not None and not df.empty:
    st.markdown('<div class="section-header">Traffic Charts</div>', unsafe_allow_html=True)

    # Thin out for very long videos (plot every Nth row)
    max_plot_rows = 2000
    step = max(1, len(df) // max_plot_rows)
    dfs  = df.iloc[::step].copy()

    ch1, ch2 = st.columns(2)

    with ch1:
        fig1 = px.line(
            dfs, x="timestamp_sec", y="vehicles_in_frame",
            color_discrete_sequence=["#58a6ff"],
            labels={"timestamp_sec": "Time (s)", "vehicles_in_frame": "Vehicles in Frame"},
        )
        fig1 = plotly_dark_layout(fig1, "Vehicles in Frame over Time")
        st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        fig2 = px.line(
            dfs, x="timestamp_sec", y="density_score",
            color_discrete_sequence=["#3fb950"],
            labels={"timestamp_sec": "Time (s)", "density_score": "Density Score (%)"},
        )
        fig2 = plotly_dark_layout(fig2, "Traffic Density over Time")
        st.plotly_chart(fig2, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        fig3 = px.line(
            dfs, x="timestamp_sec", y="congestion_score",
            color_discrete_sequence=["#f78166"],
            labels={"timestamp_sec": "Time (s)", "congestion_score": "Congestion Score"},
        )
        fig3 = plotly_dark_layout(fig3, "Congestion Score over Time")
        st.plotly_chart(fig3, use_container_width=True)

    with ch4:
        pie_data = {
            "Class": ["Cars", "Motorcycles", "Buses", "Trucks"],
            "Count": [
                counts.get("car",        0),
                counts.get("motorcycle", 0),
                counts.get("bus",        0),
                counts.get("truck",      0),
            ],
        }
        pie_df = pd.DataFrame(pie_data)
        pie_df = pie_df[pie_df["Count"] > 0]
        if not pie_df.empty:
            fig4 = px.pie(
                pie_df, values="Count", names="Class",
                color_discrete_sequence=["#58a6ff","#3fb950","#d29922","#f78166"],
                hole=0.45,
            )
            fig4 = plotly_dark_layout(fig4, "Vehicle Class Distribution")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No vehicles crossed the counting line — no distribution to show.")

    # Cumulative crossings over time
    fig5 = px.line(
        dfs, x="timestamp_sec", y="total_crossed",
        color_discrete_sequence=["#a371f7"],
        labels={"timestamp_sec": "Time (s)", "total_crossed": "Cumulative Vehicles Crossed"},
    )
    fig5 = plotly_dark_layout(fig5, "Cumulative Vehicle Count over Time")
    st.plotly_chart(fig5, use_container_width=True)


# ── Downloads ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
dl1, dl2 = st.columns(2)

with dl1:
    if out_path and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️  Download Processed Video",
                data=f,
                file_name="ai_traffic_vision_processed.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

with dl2:
    csv_path = st.session_state.stats_path
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                "⬇️  Download Statistics CSV",
                data=f,
                file_name="traffic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Processing info ────────────────────────────────────────────────────────
with st.expander("📋 Processing Details"):
    res = summary.get("resolution", (0, 0))
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Frames processed | {summary['frames_processed']} |
    | Video resolution | {res[0]} × {res[1]} |
    | Original FPS | {summary['fps']:.2f} |
    | Processing time | {summary['processing_time_s']} s |
    | Confidence threshold | {conf_threshold} |
    | Counting line position | {line_y_ratio:.0%} of frame height |
    | Density calibration max | {density_max} vehicles |
    """)

