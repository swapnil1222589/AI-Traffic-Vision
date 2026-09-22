"""
config.py — Central configuration for AI Traffic Vision.
All tunable parameters live here. Edit this file to customise behaviour.
"""

import os
import tempfile   

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUT_DIR  = os.path.join(BASE_DIR, "input")

# On Streamlit Cloud the repo is read-only — write outputs to /tmp.
# Locally (Windows) /tmp doesn't exist so fall back to the project dirs.
_TMP = tempfile.gettempdir()          # /tmp on Linux cloud, %TEMP% on Windows
OUTPUT_DIR = os.path.join(_TMP, "ai_traffic_vision", "output")
DATA_DIR   = os.path.join(_TMP, "ai_traffic_vision", "data")

# Model weights: auto-downloaded by Ultralytics into the models/ folder.
# On cloud the first run downloads yolov8n.pt into a writable cache dir.
MODEL_PATH   = os.path.join(MODELS_DIR, "yolov8n.pt")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "processed.mp4")
STATS_CSV    = os.path.join(DATA_DIR,   "traffic_data.csv")

# Ensure writable dirs exist at import time
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# COCO class IDs for vehicles we care about
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

DEFAULT_CONF_THRESHOLD = 0.40   # minimum detection confidence (0-1)
DEFAULT_IOU_THRESHOLD  = 0.45   # NMS IoU threshold

# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
TRACKER_CONFIG = "bytetrack.yaml"   # bundled inside Ultralytics

# ---------------------------------------------------------------------------
# Counting line
# ---------------------------------------------------------------------------
# Fraction of frame height where the virtual counting line sits (0.0 – 1.0)
# 0.5 = middle of the frame.  Adjust per camera angle.
DEFAULT_LINE_Y_RATIO = 0.55

# ---------------------------------------------------------------------------
# Traffic density
# ---------------------------------------------------------------------------
# Maximum number of vehicles expected in a single frame to achieve 100 % density.
# Adjust for your camera FOV / road type.
DENSITY_MAX_VEHICLES = 20

# Density score → level thresholds (inclusive upper bounds)
DENSITY_LOW_MAX    = 35   # 0-35   → LOW
DENSITY_MEDIUM_MAX = 70   # 36-70  → MEDIUM
                          # 71-100 → HIGH

# ---------------------------------------------------------------------------
# Congestion
# ---------------------------------------------------------------------------
# Congestion score → level thresholds (inclusive upper bounds)
CONGESTION_LOW_MAX      = 30   # 0-30   → LOW
CONGESTION_MODERATE_MAX = 60   # 31-60  → MODERATE
CONGESTION_HIGH_MAX     = 80   # 61-80  → HIGH
                               # 81-100 → SEVERE

# Weight of density vs raw count when computing congestion
CONGESTION_DENSITY_WEIGHT = 0.65
CONGESTION_COUNT_WEIGHT   = 0.35

# ---------------------------------------------------------------------------
# Overlay colours  (BGR)
# ---------------------------------------------------------------------------
COLOUR_CAR        = (0,   200, 255)
COLOUR_MOTORCYCLE = (0,   255, 100)
COLOUR_BUS        = (255, 100,   0)
COLOUR_TRUCK      = (255,   0, 180)
COLOUR_DEFAULT    = (200, 200, 200)
COLOUR_LINE       = (0,   255,   0)
COLOUR_HUD_BG     = (20,   20,  20)
COLOUR_HUD_TEXT   = (240, 240, 240)

CLASS_COLOURS = {
    "car":        COLOUR_CAR,
    "motorcycle": COLOUR_MOTORCYCLE,
    "bus":        COLOUR_BUS,
    "truck":      COLOUR_TRUCK,
}

# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------
OUTPUT_CODEC    = "mp4v"   # fallback codec; works everywhere without extras
OUTPUT_EXT      = ".mp4"
