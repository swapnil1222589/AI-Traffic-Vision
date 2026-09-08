"""
utils/video_utils.py — OpenCV video helpers and frame-overlay utilities.
"""

import cv2
import numpy as np
from config import (
    CLASS_COLOURS, COLOUR_LINE, COLOUR_HUD_BG, COLOUR_HUD_TEXT,
    COLOUR_DEFAULT, OUTPUT_CODEC, OUTPUT_EXT
)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def open_video(path: str):
    """Open a video file and return (cap, fps, width, height).

    Raises RuntimeError if the file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def create_writer(path: str, fps: float, width: int, height: int):
    """Create an OpenCV VideoWriter.  Falls back codec if needed."""
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video writer at: {path}")
    return writer


def release_all(*resources):
    """Safely release VideoCapture and VideoWriter objects."""
    for r in resources:
        if r is not None:
            try:
                r.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_bounding_box(
    frame: np.ndarray,
    bbox: tuple,          # (x1, y1, x2, y2)
    label: str,
    conf: float,
    track_id: int | None = None,
) -> np.ndarray:
    """Draw a coloured bounding box with label and optional tracking ID."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    colour = CLASS_COLOURS.get(label, COLOUR_DEFAULT)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    # Build label text
    id_part   = f"#{track_id} " if track_id is not None else ""
    text      = f"{id_part}{label} {conf:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Background pill for readability
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(
        frame, text, (x1 + 2, y1 - 4),
        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
    )
    return frame


def draw_counting_line(
    frame: np.ndarray,
    line_y: int,
    counts: dict,          # {"car": n, "motorcycle": n, ...}
    total_crossed: int,
) -> np.ndarray:
    """Draw the virtual counting line and the crossed-vehicle tally."""
    h, w = frame.shape[:2]
    # Dashed line
    dash_len = 20
    gap_len  = 10
    x = 0
    while x < w:
        x_end = min(x + dash_len, w)
        cv2.line(frame, (x, line_y), (x_end, line_y), COLOUR_LINE, 2)
        x += dash_len + gap_len

    # Label on the line
    label = f"COUNT LINE  |  Total crossed: {total_crossed}"
    cv2.putText(
        frame, label, (10, line_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_LINE, 1, cv2.LINE_AA
    )
    return frame


def draw_hud(
    frame: np.ndarray,
    analytics: dict,
    counts: dict,
    total_crossed: int,
    frame_vehicles: int,
) -> np.ndarray:
    """Draw a translucent HUD panel in the top-left corner."""
    h, w = frame.shape[:2]

    panel_w = 280
    panel_h = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), COLOUR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        "AI TRAFFIC VISION",
        f"In Frame: {frame_vehicles} vehicles",
        f"Total Crossed: {total_crossed}",
        f"  Cars: {counts.get('car', 0)}  Moto: {counts.get('motorcycle', 0)}",
        f"  Buses: {counts.get('bus', 0)}  Trucks: {counts.get('truck', 0)}",
        f"Density: {analytics.get('density_score', 0):.0f}%  [{analytics.get('density_level', '?')}]",
        f"Congestion: {analytics.get('congestion_score', 0):.0f}  [{analytics.get('congestion_level', '?')}]",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    y          = 22
    for i, line in enumerate(lines):
        colour = (100, 220, 255) if i == 0 else COLOUR_HUD_TEXT
        scale  = 0.52 if i == 0 else font_scale
        cv2.putText(frame, line, (8, y), font, scale, colour, 1, cv2.LINE_AA)
        y += 26

    return frame
