"""
tracker.py — Vehicle tracking (ByteTrack via Ultralytics) and line-crossing counter.

The tracker uses YOLO`s built-in model.track() which runs ByteTrack internally.
The counter keeps track of unique IDs that have crossed the virtual line.
"""
     
import numpy as np
from ultralytics import YOLO
                  
from config import (
    VEHICLE_CLASSES, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    TRACKER_CONFIG, MODEL_PATH, DEFAULT_LINE_Y_RATIO
)
from utils.video_utils import draw_bounding_box, draw_counting_line


# ---------------------------------------------------------------------------                
# Tracker
# ---------------------------------------------------------------------------

class VehicleTracker:
    """Run YOLO ByteTrack on each frame and return tracked vehicle objects."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.conf = conf_threshold
        self.iou  = iou_threshold
        print(f"[Tracker] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("[Tracker] Model + ByteTrack ready.")

    def track(self, frame: np.ndarray) -> list[dict]:
        """Run tracking on a single BGR frame.

        Returns a list of tracked vehicle dicts:
            {track_id, class_id, class_name, confidence, bbox, center}
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            persist=True,                        # maintain ID history across calls
            tracker=TRACKER_CONFIG,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )
        return self._parse_results(results)

    # ------------------------------------------------------------------

    def _parse_results(self, results) -> list[dict]:
        tracked = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                track_id = int(box.id[0]) if box.id is not None else None
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                tracked.append({
                    "track_id":   track_id,
                    "class_id":   cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "center":     (cx, cy),
                })
        return tracked

    def draw_tracked(self, frame: np.ndarray, tracked: list[dict]) -> np.ndarray:
        """Draw bounding boxes with IDs onto frame (in-place copy)."""
        out = frame.copy()
        for obj in tracked:
            out = draw_bounding_box(
                out,
                bbox=obj["bbox"],
                label=obj["class_name"],
                conf=obj["confidence"],
                track_id=obj["track_id"],
            )
        return out


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

class VehicleCounter:
    """Count unique vehicles as they cross a virtual horizontal line.

    A vehicle is counted once: the first time its tracked center crosses
    the line (travelling in either direction).
    """

    CLASSES = ["car", "motorcycle", "bus", "truck"]

    def __init__(self, line_y_ratio: float = DEFAULT_LINE_Y_RATIO):
        """
        Parameters
        ----------
        line_y_ratio : fraction of frame height (0-1) where the line sits.
        """
        self.line_y_ratio = line_y_ratio

        # Unique IDs that have already crossed — never counted twice
        self._counted_ids: set[int] = set()

        # Per-class counts
        self.counts: dict[str, int] = {c: 0 for c in self.CLASSES}

        # History: list of event dicts for CSV export
        self.events: list[dict] = []

        # {track_id: previous_cy} — used to detect line crossing direction
        self._prev_cy: dict[int, int] = {}

    # ------------------------------------------------------------------
    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def line_y(self, frame_h: int) -> int:
        return int(frame_h * self.line_y_ratio)

    # ------------------------------------------------------------------
    def update(
        self,
        tracked_objects: list[dict],
        frame_h: int,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> list[dict]:
        """Check each tracked object for line crossing.

        Returns list of newly-counted vehicle dicts in this frame.
        """
        line = self.line_y(frame_h)
        newly_counted = []

        for obj in tracked_objects:
            tid = obj.get("track_id")
            if tid is None:
                continue

            cy        = obj["center"][1]
            prev_cy   = self._prev_cy.get(tid)

            # Detect crossing: center moved from one side of line to other
            crossed = (
                prev_cy is not None
                and tid not in self._counted_ids
                and (
                    (prev_cy < line <= cy) or   # downward crossing
                    (prev_cy > line >= cy)       # upward crossing
                )
            )

            if crossed:
                self._counted_ids.add(tid)
                cls = obj["class_name"]
                if cls in self.counts:
                    self.counts[cls] += 1
                event = {
                    "track_id":      tid,
                    "class_name":    cls,
                    "frame_number":  frame_number,
                    "timestamp_sec": round(timestamp, 3),
                }
                self.events.append(event)
                newly_counted.append(obj)

            self._prev_cy[tid] = cy

        return newly_counted

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the counting line onto frame."""
        h = frame.shape[0]
        return draw_counting_line(frame, self.line_y(h), self.counts, self.total)

    def reset(self):
        """Reset all counts (e.g., for a new video)."""
        self._counted_ids.clear()
        self.counts = {c: 0 for c in self.CLASSES}
        self.events.clear()
        self._prev_cy.clear()
