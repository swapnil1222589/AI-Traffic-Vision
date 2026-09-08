"""
detector.py — Vehicle detection using Ultralytics YOLOv8.

This module is responsible ONLY for detection (no tracking).
It filters COCO detections to the four vehicle classes we care about.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import VEHICLE_CLASSES, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, MODEL_PATH
from utils.video_utils import draw_bounding_box


class VehicleDetector:
    """Load a YOLO model and run per-frame vehicle detection."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        """
        Parameters
        ----------
        model_path     : Path to .pt file.  Ultralytics auto-downloads if missing.
        conf_threshold : Only keep detections above this confidence.
        iou_threshold  : NMS IoU threshold.
        """
        self.conf = conf_threshold
        self.iou  = iou_threshold
        print(f"[Detector] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        print("[Detector] Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Run detection on a single BGR frame.

        Returns
        -------
        annotated_frame : frame with bounding boxes drawn
        detections      : list of dicts, one per vehicle:
                          {class_id, class_name, confidence, bbox (x1,y1,x2,y2), center}
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            classes=list(VEHICLE_CLASSES.keys()),  # only vehicle class IDs
        )

        detections = self._parse_results(results)
        annotated  = self._draw_detections(frame.copy(), detections)
        return annotated, detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_results(self, results) -> list[dict]:
        """Extract per-box info from YOLO results."""
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id  = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                conf    = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    "class_id":   cls_id,
                    "class_name": VEHICLE_CLASSES[cls_id],
                    "confidence": conf,
                    "bbox":       (x1, y1, x2, y2),
                    "center":     (cx, cy),
                    "track_id":   None,  # filled in by tracker
                })
        return detections

    def _draw_detections(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes for all detections."""
        for det in detections:
            frame = draw_bounding_box(
                frame,
                bbox=det["bbox"],
                label=det["class_name"],
                conf=det["confidence"],
                track_id=det.get("track_id"),
            )
        return frame
