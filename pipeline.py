"""
pipeline.py — End-to-end video processing orchestrator.

Flow:
    Input MP4
    → OpenCV frame reader
    → VehicleTracker  (YOLO + ByteTrack)
    → VehicleCounter  (line-crossing)
    → TrafficAnalytics
    → Overlay renderer
    → Output MP4 + CSV
"""
            
import os
import sys
import time
import pandas as pd

import cv2

from config import (
    MODEL_PATH, OUTPUT_VIDEO, STATS_CSV,
    DEFAULT_CONF_THRESHOLD, DEFAULT_LINE_Y_RATIO, DENSITY_MAX_VEHICLES,
)
from tracker import VehicleTracker, VehicleCounter
from analytics import TrafficAnalytics
from utils.video_utils import open_video, create_writer, release_all, draw_hud


# ---------------------------------------------------------------------------

class VideoProcessor:
    """Orchestrate the full detection → tracking → counting → analytics pipeline."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        line_y_ratio: float = DEFAULT_LINE_Y_RATIO,
        density_max_vehicles: int = DENSITY_MAX_VEHICLES,
    ):
        self.conf              = conf_threshold
        self.line_y_ratio      = line_y_ratio
        self.density_max       = density_max_vehicles
        self.model_path        = model_path

        self.tracker   = VehicleTracker(model_path, conf_threshold)
        self.counter   = VehicleCounter(line_y_ratio)
        self.analytics = TrafficAnalytics()

    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str,
        output_path: str = OUTPUT_VIDEO,
        stats_path: str  = STATS_CSV,
        progress_callback=None,          # callable(frame_idx, total_frames)
    ) -> dict:
        """Process the video end-to-end.

        Parameters
        ----------
        input_path        : path to input MP4
        output_path       : path to write processed MP4
        stats_path        : path to write CSV statistics
        progress_callback : optional callable(current_frame, total_frames)

        Returns
        -------
        summary dict with aggregate statistics
        """
        self.counter.reset()

        # -- Open input ---------------------------------------------------
        cap, fps, w, h = open_video(input_path)
        total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # -- Create output ------------------------------------------------
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = create_writer(output_path, fps, w, h)

        # -- Per-frame stats list -----------------------------------------
        frame_stats: list[dict] = []

        frame_idx  = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                # --- Track -----------------------------------------------
                tracked = self.tracker.track(frame)

                # --- Count -----------------------------------------------
                self.counter.update(tracked, h, frame_idx, timestamp)

                # --- Analytics -------------------------------------------
                frame_vehicle_count = len(tracked)
                ana = self.analytics.analyse_frame(frame_vehicle_count, self.density_max)

                # --- Draw ------------------------------------------------
                out_frame = self.tracker.draw_tracked(frame, tracked)
                out_frame = self.counter.draw(out_frame)
                out_frame = draw_hud(
                    out_frame, ana,
                    self.counter.counts,
                    self.counter.total,
                    frame_vehicle_count,
                )

                # --- Write -----------------------------------------------
                writer.write(out_frame)

                # --- Record stats row ------------------------------------
                frame_stats.append({
                    "frame_number":    frame_idx,
                    "timestamp_sec":   round(timestamp, 3),
                    "vehicles_in_frame": frame_vehicle_count,
                    "total_crossed":   self.counter.total,
                    "cars":            self.counter.counts.get("car",        0),
                    "motorcycles":     self.counter.counts.get("motorcycle", 0),
                    "buses":           self.counter.counts.get("bus",        0),
                    "trucks":          self.counter.counts.get("truck",      0),
                    "density_score":   ana["density_score"],
                    "density_level":   ana["density_level"],
                    "congestion_score":ana["congestion_score"],
                    "congestion_level":ana["congestion_level"],
                })

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        finally:
            release_all(cap, writer)

        # -- Save CSV -----------------------------------------------------
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        df = pd.DataFrame(frame_stats)
        df.to_csv(stats_path, index=False)

        elapsed = time.time() - start_time
        summary = {
            "frames_processed":  frame_idx,
            "total_crossed":     self.counter.total,
            "counts":            dict(self.counter.counts),
            "avg_density":       round(df["density_score"].mean(),    1) if len(df) else 0,
            "avg_congestion":    round(df["congestion_score"].mean(), 1) if len(df) else 0,
            "peak_density":      round(df["density_score"].max(),     1) if len(df) else 0,
            "peak_congestion":   round(df["congestion_score"].max(),  1) if len(df) else 0,
            "processing_time_s": round(elapsed, 1),
            "output_path":       output_path,
            "stats_path":        stats_path,
            "events":            self.counter.events,
            "fps":               fps,
            "resolution":        (w, h),
        }
        return summary


# ---------------------------------------------------------------------------
# CLI test helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Traffic Vision — pipeline test")
    parser.add_argument("video", help="Path to input MP4")
    parser.add_argument("--output", default=OUTPUT_VIDEO)
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF_THRESHOLD)
    args = parser.parse_args()

    def progress(cur, total):
        pct = (cur / total * 100) if total else 0
        print(f"\r  Frame {cur}/{total}  ({pct:.1f}%)", end="", flush=True)

    proc = VideoProcessor(conf_threshold=args.conf)
    result = proc.process(args.video, args.output, progress_callback=progress)
    print("\n\n=== Summary ===")
    for k, v in result.items():
        if k not in ("events",):
            print(f"  {k}: {v}")
