"""
test_model.py — Quick sanity-check: load YOLO model and run inference on
                a synthetic frame (no real video needed).

Run:  python test_model.py
"""

import sys
import os
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH, VEHICLE_CLASSES

print("=" * 55)
print("  AI Traffic Vision — Model Sanity Check")
print("=" * 55)

# 1. Load model
print(f"\n[1/3] Loading YOLO model: {MODEL_PATH}")
try:
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)          # auto-downloads yolov8n.pt if missing
    print("      ✅ Model loaded successfully.")
except Exception as e:
    print(f"      ❌ FAILED to load model: {e}")
    sys.exit(1)

# 2. Create a synthetic frame (640x640 random image)
print("\n[2/3] Creating synthetic test frame (640×640)...")
test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 3. Run inference
print("[3/3] Running inference...")
try:
    results = model.predict(
        test_frame,
        conf=0.25,
        classes=list(VEHICLE_CLASSES.keys()),
        verbose=False,
    )
    n_det = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"      ✅ Inference succeeded.  Detections on random frame: {n_det}")
    print("         (0 detections on a random frame is expected and OK.)")
except Exception as e:
    print(f"      ❌ Inference FAILED: {e}")
    sys.exit(1)

print("\n✅ All checks passed.  The model is ready.\n")
