---
title: AI Traffic Vision
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: true
license: mit
tags:
  - computer-vision
  - yolov8
  - object-detection
  - traffic-monitoring
  - streamlit
  - opencv
  - deep-learning
  - bytetrack
  - vehicle-detection
  - python
---

# 🚦 AI Traffic Vision

> **Real-time computer vision traffic monitoring powered by YOLOv8 and ByteTrack.**

Upload a traffic video → detect vehicles → track them → count crossings → analyse density & congestion.

## Features

| Feature | Description |
|---|---|
| **Vehicle detection** | YOLOv8n — car, motorcycle, bus, truck |
| **Tracking** | ByteTrack — consistent IDs across frames |
| **Line counting** | Configurable virtual line; each vehicle counted once |
| **Traffic density** | Normalised score 0-100, classified LOW / MEDIUM / HIGH |
| **Congestion score** | Weighted score 0-100, classified LOW / MODERATE / HIGH / SEVERE |
| **Annotated video** | Bounding boxes, labels, confidence, IDs, HUD overlay |
| **CSV export** | Per-frame statistics for downstream analysis |
| **Plotly charts** | Vehicle count, density, congestion, class distribution |

## How to Use

1. Upload an MP4 traffic video using the sidebar
2. Adjust confidence threshold, density calibration, and counting line position
3. Click **▶ Process Video**
4. View annotated video, KPI cards, and charts
5. Download processed video and CSV

## Architecture

```
Upload MP4
    ↓
VideoProcessor (pipeline.py) — frame-by-frame
    ↓
VehicleTracker  — YOLOv8n + ByteTrack
    ↓
VehicleCounter  — virtual line crossing, unique IDs
    ↓
TrafficAnalytics — density + congestion scores
    ↓
Overlay renderer — boxes · labels · IDs · line · HUD
    ↓
output/processed.mp4  +  data/traffic_data.csv
```

## Technology Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- ByteTrack (built into Ultralytics)
- OpenCV
- Streamlit
- Plotly
- Pandas / NumPy

## Limitations

> ⚠️ Density and congestion scores are **estimates** based on computer-vision measurements.
> They are NOT scientifically calibrated. Accuracy depends on camera angle, video quality,
> detection accuracy, and the max-vehicles calibration value.
> Speed is not estimated — doing so reliably requires camera calibration data.
> 
## License

MIT License
