# 🚦 AI Traffic Vision

> **Real-time computer vision traffic monitoring powered by YOLOv8 and ByteTrack.**

---

## Problem Statement

Urban traffic congestion costs billions in lost productivity and emissions every year.
Road authorities need cost-effective tools to monitor traffic volume, detect congestion early,
and extract actionable insights — without requiring expensive in-road sensors.

## Solution

**AI Traffic Vision** processes any traffic camera video using state-of-the-art deep learning:
- Detects vehicles frame-by-frame with YOLOv8
- Tracks individual vehicles across frames with ByteTrack
- Counts vehicles crossing a configurable virtual line
- Estimates traffic density and congestion in real time
- Exports annotated video and per-frame CSV statistics

---

## Features

| Feature | Description |
|---|---|
| **Vehicle detection** | YOLOv8 — car, motorcycle, bus, truck |
| **Tracking** | ByteTrack — consistent IDs across frames |
| **Line counting** | Configurable virtual line; each vehicle counted once |
| **Traffic density** | Normalised score 0-100, classified LOW / MEDIUM / HIGH |
| **Congestion score** | Weighted score 0-100, classified LOW / MODERATE / HIGH / SEVERE |
| **Annotated video** | Bounding boxes, class labels, confidence, tracking IDs, HUD overlay |
| **CSV export** | Per-frame statistics for downstream analysis |
| **Plotly charts** | Vehicle count, density, congestion, class distribution over time |
| **Streamlit UI** | Dark professional dashboard with download buttons |

---

## Architecture

```
Upload MP4
    │
    ▼
VideoProcessor (pipeline.py)
    │  frame-by-frame (no full RAM load)
    │
    ├── VehicleTracker (tracker.py)     YOLOv8n + ByteTrack
    │       filters → car / motorcycle / bus / truck
    │
    ├── VehicleCounter (tracker.py)     virtual line crossing, unique IDs
    │
    ├── TrafficAnalytics (analytics.py) density + congestion scores
    │
    └── Overlay renderer (video_utils.py)
            boxes · labels · IDs · line · HUD panel
    │
    ▼
output/processed.mp4  +  data/traffic_data.csv
    │
    ▼
Streamlit Dashboard (app.py)
    KPI cards · Video · Charts · Downloads
```

---

## Technology Stack

| Layer | Library |
|---|---|
| **Object detection** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **Multi-object tracking** | ByteTrack (built into Ultralytics) |
| **Computer vision** | OpenCV |
| **Numerical computing** | NumPy |
| **Data analysis** | Pandas |
| **Dashboard** | Streamlit |
| **Charts** | Plotly |
| **Language** | Python 3.11+ |

---

## ML Pipeline

```
Input frame (BGR)
    │
    ▼  model.track(frame, persist=True, tracker="bytetrack.yaml")
YOLO detection head
    │  confidence filter (default 0.40)
    │  class filter: {2:car, 3:motorcycle, 5:bus, 7:truck}
    ▼
ByteTrack associator
    │  Kalman filter prediction
    │  IoU matching (confirmed + unconfirmed tracks)
    │  lost track buffering
    ▼
Tracked objects  [{track_id, class, bbox, center, conf}, ...]
    │
    ▼
VehicleCounter
    │  per-frame centre-Y vs line-Y comparison
    │  set of counted IDs (no double count)
    ▼
TrafficAnalytics
    density_score  = min(100, count / max_vehicles * 100)
    congestion     = 0.65 * density + 0.35 * normalised_count * 100
```

---

## Computer Vision Pipeline

1. **Frame extraction** — OpenCV reads frames one at a time (no full-video RAM load)
2. **Detection** — YOLOv8 runs on each frame; only vehicle classes kept
3. **Bounding boxes** — drawn in class-specific colours with confidence and ID label
4. **Tracking** — ByteTrack links detections across frames via Kalman + IoU
5. **Counting line** — dashed horizontal line drawn at configurable height
6. **HUD overlay** — semi-transparent info panel: counts, density, congestion
7. **Video writing** — OpenCV VideoWriter (mp4v codec) preserves original FPS and resolution

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/AI-Traffic-Vision.git
cd AI-Traffic-Vision

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify the model works (auto-downloads yolov8n.pt)
python test_model.py
```

---

## How to Run

```bash
# Start the dashboard
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

1. Upload an MP4 traffic video in the sidebar
2. Adjust confidence threshold, density calibration, and line position if needed
3. Click **▶ Process Video**
4. View annotated video, KPI cards, and charts
5. Download processed video and CSV

### Command-line pipeline (no UI)

```bash
python pipeline.py path/to/traffic.mp4 --output output/processed.mp4 --conf 0.4
```

---

## Project Structure

```
AI-Traffic-Vision/
├── app.py              Streamlit dashboard
├── detector.py         YOLO detection class (detection only, no tracking)
├── tracker.py          VehicleTracker + VehicleCounter
├── analytics.py        Density & congestion estimation
├── pipeline.py         End-to-end video processing orchestrator
├── config.py           All configurable parameters
├── test_model.py       Model sanity check
├── requirements.txt
├── README.md
├── models/             yolov8n.pt stored here (auto-downloaded)
├── input/              Place input videos here for CLI use
├── output/             Processed video written here
├── utils/
│   ├── __init__.py
│   └── video_utils.py  OpenCV helpers + overlay drawing
└── data/               traffic_data.csv written here
```

---

## Example Workflow

```
Upload 2-minute road-junction MP4
→ 30 FPS → 3 600 frames processed
→ 47 unique vehicles detected and tracked
→ 47 line crossings recorded (cars: 31, motorcycles: 9, buses: 4, trucks: 3)
→ Average density: 38 %  (MEDIUM)
→ Average congestion score: 42  (MODERATE)
→ Processed video + CSV exported
```

---

## Metrics

### Traffic Density Score

```
density_score = min(100, (vehicles_in_frame / max_vehicles) × 100)

LOW      0 – 35
MEDIUM  36 – 70
HIGH    71 – 100
```

`max_vehicles` defaults to 20; adjust in the sidebar to match your camera FOV.

### Congestion Score

```
congestion = 0.65 × density_score + 0.35 × (vehicles_in_frame / max_vehicles) × 100

LOW       0 – 30
MODERATE 31 – 60
HIGH     61 – 80
SEVERE   81 – 100
```

---

## Limitations

> **Important:** Please read before drawing conclusions from the outputs.

- **Not scientifically calibrated.** Density and congestion are estimates derived from
  pixel-level detections, not physical sensors.
- **Camera-dependent.** Accuracy varies significantly with camera angle, mount height,
  lens focal length, and video resolution.  Results from a bird-eye camera differ from
  a low-angle roadside camera.
- **No speed estimation.** Reliable speed calculation requires camera calibration
  (focal length, mount height, tilt angle).  Speed is intentionally not included to
  avoid presenting fabricated data.
- **YOLOv8n accuracy.** The Nano model trades some accuracy for speed.  Swap to
  `yolov8s.pt` or larger for higher recall, especially for small/occluded vehicles.
- **Counting line.** Counting depends on vehicles crossing a single line.  Vehicles that
  never cross the line (e.g., parked cars) are not counted.
- **Occlusion.** Heavily occluded vehicles may be missed or tracked with ID switches.
- **Hardware.** Processing speed depends heavily on hardware.  GPU strongly recommended
  for near-real-time processing.

---

## Future Improvements

- [ ] GPU / CUDA acceleration with device selection
- [ ] Multiple counting lines / zones
- [ ] Camera calibration input for speed estimation
- [ ] Real-time webcam / RTSP stream support
- [ ] Wrong-way vehicle detection
- [ ] Vehicle re-identification across cameras
- [ ] Alert system (congestion threshold notifications)
- [ ] Larger YOLO models (YOLOv8s/m/l/x) selectable in UI
- [ ] Heatmap of vehicle density over time
- [ ] Docker container for easy deployment

---

## Screenshots

_Add screenshots of the Streamlit dashboard here._

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is provided for research and educational purposes.
Traffic density and congestion outputs are computer-vision estimates and should not be
used as the sole basis for traffic engineering or public-safety decisions without
independent verification.
