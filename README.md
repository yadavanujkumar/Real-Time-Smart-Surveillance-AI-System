# Real-Time Smart Surveillance AI System

A production-grade AI/ML project demonstrating computer vision, deep learning, real-time tracking, behavior analysis, and a live web dashboard.

## Features
- **Real-Time Person Detection**: YOLOv8n (`yolov8n.pt`) — automatically downloaded on first run.
- **Multi-Object Tracking**: ByteTrack assigns stable IDs to individuals across frames.
- **Face Recognition**: Identifies known individuals using facial embeddings (powered by `face_recognition`).
- **Behavior Analysis**: Detects loitering and unauthorized entry into configurable restricted zones.
- **Real-Time Alerts**: `FACE_MATCH`, `LOITERING`, and `RESTRICTED_ACCESS` events are logged and sent to the API.
- **Screenshot Capture**: Annotated frames are saved to `data/screenshots/` whenever an alert fires.
- **FastAPI Backend**: Serves events, a live MJPEG video stream (`/stream/video`), and screenshots.
- **Streamlit Dashboard**: Displays the live feed, detection logs (colour-coded by type), recent screenshots, and system metrics.

## Folder Structure
```text
Real-Time-Smart-Surveillance-AI-System/
├── core/
│   ├── video_input.py          # Frame ingestion from webcam / IP camera / video file
│   ├── detection.py            # YOLOv8 + ByteTrack — person detection & tracking
│   ├── face_recognition.py     # Face embeddings and identity matching
│   ├── behavior_analysis.py    # Loitering & restricted-zone detection
│   └── alert_engine.py         # Async alert dispatch — API POST + screenshot saving
├── pipeline/
│   └── inference.py            # Main inference loop wiring all AI modules
├── api/
│   └── main.py                 # FastAPI app (events, MJPEG stream, screenshots)
├── dashboard/
│   └── app.py                  # Streamlit frontend
├── utils/
│   ├── config.py               # All configurable constants / thresholds / paths
│   ├── frame_buffer.py         # Thread-safe shared frame buffer for MJPEG streaming
│   └── logger.py               # Logging utility
├── data/
│   ├── known_faces/            # Place person images here (see README inside)
│   ├── logs/                   # system.log written here
│   ├── models/                 # Cached face encodings (.pkl)
│   └── screenshots/            # Alert screenshots saved here
├── main.py                     # Entry point — starts pipeline + embedded API server
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.10+
- A webcam **or** an IP camera / video file.
- (Optional) CUDA-capable GPU for real-time inference.

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Add known faces *(optional)*
```
data/known_faces/
├── John_Doe/
│   └── photo.jpg
└── Jane_Smith/
    └── photo.jpg
```
The system computes embeddings on startup and caches them in `data/models/face_encodings.pkl`.  
Delete that file to re-encode after adding new people.

### 3 — Run (all-in-one)
```bash
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/video.mp4

# IP camera
python main.py --source "rtsp://user:pass@192.168.1.100/stream"
```

`main.py` starts the FastAPI server (port 8000) in a background thread **and** the inference pipeline in the same process.  
This means the live MJPEG stream at `http://localhost:8000/stream/video` receives annotated frames directly from the pipeline.

### 4 — Open the Streamlit dashboard
In a **separate terminal**:
```bash
streamlit run dashboard/app.py
```
Browse to `http://localhost:8501`.

### 5 — Standalone API (advanced)
If you prefer to run the API separately (e.g., to serve multiple pipelines):
```bash
# Terminal 1 — API only
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Pipeline without embedded API
python main.py --no-api
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/events/` | Receive an alert event from the pipeline |
| GET | `/events/?limit=N` | Fetch the N most recent events |
| DELETE | `/events/` | Clear the event store |
| GET | `/stream/video` | MJPEG live video feed |
| GET | `/screenshots/` | List recent screenshot filenames |
| GET | `/screenshots/{filename}` | Serve a screenshot image |

Interactive API docs: `http://localhost:8000/docs`

## Docker Deployment
```bash
docker-compose up --build
```
*For webcam access inside Docker, uncomment the `devices` section in `docker-compose.yml`.*

## Configuration
All thresholds and paths live in `utils/config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_SOURCE` | `0` | Video source |
| `YOLO_MODEL_PATH` | `yolov8n.pt` | YOLO model file |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence |
| `FACE_RECOGNITION_TOLERANCE` | `0.5` | Face match threshold |
| `LOITERING_THRESHOLD_SECONDS` | `10` | Seconds before loitering alert |
| `API_PORT` | `8000` | FastAPI bind port |

## Performance Tips
- **GPU**: Ensure PyTorch is installed with CUDA support.
- **Frame Skipping**: Run face recognition every N frames for higher FPS.
- **Larger Models**: Switch `YOLO_MODEL_PATH` to `yolov8s.pt` or `yolov8m.pt` for better accuracy at the cost of speed.
- **TensorRT**: Export with `yolo export model=yolov8n.pt format=engine` for maximum GPU throughput.
- **INT8 Quantization**: Reduces model size for edge devices (Jetson Nano, Raspberry Pi).

---
*Built as a production-level demonstration of end-to-end AI System Architecture.*
