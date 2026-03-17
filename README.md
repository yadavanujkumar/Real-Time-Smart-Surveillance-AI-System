# Real-Time Smart Surveillance AI System

A portfolio-grade AI/ML project demonstrating computer vision, deep learning, real-time tracking, and behavior analysis.

## Features
- **Real-Time Human Detection**: Uses Ultralytics YOLOv8 for fast and accurate person detection.
- **Multi-Object Tracking**: Uses ByteTrack for assigning unique IDs to moving individuals.
- **Face Recognition**: Detects and identifies known individuals using facial embeddings (powered by `face_recognition`).
- **Behavior Analysis**: Detects loitering and unauthorized entry into restricted zones.
- **FastAPI Backend & Event Logging**: Asynchronous alert logging in a robust backend.
- **Streamlit Dashboard**: A dashboard to monitor live security alerts and system status.

## Folder Structure
```text
smart-surveillance-system/
├── core/
│   ├── video_input.py          # Frame ingestion
│   ├── detection.py            # YOLOv8 integration
│   ├── face_recognition.py     # Embeddings and matching
│   ├── behavior_analysis.py    # Loitering and zone rules
│   └── alert_engine.py         # Asynchronous logging via API
├── pipeline/
│   └── inference.py            # Main inference loop using all AI modules
├── api/
│   └── main.py                 # FastAPI app
├── dashboard/
│   └── app.py                  # Streamlit frontend
├── utils/
│   ├── config.py               # Constants, thresholds, zones
│   └── logger.py               # Logging utility
├── data/                       # Models, faces, logs
├── main.py                     # Entry point to run pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model Setup & Datasets
1. **YOLOv8**: The system automatically downloads `yolov8n.pt` on the first run.
2. **Face Recognition**: Place images of known individuals in `data/known_faces/PersonName/image.jpg`. The system will compute embeddings on start.
3. **Tracking**: `bytetrack.yaml` is provided out-of-the-box by the Ultralytics package.
4. **Dataset Recommendations**: For fine-tuning YOLOv8 for specific environments, consider datasets like MS COCO, CrowdHuman, or MOT17/MOT20.

## Running Locally

### Prerequisites
- Python 3.10+
- A working webcam or an IP camera stream URL.

### Installation
```bash
pip install -r requirements.txt
```

### 1. Start the API and Dashboard
```bash
# Start FastAPI (Terminal 1)
uvicorn api.main:app --reload --port 8000

# Start Streamlit (Terminal 2)
streamlit run dashboard/app.py
```

### 2. Run the AI Pipeline
```bash
# Start Inference Pipeline (Terminal 3)
# Uses webcam (source 0) by default. To use a video file or IP camera:
# python main.py --source "path/to/video.mp4"
python main.py
```

## Docker Deployment

To deploy using Docker (supports GPU via Nvidia Container Toolkit):

```bash
docker-compose up --build
```
*Note: Ensure your web camera is accessible to Docker, or use an IP camera URL in `docker-compose.yml` (`CAMERA_SOURCE`).*

## Performance Optimization Strategies
- **GPU Acceleration**: Always ensure PyTorch and OpenCV are compiled with CUDA support for real-time inference.
- **Frame Skipping**: If FPS drops, process behavior analytics or face recognition every $N$ frames instead of every single frame.
- **TensorRT**: Convert YOLOv8 to TensorRT engine (`.engine`) for maximum inference speed on NVIDIA GPUs.
- **Model Quantization**: Use INT8 optimization to reduce model footprint and improve inference speed on edge devices (e.g., Jetson Nano).
- **Asynchronous Processing**: The Alert Engine already operates asynchronously to prevent HTTP requests from blocking frame processing. Complex tasks like face recognition can similarly be offloaded to worker threads.

## Future Extensions
- **Weapon Detection**: Add a classification or secondary detection model to identify firearms or knives.
- **Pose Estimation**: Incorporate YOLO-Pose to recognize physical actions (falling, fighting).
- **Video Archiving**: Automatically record and compress video snippets corresponding to alert timestamps.

---
*Built as a production-level demonstration of end-to-end AI System Architecture.*
