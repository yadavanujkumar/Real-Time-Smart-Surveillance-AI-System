import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Video Input
CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", 0))  # 0 for webcam; set env var for IP cameras

# Detection
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")  # default lightweight model
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.environ.get("IOU_THRESHOLD", 0.45))
CLASSES_TO_DETECT = [0]  # 0 is "person" in COCO dataset

# Tracking
TRACKER_TYPE = "bytetrack.yaml"  # or "botsort.yaml"

# Face Recognition
FACE_RECOGNITION_TOLERANCE = float(os.environ.get("FACE_RECOGNITION_TOLERANCE", 0.5))

# Behavior Analytics
LOITERING_THRESHOLD_SECONDS = int(os.environ.get("LOITERING_THRESHOLD_SECONDS", 10))
RESTRICTED_AREAS = [
    # List of polygons defining restricted zones, e.g. [[x1,y1], [x2,y2], ...]
    # Example: [[[0, 0], [200, 0], [200, 200], [0, 200]]]
]

# API / Streaming
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 8000))

# Screenshot settings
SAVE_SCREENSHOTS = True  # Set to False to disable screenshot captures
