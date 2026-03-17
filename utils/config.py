import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Video Input
CAMERA_SOURCE = 0 # Use 0 for webcam, or provide a string URL/path for IP camera/video file

# Detection
YOLO_MODEL_PATH = "yolov8n.pt" # Consider yolov8s.pt or yolov8m.pt for better accuracy
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
CLASSES_TO_DETECT = [0] # 0 is person in COCO dataset

# Tracking
TRACKER_TYPE = "bytetrack.yaml" # or "botsort.yaml"

# Behavior Analytics (Examples)
LOITERING_THRESHOLD_SECONDS = 10
RESTRICTED_AREAS = [
    # List of polygons defining restricted zones, e.g. [[x1,y1], [x2,y2], ...]
]
