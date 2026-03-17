import cv2
import numpy as np
from ultralytics import YOLO
from utils.logger import logger
from utils.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASSES_TO_DETECT, TRACKER_TYPE

class ObjectDetectorTracker:
    """Handles object detection and tracking using YOLOv8."""
    
    def __init__(self):
        logger.info(f"Loading YOLO model from {YOLO_MODEL_PATH}")
        self.model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded successfully.")

    def process_frame(self, frame):
        """
        Runs tracking on a single frame.
        Returns the annotated frame and the detections (boxes, classes, ids).
        """
        # Run YOLO tracking on the frame
        # persist=True ensures IDs are consistent across frames
        results = self.model.track(
            source=frame,
            persist=True,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=CLASSES_TO_DETECT,  # filter by class (e.g., person=0)
            tracker=TRACKER_TYPE,       # bytetrack or botsort
            verbose=False               # disable verbose logging per frame
        )

        detections = []
        annotated_frame = frame.copy()

        if results and len(results) > 0:
            result = results[0]
            # Use Ultralytics built-in plotter if desired, but we'll do custom or use theirs
            # annotated_frame = result.plot()
            
            # Extract tracking info
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # IDs might be None if no objects are tracked yet
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)

                for box, conf, cls, track_id in zip(boxes, confs, classes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                        "class_id": int(cls),
                        "track_id": int(track_id) if track_id is not None else None
                    }
                    detections.append(detection)
                    
                    # Draw custom bounding box to have control
                    label = f"ID: {detection['track_id']} C: {detection['confidence']:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_frame, detections
