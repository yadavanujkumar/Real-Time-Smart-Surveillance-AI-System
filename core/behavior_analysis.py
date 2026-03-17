import time
import cv2
import numpy as np
from collections import defaultdict
from ultralytics.utils.plotting import colors
from utils.logger import logger
from utils.config import LOITERING_THRESHOLD_SECONDS, RESTRICTED_AREAS

class BehaviorAnalyzer:
    """Analyzes trajectories and timestamps of tracked objects to flag suspicious activities."""
    def __init__(self):
        # Store {track_id: list of (timestamp, centroid_x, centroid_y)}
        self.track_history = defaultdict(list)
        # Store initial seen time for loitering detection: {track_id: timestamp}
        self.first_seen = {}
        
        # Restricted areas configuration
        # For simplicity, RESTRICTED_AREAS could be a list of lists of points:
        # e.g. [[[0,0], [100,0], [100,100], [0,100]]] 
        self.restricted_polygons = [np.array(poly, np.int32) for poly in RESTRICTED_AREAS]

    def process(self, frame, detections):
        """Processes detections to find loitering and restricted area entries."""
        current_time = time.time()
        events = []
        
        # Update trackers
        current_ids = set()
        
        for det in detections:
            track_id = det["track_id"]
            if track_id is None:
                continue
                
            current_ids.add(track_id)
            x1, y1, x2, y2 = det["bbox"]
            centroid = (int((x1 + x2) / 2), int(y2)) # Use bottom center for position
            
            # Keep history short (e.g. last 30 frames)
            self.track_history[track_id].append((current_time, centroid))
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
                
            if track_id not in self.first_seen:
                self.first_seen[track_id] = current_time
                
            # 1. Check Loitering
            time_present = current_time - self.first_seen[track_id]
            if time_present > LOITERING_THRESHOLD_SECONDS:
                events.append({
                    "type": "LOITERING",
                    "track_id": track_id,
                    "duration": time_present,
                    "bbox": det["bbox"]
                })
                # Highlight box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"LOITERING ({int(time_present)}s)", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 2. Check Restricted Area
            for poly in self.restricted_polygons:
                # pointPolygonTest returns >0 if inside, 0 if on edge, <0 if outside
                if cv2.pointPolygonTest(poly, centroid, False) >= 0:
                    events.append({
                        "type": "RESTRICTED_ACCESS",
                        "track_id": track_id,
                        "bbox": det["bbox"]
                    })
                    cv2.putText(frame, "RESTRICTED ACCESS ALERT!", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    break
        
        # Cleanup unseen IDs
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                # If they leave, clear them, or keep them longer if occlusion is possible
                # But ByteTrack handles occlusion well inside the model.
                del self.track_history[tid]
                if tid in self.first_seen:
                    del self.first_seen[tid]

        # Draw Restricted Polygons
        for poly in self.restricted_polygons:
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
            
        return events
