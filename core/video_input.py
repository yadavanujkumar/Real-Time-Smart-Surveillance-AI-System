import cv2
import time
from utils.logger import logger

class VideoInput:
    """Handles video frame ingestion from webcams, IP cameras, or video files."""
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0

    def start(self):
        """Initializes the video capture."""
        logger.info(f"Initializing video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise ValueError(f"Unable to open video source: {self.source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video source started. Resolution: {self.width}x{self.height} @ {self.fps} FPS")

    def read_frame(self):
        """Reads a single frame from the video source."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame

    def stop(self):
        """Releases the video capture gracefully."""
        if self.cap:
            logger.info("Stopping video source.")
            self.cap.release()
            self.cap = None
