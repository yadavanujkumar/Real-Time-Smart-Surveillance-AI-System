import threading
import cv2
import numpy as np

class FrameBuffer:
    """Thread-safe buffer that stores the latest annotated frame for MJPEG streaming."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_bytes: bytes = b""

    def update(self, frame: np.ndarray) -> None:
        """Encode a BGR frame as JPEG and store it."""
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            with self._lock:
                self._frame_bytes = buffer.tobytes()

    def get(self) -> bytes:
        """Return the latest JPEG-encoded frame bytes."""
        with self._lock:
            return self._frame_bytes


# Module-level singleton — imported by both the inference pipeline and the API
frame_buffer = FrameBuffer()
