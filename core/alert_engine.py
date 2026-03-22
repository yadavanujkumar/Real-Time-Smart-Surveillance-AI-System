import threading
import datetime
import requests
import cv2
import numpy as np
from utils.logger import logger
from utils.config import SCREENSHOTS_DIR, SAVE_SCREENSHOTS, API_PORT


class AlertEngine:
    """Handles asynchronous event logging to file and backend API, with optional screenshot saving."""

    def __init__(self, api_url: str | None = None):
        if api_url is None:
            api_url = f"http://localhost:{API_PORT}/events/"
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def log_event(self, event_type: str, track_id: int | None, details: str = "",
                  frame: np.ndarray | None = None) -> None:
        """Logs an event, optionally saves a screenshot, and sends to the API asynchronously."""
        timestamp = datetime.datetime.now().isoformat()

        logger.warning(f"ALERT [{event_type}] ID: {track_id} - {details}")

        screenshot_path = ""
        if SAVE_SCREENSHOTS and frame is not None:
            screenshot_path = self._save_screenshot(frame, event_type, track_id, timestamp)

        payload = {
            "event_type": event_type,
            "track_id": track_id if track_id is not None else -1,
            "timestamp": timestamp,
            "details": details,
            "screenshot_path": screenshot_path,
        }

        thread = threading.Thread(target=self._send_to_api, args=(payload,))
        thread.daemon = True
        thread.start()

    def _save_screenshot(self, frame: np.ndarray, event_type: str,
                         track_id: int | None, timestamp: str) -> str:
        """Saves the current frame as a JPEG screenshot and returns the file path."""
        try:
            safe_ts = timestamp.replace(":", "-").replace(".", "-")
            filename = f"{safe_ts}_{event_type}_id{track_id}.jpg"
            path = SCREENSHOTS_DIR / filename
            cv2.imwrite(str(path), frame)
            logger.info(f"Screenshot saved: {path}")
            return str(path)
        except Exception as exc:
            logger.error(f"Failed to save screenshot: {exc}")
            return ""

    def _send_to_api(self, payload: dict) -> None:
        try:
            response = requests.post(self.api_url, json=payload, timeout=2.0)
            if response.status_code not in (200, 201):
                logger.error(f"Failed to send event to API: {response.status_code}")
        except Exception as exc:
            logger.error(f"API Connection Error: {exc}")
