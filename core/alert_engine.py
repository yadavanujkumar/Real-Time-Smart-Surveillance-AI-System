import threading
import requests
import datetime
from utils.logger import logger

class AlertEngine:
    """Handles asynchronous event logging to file and backend API."""
    def __init__(self, api_url="http://localhost:8000/events/"):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def log_event(self, event_type, track_id, details=""):
        """Logs an event and sends it to the API asynchronously."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Local log
        logger.warning(f"ALERT [{event_type}] ID: {track_id} - {details}")

        # Send to API in a background thread to avoid blocking inference
        payload = {
            "event_type": event_type,
            "track_id": track_id,
            "timestamp": timestamp,
            "details": details
        }
        
        thread = threading.Thread(target=self._send_to_api, args=(payload,))
        thread.daemon = True
        thread.start()

    def _send_to_api(self, payload):
        try:
            response = requests.post(self.api_url, json=payload, timeout=2.0)
            if response.status_code != 200:
                logger.error(f"Failed to send event to API: {response.status_code}")
        except Exception as e:
            logger.error(f"API Connection Error: {e}")
