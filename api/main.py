import os
import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from utils.frame_buffer import frame_buffer
from utils.config import SCREENSHOTS_DIR

app = FastAPI(title="Smart Surveillance API", version="2.0")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class EventLog(BaseModel):
    event_type: str
    track_id: int
    timestamp: str
    details: str
    screenshot_path: Optional[str] = ""


# In-memory event store (last 200 events)
recent_events: list[EventLog] = []


# ---------------------------------------------------------------------------
# Health / Root
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Smart Surveillance System API v2", "status": "running"}


# ---------------------------------------------------------------------------
# Event endpoints
# ---------------------------------------------------------------------------

@app.post("/events/", status_code=201)
def log_event(event: EventLog):
    """Receive a new event from the inference pipeline."""
    recent_events.append(event)
    if len(recent_events) > 200:
        recent_events.pop(0)
    return {"status": "success", "event": event}


@app.get("/events/")
def get_recent_events(limit: int = 50):
    """Fetch the most recent events for the dashboard."""
    return recent_events[-limit:]


@app.delete("/events/")
def clear_events():
    """Clear all stored events (useful for testing)."""
    recent_events.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Live video streaming (MJPEG)
# ---------------------------------------------------------------------------

def _mjpeg_generator():
    """Yields MJPEG frames from the shared frame buffer."""
    boundary = b"frame"
    while True:
        jpg_bytes = frame_buffer.get()
        if not jpg_bytes:
            # Send a placeholder frame if no live feed is available yet
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video feed...", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            _, buf = cv2.imencode(".jpg", blank)
            jpg_bytes = buf.tobytes()

        yield (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )


@app.get("/stream/video")
def video_feed():
    """MJPEG video stream endpoint — embed as <img src='/stream/video'> in HTML."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Screenshots
# ---------------------------------------------------------------------------

@app.get("/screenshots/")
def list_screenshots(limit: int = 20):
    """Return a list of the most recent screenshot filenames."""
    files = sorted(SCREENSHOTS_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    return [f.name for f in files[:limit]]


@app.get("/screenshots/{filename}")
def get_screenshot(filename: str):
    """Serve a screenshot image file."""
    # Prevent path traversal
    safe_name = Path(filename).name
    path = SCREENSHOTS_DIR / safe_name
    if not path.exists():
        return Response(content="Not found", status_code=404)
    return FileResponse(str(path), media_type="image/jpeg")
