from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import datetime

app = FastAPI(title="Smart Surveillance API", version="1.0")

class EventLog(BaseModel):
    event_type: str
    track_id: int
    timestamp: str
    details: str

# In-memory storage for demonstration purposes
recent_events = []

@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Surveillance System API"}

@app.post("/events/")
def log_event(event: EventLog):
    """Log a new event from the inference pipeline."""
    recent_events.append(event)
    # Keep only the last 100 events in memory
    if len(recent_events) > 100:
        recent_events.pop(0)
    return {"status": "success", "event": event}

@app.get("/events/")
def get_recent_events(limit: int = 20):
    """Fetch the most recent events for the dashboard."""
    return recent_events[-limit:]
