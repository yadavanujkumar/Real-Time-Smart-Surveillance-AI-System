import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Smart Surveillance Dashboard", layout="wide", page_icon="🛡️")

API_BASE = "http://localhost:8000"
STREAM_URL = f"{API_BASE}/stream/video"


def fetch_events(limit: int = 50) -> list:
    try:
        response = requests.get(f"{API_BASE}/events/", params={"limit": limit}, timeout=2)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def api_online() -> bool:
    try:
        return requests.get(f"{API_BASE}/", timeout=1).status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🛡️ Real-Time Smart Surveillance Dashboard")
st.caption("Powered by YOLOv8 · ByteTrack · FaceNet · FastAPI")

st.markdown("---")

# Two main columns: video feed (left) + metrics (right)
col_video, col_right = st.columns([3, 2], gap="large")

with col_video:
    st.subheader("📹 Live Video Feed")
    st.markdown(
        f'<img src="{STREAM_URL}" width="100%" style="border-radius:8px; border:1px solid #333;" '
        f'onerror="this.style.display=\'none\'">',
        unsafe_allow_html=True,
    )
    st.caption("Stream served from the inference pipeline via MJPEG. Start the pipeline to see the feed.")

with col_right:
    status_placeholder = st.empty()
    metric_placeholder = st.empty()

st.markdown("---")

# Event log section
log_placeholder = st.empty()
screenshots_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Auto-refresh loop
# ---------------------------------------------------------------------------

while True:
    online = api_online()
    events = fetch_events()

    # Status + metrics
    with status_placeholder.container():
        st.subheader("📊 System Status")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("API Status", "🟢 Online" if online else "🔴 Offline")
        c2.metric("Total Alerts", len(events))
        face_matches = sum(1 for e in events if e.get("event_type") == "FACE_MATCH")
        restricted = sum(1 for e in events if e.get("event_type") == "RESTRICTED_ACCESS")
        c3.metric("Face Matches", face_matches)
        c4.metric("Restricted Access", restricted)

    # Event log
    with log_placeholder.container():
        st.subheader("📋 Detection & Alert Log")
        if events:
            df = pd.DataFrame(events)
            # Ensure all expected columns exist
            for col in ["timestamp", "event_type", "track_id", "details", "screenshot_path"]:
                if col not in df.columns:
                    df[col] = ""

            df = df[["timestamp", "event_type", "track_id", "details", "screenshot_path"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            df = df.sort_values("timestamp", ascending=False)

            def color_row(row):
                if row["event_type"] == "RESTRICTED_ACCESS":
                    return ["background-color: #ffcccc"] * len(row)
                if row["event_type"] == "LOITERING":
                    return ["background-color: #fff3cc"] * len(row)
                if row["event_type"] == "FACE_MATCH":
                    return ["background-color: #ccffcc"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(color_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No events recorded yet. Start the inference pipeline to begin monitoring.")

    # Recent screenshots
    with screenshots_placeholder.container():
        st.subheader("📷 Recent Screenshots")
        try:
            resp = requests.get(f"{API_BASE}/screenshots/", params={"limit": 6}, timeout=2)
            if resp.status_code == 200:
                filenames = resp.json()
                if filenames:
                    img_cols = st.columns(min(len(filenames), 3))
                    for idx, fname in enumerate(filenames[:6]):
                        with img_cols[idx % 3]:
                            st.image(f"{API_BASE}/screenshots/{fname}", caption=fname, use_container_width=True)
                else:
                    st.caption("No screenshots saved yet.")
            else:
                st.caption("Screenshots endpoint unavailable.")
        except Exception:
            st.caption("Could not fetch screenshots.")

    time.sleep(2)
