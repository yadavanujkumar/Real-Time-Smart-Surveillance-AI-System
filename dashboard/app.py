import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(page_title="Smart Surveillance Dashboard", layout="wide")

API_URL = "http://localhost:8000/events/"

st.title("🛡️ Real-Time Smart Surveillance Dashboard")
st.markdown("Monitor security alerts and live status.")

def fetch_events():
    try:
        response = requests.get(API_URL, params={"limit": 50})
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return []

placeholder = st.empty()

while True:
    events = fetch_events()
    
    with placeholder.container():
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("System Status", "Online" if events is not None else "Offline", "Active")
        col2.metric("Recent Alerts", len(events))
        col3.metric("Critical Warnings", sum(1 for e in events if e.get("event_type") == "RESTRICTED_ACCESS"))
        
        st.subheader("Event Log")
        if events:
            df = pd.DataFrame(events)
            # Reorder columns for better UI
            df = df[["timestamp", "event_type", "track_id", "details"]]
            # Format timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Apply styling
            def color_row(row):
                if row["event_type"] == "RESTRICTED_ACCESS":
                    return ["background-color: #ffcccc"] * len(row)
                elif row["event_type"] == "LOITERING":
                    return ["background-color: #fff3cc"] * len(row)
                return [""] * len(row)

            st.dataframe(df.style.apply(color_row, axis=1), use_container_width=True, hide_index=True)
        else:
            st.info("No events recorded yet or API is down.")
            
    # Refresh every 2 seconds
    time.sleep(2)
