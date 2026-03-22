FROM python:3.10-slim

# Install system dependencies for OpenCV, dlib, and libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run the inference pipeline (which also starts the FastAPI server in a background thread)
# and run the Streamlit dashboard side-by-side.
CMD ["sh", "-c", "python main.py --source ${CAMERA_SOURCE:-0} & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
