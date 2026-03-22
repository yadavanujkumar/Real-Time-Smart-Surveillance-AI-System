import argparse
import threading
import uvicorn

from pipeline.inference import InferencePipeline
from utils.logger import logger
from utils.config import CAMERA_SOURCE, API_HOST, API_PORT


def _start_api_server(host: str, port: int) -> None:
    """Launch the FastAPI server in a background thread."""
    uvicorn.run("api.main:app", host=host, port=port, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Smart Surveillance AI System")
    parser.add_argument(
        "--source",
        default=CAMERA_SOURCE,
        help="Video source (0 for webcam, path to video file, or IP stream URL)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable the built-in FastAPI server (use when running API separately)",
    )
    parser.add_argument(
        "--api-host", default=API_HOST, help="FastAPI bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--api-port", type=int, default=API_PORT, help="FastAPI bind port (default: 8000)"
    )
    args = parser.parse_args()

    # Convert source to int if it's a digit (webcam index)
    source = int(args.source) if str(args.source).isdigit() else args.source

    if not args.no_api:
        logger.info(f"Starting FastAPI server on {args.api_host}:{args.api_port} ...")
        api_thread = threading.Thread(
            target=_start_api_server, args=(args.api_host, args.api_port), daemon=True
        )
        api_thread.start()

    logger.info(f"Initializing inference pipeline with source: {source}")
    pipeline = InferencePipeline(source=source)
    pipeline.run()


if __name__ == "__main__":
    main()
