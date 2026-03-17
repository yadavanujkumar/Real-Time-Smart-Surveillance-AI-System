import argparse
from pipeline.inference import InferencePipeline
from utils.logger import logger
from utils.config import CAMERA_SOURCE

def main():
    parser = argparse.ArgumentParser(description="Real-Time Smart Surveillance AI System")
    parser.add_argument("--source", default=CAMERA_SOURCE, help="Video source (0 for webcam, path to video file, or IP stream URL)")
    args = parser.parse_args()

    # Convert source to int if it's a digit (for webcam)
    source = int(args.source) if str(args.source).isdigit() else args.source

    logger.info(f"Initializing system with source: {source}")
    pipeline = InferencePipeline(source=source)
    pipeline.run()

if __name__ == "__main__":
    main()
