import logging
import sys
from pathlib import Path

def setup_logger(name="SmartSurveillance"):
    """Configures and returns a logger for the system."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        
        # File handler
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(log_dir / "system.log")
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger

logger = setup_logger()
