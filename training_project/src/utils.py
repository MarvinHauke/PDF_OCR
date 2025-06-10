# ===========================================
# File: src/utils.py
# ===========================================
import logging
from pathlib import Path


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def check_file_exists(file_path, description="File"):
    """Check if file exists and log appropriate message"""
    logger = logging.getLogger(__name__)
    if Path(file_path).exists():
        logger.info(f"{description} found: {file_path}")
        return True
    else:
        logger.warning(f"{description} not found: {file_path}")
        return False
