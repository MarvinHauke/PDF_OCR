# ===========================================
# File: training_project/config/settings.py
# ===========================================
import os
from pathlib import Path


class Config:
    """Configuration class for YOLO training"""

    # Base paths - adjusted for your structure
    PROJECT_ROOT = Path(__file__).parent.parent  # training_project/
    print(PROJECT_ROOT)
    TRAINING_DATA_PATH = PROJECT_ROOT / "training_data"

    # Dataset configuration
    DATASET_PATH = TRAINING_DATA_PATH
    YAML_PATH = DATASET_PATH / "data.yaml"

    # Model configuration - these are in your training_data folder
    MODEL_ARCH = "yolo11n.pt"  # Will look for yolo11n.pt in training_data/
    USE_BEST_WEIGHTS = False  # Set to True to use best.pt instead of last.pt

    # Training configuration
    RUN_NAME = "run"
    PROJECT_PATH = TRAINING_DATA_PATH / "runs"  # This matches your existing runs folder
    IMGSZ = 640
    BATCH_SIZE = 16
    EPOCHS = 100
    WORKERS = 8
    DEVICE = "mps"  # or "cuda", "cpu"
    PATIENCE = 50
    SAVE_PERIOD = 10

    # Prediction configuration
    CONFIDENCE_THRESHOLD = 0.25

    # Output paths for predictions
    OUTPUT_PATH = PROJECT_ROOT.parent / "output"  # Uses the main output folder

    @classmethod
    def get_model_path(cls):
        """Get the full path to the model architecture file"""
        return cls.TRAINING_DATA_PATH / cls.MODEL_ARCH

    @classmethod
    def get_weights_path(cls):
        """Get the path to model weights"""
        weights_name = "best.pt" if cls.USE_BEST_WEIGHTS else "last.pt"
        return cls.PROJECT_PATH / cls.RUN_NAME / "weights" / weights_name

    @classmethod
    def validate_paths(cls):
        """Validate that required paths exist"""
        if not cls.YAML_PATH.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {cls.YAML_PATH}")

        if not cls.TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training data path not found: {cls.TRAINING_DATA_PATH}"
            )

        # Check if model file exists
        model_path = cls.get_model_path()
        if not model_path.exists():
            print(
                f"Warning: Model file {model_path} not found. YOLO will download it automatically."
            )

        return True

    @classmethod
    def is_run_complete(cls):
        """Check if the training run appears to be complete (based on weight file presence)"""
        weights_path = cls.get_weights_path()
        return weights_path.exists()

    @classmethod
    def generate_run_name(cls, prefix="run"):
        """Generate a new run name like run1, run2, ..."""
        cls.PROJECT_PATH.mkdir(parents=True, exist_ok=True)
        existing_runs = [
            d.name
            for d in cls.PROJECT_PATH.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        run_nums = [
            int(name[len(prefix) :])
            for name in existing_runs
            if name[len(prefix) :].isdigit()
        ]
        next_run = max(run_nums, default=0) + 1
        cls.RUN_NAME = f"{prefix}{next_run}"
        return cls.RUN_NAME

    @classmethod
    def print_paths(cls):
        """Print all configured paths for debugging"""
        print("=== Configuration Paths ===")
        print(f"Project Root: {cls.PROJECT_ROOT}")
        print(f"Training Data: {cls.TRAINING_DATA_PATH}")
        print(f"Dataset YAML: {cls.YAML_PATH}")
        print(f"Model Architecture: {cls.get_model_path()}")
        print(f"Weights Path: {cls.get_weights_path()}")
        print(f"Project Path: {cls.PROJECT_PATH}")
        print(f"Output Path: {cls.OUTPUT_PATH}")
        print("=" * 30)
