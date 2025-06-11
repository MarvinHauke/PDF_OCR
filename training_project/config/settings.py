# ===========================================
# File: training_project/config/settings.py
# ===========================================
import os
from pathlib import Path


class Config:
    """Configuration class for YOLO training with stability optimizations"""

    # Base paths - adjusted for your structure
    PROJECT_ROOT = Path(__file__).parent.parent  # training_project/
    print(PROJECT_ROOT)
    TRAINING_DATA_PATH = PROJECT_ROOT / "training_data"

    # Dataset configuration
    DATASET_PATH = TRAINING_DATA_PATH
    YAML_PATH = DATASET_PATH / "data.yaml"

    # Model configuration - these are in your training_data folder
    MODEL_ARCH = "yolo11n.pt"  # Using nano for speed
    USE_BEST_WEIGHTS = False  # Set to True to use best.pt instead of last.pt

    # Training configuration - optimized for stability
    RUN_NAME = "run"
    PROJECT_PATH = TRAINING_DATA_PATH / "runs"  # This matches your existing runs folder

    # Core training parameters
    IMGSZ = 320  # Reduced from 640 for speed
    BATCH_SIZE = 32  # Good balance for M4
    EPOCHS = 25  # Reduced to avoid instability
    WORKERS = 8  # Matches your M4 cores
    DEVICE = "mps"  # M4 Metal Performance Shaders

    # Learning rate and optimization (stability focused)
    LR0 = 0.0005  # Reduced from 0.001 for stability
    LRF = 0.1  # Final learning rate factor
    COS_LR = True  # Enable cosine learning rate scheduling
    WARMUP_EPOCHS = 5.0  # Longer warmup for stability
    WARMUP_BIAS_LR = 0.1  # Warmup bias learning rate
    WARMUP_MOMENTUM = 0.8  # Warmup momentum

    # Regularization parameters
    WEIGHT_DECAY = 0.001  # Weight decay for regularization
    DROPOUT = 0.1  # Dropout rate
    MOMENTUM = 0.937  # SGD momentum

    # Early stopping and monitoring
    PATIENCE = 5  # Reduced from 50 for quicker stops
    SAVE_PERIOD = 5  # Save checkpoints more frequently
    VAL = True  # Run validation
    PLOTS = True  # Generate training plots

    # Speed optimizations
    CACHE = True  # Cache dataset in memory
    RECT = True  # Rectangular training
    AMP = True  # Automatic mixed precision

    # Augmentation settings (reduced for speed)
    MOSAIC = 0.0  # Disabled mosaic (expensive)
    MIXUP = 0.0  # Disabled mixup
    COPY_PASTE = 0.0  # Disabled copy-paste
    AUGMENT = False  # Disable auto-augment for speed

    # Data augmentation (lighter settings)
    HSV_H = 0.015  # Hue augmentation
    HSV_S = 0.7  # Saturation augmentation
    HSV_V = 0.4  # Value augmentation
    DEGREES = 0.0  # Rotation (disabled for speed)
    TRANSLATE = 0.1  # Translation
    SCALE = 0.5  # Scale augmentation
    SHEAR = 0.0  # Shear (disabled for speed)
    PERSPECTIVE = 0.0  # Perspective (disabled for speed)
    FLIPLR = 0.5  # Horizontal flip probability
    FLIPUD = 0.0  # Vertical flip (disabled)

    # Loss function weights
    BOX = 7.5  # Box loss weight
    CLS = 0.5  # Classification loss weight
    DFL = 1.5  # Distribution focal loss weight

    # Confidence and NMS settings
    CONFIDENCE_THRESHOLD = 0.25
    IOU = 0.7  # IoU threshold for NMS
    MAX_DET = 300  # Maximum detections

    # Advanced settings
    CLOSE_MOSAIC = 10  # Epochs to close mosaic augmentation
    DETERMINISTIC = True  # Deterministic training
    SINGLE_CLS = False  # Single class training
    MULTI_SCALE = False  # Multi-scale training (disabled for speed)

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
    def get_training_params(cls):
        """Get all training parameters as a dictionary for YOLO"""
        return {
            # Core parameters
            "epochs": cls.EPOCHS,
            "imgsz": cls.IMGSZ,
            "batch": cls.BATCH_SIZE,
            "workers": cls.WORKERS,
            "device": cls.DEVICE,
            # Learning rate and optimization
            "lr0": cls.LR0,
            "lrf": cls.LRF,
            "cos_lr": cls.COS_LR,
            "warmup_epochs": cls.WARMUP_EPOCHS,
            "warmup_bias_lr": cls.WARMUP_BIAS_LR,
            "warmup_momentum": cls.WARMUP_MOMENTUM,
            "weight_decay": cls.WEIGHT_DECAY,
            "momentum": cls.MOMENTUM,
            "dropout": cls.DROPOUT,
            # Monitoring and saving
            "patience": cls.PATIENCE,
            "save_period": cls.SAVE_PERIOD,
            "val": cls.VAL,
            "plots": cls.PLOTS,
            # Speed optimizations
            "cache": cls.CACHE,
            "rect": cls.RECT,
            "amp": cls.AMP,
            # Augmentation
            "mosaic": cls.MOSAIC,
            "mixup": cls.MIXUP,
            "copy_paste": cls.COPY_PASTE,
            "augment": cls.AUGMENT,
            "hsv_h": cls.HSV_H,
            "hsv_s": cls.HSV_S,
            "hsv_v": cls.HSV_V,
            "degrees": cls.DEGREES,
            "translate": cls.TRANSLATE,
            "scale": cls.SCALE,
            "shear": cls.SHEAR,
            "perspective": cls.PERSPECTIVE,
            "fliplr": cls.FLIPLR,
            "flipud": cls.FLIPUD,
            # Loss weights
            "box": cls.BOX,
            "cls": cls.CLS,
            "dfl": cls.DFL,
            # NMS settings
            "iou": cls.IOU,
            "max_det": cls.MAX_DET,
            # Advanced
            "close_mosaic": cls.CLOSE_MOSAIC,
            "deterministic": cls.DETERMINISTIC,
            "single_cls": cls.SINGLE_CLS,
            "multi_scale": cls.MULTI_SCALE,
        }

    @classmethod
    def print_config_summary(cls):
        """Print a summary of the current configuration"""
        print("=" * 60)
        print("🔧 YOLO TRAINING CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"📁 Project Root: {cls.PROJECT_ROOT}")
        print(f"📊 Dataset: {cls.YAML_PATH}")
        print(f"🤖 Model: {cls.MODEL_ARCH}")
        print(f"💾 Device: {cls.DEVICE}")
        print()
        print("🏋️  Training Parameters:")
        print(f"   • Image Size: {cls.IMGSZ}px")
        print(f"   • Batch Size: {cls.BATCH_SIZE}")
        print(f"   • Epochs: {cls.EPOCHS}")
        print(f"   • Learning Rate: {cls.LR0}")
        print(f"   • Patience: {cls.PATIENCE}")
        print()
        print("⚡ Speed Optimizations:")
        print(f"   • Cache: {cls.CACHE}")
        print(f"   • Rectangular Training: {cls.RECT}")
        print(f"   • Mixed Precision: {cls.AMP}")
        print(f"   • Mosaic: {cls.MOSAIC}")
        print()
        print("🎯 Stability Features:")
        print(f"   • Cosine LR: {cls.COS_LR}")
        print(f"   • Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"   • Dropout: {cls.DROPOUT}")
        print(f"   • Warmup Epochs: {cls.WARMUP_EPOCHS}")
        print("=" * 60)

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
