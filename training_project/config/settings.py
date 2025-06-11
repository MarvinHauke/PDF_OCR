# ===========================================
# File: training_project/config/settings.py (MPS Enhanced)
# ===========================================
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


class Config:
    """Configuration class for YOLO training with enhanced MPS support"""

    def __init__(self, config_file: Optional[str] = None):
        # Base paths
        self.PROJECT_ROOT = Path(__file__).parent.parent  # training_project/

        # Load configuration from YAML
        config_path = config_file or (self.PROJECT_ROOT / "config" / "config.yaml")
        self._load_config(config_path)

        # Set up computed paths
        self._setup_paths()

        # Apply MPS-specific optimizations
        self._optimize_for_mps()

    def _load_config(self, config_path: Path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Model configuration
            model_config = config_data.get("model", {})
            self.MODEL_ARCH = model_config.get("architecture", "yolo11n.pt")
            self.USE_BEST_WEIGHTS = model_config.get("use_best_weights", False)

            # Training configuration
            training_config = config_data.get("training", {})
            self.RUN_NAME = training_config.get("run_name", "run")
            self.IMGSZ = training_config.get("image_size", 640)
            self.BATCH_SIZE = training_config.get("batch_size", -1)
            self.EPOCHS = training_config.get("epochs", 100)
            self.WORKERS = training_config.get("workers", 0)
            self.DEVICE = training_config.get("device", "mps")
            self.PATIENCE = training_config.get("patience", 50)
            self.SAVE_PERIOD = training_config.get("save_period", 10)

            # MPS optimizations
            self.AMP = training_config.get("amp", True)
            self.HALF = training_config.get("half", False)
            self.CACHE = training_config.get("cache", False)
            self.MULTI_SCALE = training_config.get("multi_scale", False)
            self.DETERMINISTIC = training_config.get("deterministic", False)

            # Optimizer settings
            self.OPTIMIZER = training_config.get("optimizer", "AdamW")
            self.LR0 = training_config.get("lr0", 0.01)
            self.LRF = training_config.get("lrf", 0.01)
            self.MOMENTUM = training_config.get("momentum", 0.937)
            self.WEIGHT_DECAY = training_config.get("weight_decay", 0.0005)
            self.WARMUP_EPOCHS = training_config.get("warmup_epochs", 3.0)
            self.WARMUP_MOMENTUM = training_config.get("warmup_momentum", 0.8)
            self.WARMUP_BIAS_LR = training_config.get("warmup_bias_lr", 0.1)

            # Data augmentation
            self.HSV_H = training_config.get("hsv_h", 0.015)
            self.HSV_S = training_config.get("hsv_s", 0.7)
            self.HSV_V = training_config.get("hsv_v", 0.4)
            self.DEGREES = training_config.get("degrees", 0.0)
            self.TRANSLATE = training_config.get("translate", 0.1)
            self.SCALE = training_config.get("scale", 0.5)
            self.SHEAR = training_config.get("shear", 0.0)
            self.PERSPECTIVE = training_config.get("perspective", 0.0)
            self.FLIPUD = training_config.get("flipud", 0.0)
            self.FLIPLR = training_config.get("fliplr", 0.5)
            self.MOSAIC = training_config.get("mosaic", 1.0)
            self.MIXUP = training_config.get("mixup", 0.0)
            self.COPY_PASTE = training_config.get("copy_paste", 0.0)

            # MPS performance settings
            mps_config = config_data.get("mps", {})
            self.MAX_MEMORY_FRACTION = mps_config.get("max_memory_fraction", 0.8)
            self.EMPTY_CACHE_FREQUENCY = mps_config.get("empty_cache_frequency", 10)
            self.TORCH_COMPILE = mps_config.get("torch_compile", False)
            self.SYNC_BN = mps_config.get("sync_bn", False)
            self.USE_CPU_FALLBACK = mps_config.get("use_cpu_fallback", True)
            self.PIN_MEMORY = mps_config.get("pin_memory", False)
            self.PERSISTENT_WORKERS = mps_config.get("persistent_workers", False)

            # Prediction configuration
            prediction_config = config_data.get("prediction", {})
            self.CONFIDENCE_THRESHOLD = prediction_config.get(
                "confidence_threshold", 0.25
            )
            self.IOU_THRESHOLD = prediction_config.get("iou_threshold", 0.7)
            self.MAX_DETECTIONS = prediction_config.get("max_detections", 300)
            self.HALF_PRECISION = prediction_config.get("half_precision", False)
            self.AUGMENT = prediction_config.get("augment", False)
            self.AGNOSTIC_NMS = prediction_config.get("agnostic_nms", False)

            # Validation settings
            validation_config = config_data.get("validation", {})
            self.VAL_SPLIT = validation_config.get("split", "val")
            self.SAVE_JSON = validation_config.get("save_json", False)
            self.SAVE_HYBRID = validation_config.get("save_hybrid", False)
            self.SINGLE_CLS = validation_config.get("single_cls", False)
            self.RECT = validation_config.get("rect", False)

            # Logging settings
            logging_config = config_data.get("logging", {})
            self.VERBOSE = logging_config.get("verbose", True)
            self.PLOTS = logging_config.get("plots", True)
            self.TENSORBOARD = logging_config.get("tensorboard", False)
            self.CLEARML = logging_config.get("clearml", False)
            self.COMET = logging_config.get("comet", False)

            # Environment settings
            env_config = config_data.get("environment", {})
            self.PROFILE = env_config.get("profile", False)
            self.DEBUG = env_config.get("debug", False)
            self.SEED = env_config.get("seed", 0)

            # Path configuration
            paths_config = config_data.get("paths", {})
            self._training_data_rel = paths_config.get("training_data", "training_data")
            self._dataset_yaml_rel = paths_config.get(
                "dataset_yaml", "training_data/data.yaml"
            )
            self._project_rel = paths_config.get("project", "training_data/runs")
            self._output_rel = paths_config.get("output", "../output")

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _setup_paths(self):
        """Set up all paths based on configuration"""
        self.TRAINING_DATA_PATH = self.PROJECT_ROOT / self._training_data_rel
        self.YAML_PATH = self.PROJECT_ROOT / self._dataset_yaml_rel
        self.PROJECT_PATH = self.PROJECT_ROOT / self._project_rel
        self.OUTPUT_PATH = self.PROJECT_ROOT / self._output_rel

    def _optimize_for_mps(self):
        """Apply MPS-specific optimizations"""
        if self.DEVICE == "mps":
            # Check MPS availability
            if not torch.backends.mps.is_available():
                logging.warning("MPS not available, falling back to CPU")
                self.DEVICE = "cpu"
                return

            # MPS-specific optimizations
            if self.WORKERS > 0:
                logging.info("Setting workers=0 for MPS compatibility")
                self.WORKERS = 0

            if self.HALF:
                logging.info("Disabling half precision for MPS stability")
                self.HALF = False

            if self.PIN_MEMORY:
                logging.info("Disabling pin_memory for MPS")
                self.PIN_MEMORY = False

            # Set environment variables for MPS
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
                "1" if self.USE_CPU_FALLBACK else "0"
            )

            logging.info("Applied MPS optimizations")

    def get_training_params(self) -> Dict[str, Any]:
        """Get complete training parameters dictionary"""
        params = {
            # Basic training parameters
            "data": str(self.YAML_PATH),
            "imgsz": self.IMGSZ,
            "batch": self.BATCH_SIZE,
            "epochs": self.EPOCHS,
            "workers": self.WORKERS,
            "device": self.DEVICE,
            "patience": self.PATIENCE,
            "save_period": self.SAVE_PERIOD,
            "project": str(self.PROJECT_PATH),
            "name": self.RUN_NAME,
            "exist_ok": True,
            "verbose": self.VERBOSE,
            # MPS optimizations
            "amp": self.AMP,
            "half": self.HALF,
            "cache": self.CACHE,
            "multi_scale": self.MULTI_SCALE,
            "deterministic": self.DETERMINISTIC,
            # Optimizer settings
            "optimizer": self.OPTIMIZER,
            "lr0": self.LR0,
            "lrf": self.LRF,
            "momentum": self.MOMENTUM,
            "weight_decay": self.WEIGHT_DECAY,
            "warmup_epochs": self.WARMUP_EPOCHS,
            "warmup_momentum": self.WARMUP_MOMENTUM,
            "warmup_bias_lr": self.WARMUP_BIAS_LR,
            # Data augmentation
            "hsv_h": self.HSV_H,
            "hsv_s": self.HSV_S,
            "hsv_v": self.HSV_V,
            "degrees": self.DEGREES,
            "translate": self.TRANSLATE,
            "scale": self.SCALE,
            "shear": self.SHEAR,
            "perspective": self.PERSPECTIVE,
            "flipud": self.FLIPUD,
            "fliplr": self.FLIPLR,
            "mosaic": self.MOSAIC,
            "mixup": self.MIXUP,
            "copy_paste": self.COPY_PASTE,
            # Validation
            "val": True,
            "split": self.VAL_SPLIT,
            "save_json": self.SAVE_JSON,
            "save_hybrid": self.SAVE_HYBRID,
            "single_cls": self.SINGLE_CLS,
            "rect": self.RECT,
            # Logging
            "plots": self.PLOTS,
            "profile": self.PROFILE,
            "seed": self.SEED,
        }

        return params

    def update_from_args(self, **kwargs):
        """Update configuration from command line arguments or other sources"""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()) and value is not None:
                setattr(self, key.upper(), value)

        # Re-apply MPS optimizations after updates
        if hasattr(self, "DEVICE"):
            self._optimize_for_mps()

    def get_model_path(self):
        """Get the full path to the model architecture file"""
        return self.TRAINING_DATA_PATH / self.MODEL_ARCH

    def get_weights_path(self):
        """Get the path to model weights"""
        weights_name = "best.pt" if self.USE_BEST_WEIGHTS else "last.pt"
        return self.PROJECT_PATH / self.RUN_NAME / "weights" / weights_name

    def validate_paths(self):
        """Validate that required paths exist"""
        if not self.YAML_PATH.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.YAML_PATH}")

        if not self.TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Training data path not found: {self.TRAINING_DATA_PATH}"
            )

        # Check if model file exists
        model_path = self.get_model_path()
        if not model_path.exists():
            print(
                f"Warning: Model file {model_path} not found. YOLO will download it automatically."
            )

        return True

    def is_run_complete(self):
        """Check if the training run appears to be complete"""
        weights_path = self.get_weights_path()
        return weights_path.exists()

    def generate_run_name(self, prefix="run"):
        """Generate a new run name like run1, run2, ..."""
        self.PROJECT_PATH.mkdir(parents=True, exist_ok=True)
        existing_runs = [
            d.name
            for d in self.PROJECT_PATH.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        run_nums = [
            int(name[len(prefix) :])
            for name in existing_runs
            if name[len(prefix) :].isdigit()
        ]
        next_run = max(run_nums, default=0) + 1
        self.RUN_NAME = f"{prefix}{next_run}"
        return self.RUN_NAME

    def print_paths(self):
        """Print all configured paths for debugging"""
        print("=== Configuration Paths ===")
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Training Data: {self.TRAINING_DATA_PATH}")
        print(f"Dataset YAML: {self.YAML_PATH}")
        print(f"Model Architecture: {self.get_model_path()}")
        print(f"Weights Path: {self.get_weights_path()}")
        print(f"Project Path: {self.PROJECT_PATH}")
        print(f"Output Path: {self.OUTPUT_PATH}")
        print("=" * 30)

    def print_mps_info(self):
        """Print MPS configuration and status"""
        print("=== MPS Configuration ===")
        print(f"Device: {self.DEVICE}")
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        print(f"MPS Built: {torch.backends.mps.is_built()}")
        print(f"AMP Enabled: {self.AMP}")
        print(f"Workers: {self.WORKERS}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"CPU Fallback: {self.USE_CPU_FALLBACK}")
        print("=" * 30)

    def save_config(self, output_path: str = None):
        """Save current configuration to YAML file"""
        if output_path is None:
            output_path = self.PROJECT_PATH / self.RUN_NAME / "config.yaml"

        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "model": {
                "architecture": self.MODEL_ARCH,
                "use_best_weights": self.USE_BEST_WEIGHTS,
            },
            "training": {
                "run_name": self.RUN_NAME,
                "image_size": self.IMGSZ,
                "batch_size": self.BATCH_SIZE,
                "epochs": self.EPOCHS,
                "workers": self.WORKERS,
                "device": self.DEVICE,
                "patience": self.PATIENCE,
                "save_period": self.SAVE_PERIOD,
                "amp": self.AMP,
                "half": self.HALF,
                "cache": self.CACHE,
                "multi_scale": self.MULTI_SCALE,
                "deterministic": self.DETERMINISTIC,
                "optimizer": self.OPTIMIZER,
                "lr0": self.LR0,
                "lrf": self.LRF,
                "momentum": self.MOMENTUM,
                "weight_decay": self.WEIGHT_DECAY,
                "warmup_epochs": self.WARMUP_EPOCHS,
                "warmup_momentum": self.WARMUP_MOMENTUM,
                "warmup_bias_lr": self.WARMUP_BIAS_LR,
            },
            "mps": {
                "max_memory_fraction": self.MAX_MEMORY_FRACTION,
                "empty_cache_frequency": self.EMPTY_CACHE_FREQUENCY,
                "torch_compile": self.TORCH_COMPILE,
                "sync_bn": self.SYNC_BN,
                "use_cpu_fallback": self.USE_CPU_FALLBACK,
                "pin_memory": self.PIN_MEMORY,
                "persistent_workers": self.PERSISTENT_WORKERS,
            },
            "prediction": {
                "confidence_threshold": self.CONFIDENCE_THRESHOLD,
                "iou_threshold": self.IOU_THRESHOLD,
                "max_detections": self.MAX_DETECTIONS,
                "half_precision": self.HALF_PRECISION,
                "augment": self.AUGMENT,
                "agnostic_nms": self.AGNOSTIC_NMS,
            },
            "paths": {
                "training_data": self._training_data_rel,
                "dataset_yaml": self._dataset_yaml_rel,
                "project": self._project_rel,
                "output": self._output_rel,
            },
        }

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        print(f"Configuration saved to: {output_path}")


# Create a default instance for backward compatibility
default_config = Config()
