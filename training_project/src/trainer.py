# ===========================================
# File: training_project/src/trainer.py (Updated)
# ===========================================
import logging
from pathlib import Path

from config.settings import Config
from ultralytics import YOLO

from src.utils import check_file_exists, setup_logging


class YOLOTrainer:
    """YOLO model trainer class"""

    def __init__(self, config=None):
        self.logger = setup_logging()
        self.config = config or Config
        self.config.validate_paths()

        # Print paths for debugging
        self.config.print_paths()

    def get_training_params(self):
        """Get training parameters dictionary"""
        return {
            "data": str(self.config.YAML_PATH),
            "imgsz": self.config.IMGSZ,
            "batch": self.config.BATCH_SIZE,
            "epochs": self.config.EPOCHS,
            "workers": self.config.WORKERS,
            "device": self.config.DEVICE,
            "patience": self.config.PATIENCE,
            "save_period": self.config.SAVE_PERIOD,
            "project": str(self.config.PROJECT_PATH),
            "name": self.config.RUN_NAME,
            "exist_ok": True,
            "verbose": True,
        }

    def train(self, resume_if_possible=True):
        """Train the model with automatic resume detection"""
        try:
            if self.config.is_run_complete():
                self.logger.info(
                    f"Run '{self.config.RUN_NAME}' already completed. Skipping training."
                )
                return None

            weights_path = self.config.get_weights_path()

            if resume_if_possible and weights_path.exists():
                self.logger.info(f"Found checkpoint at {weights_path}")
                self.logger.info("Resuming training from checkpoint...")

                model = YOLO(str(weights_path))
                training_params = self.get_training_params()
                training_params["resume"] = True

            else:
                self.logger.info("Starting training from scratch...")
                model_path = self.config.get_model_path()

                if check_file_exists(model_path, "Pretrained model"):
                    model = YOLO(str(model_path))
                else:
                    self.logger.info("YOLO will download the model automatically")
                    model = YOLO(self.config.MODEL_ARCH)

                training_params = self.get_training_params()

            results = model.train(**training_params)
            self.logger.info("Training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
