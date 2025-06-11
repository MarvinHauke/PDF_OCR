# ===========================================
# File: training_project/src/trainer.py (MPS Enhanced)
# ===========================================
import gc
import logging
from pathlib import Path

import torch
from config.settings import Config, default_config
from ultralytics import YOLO

from src.utils import check_file_exists, setup_logging


class YOLOTrainer:
    """YOLO model trainer class with MPS optimizations"""

    def __init__(self, config=None):
        self.logger = setup_logging()
        # Now config is an instance, not a class
        self.config = config if config is not None else default_config
        self.config.validate_paths()

        # Print configuration info
        self.config.print_paths()
        if self.config.DEVICE == "mps":
            self.config.print_mps_info()

    def get_training_params(self):
        """Get training parameters dictionary"""
        return self.config.get_training_params()

    def _clear_mps_cache(self):
        """Clear MPS cache to free memory"""
        if self.config.DEVICE == "mps" and torch.backends.mps.is_available():
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            gc.collect()
            self.logger.info("Cleared MPS cache")

    def train(self, resume_if_possible=True):
        """Train the model with automatic resume detection and MPS optimizations"""
        try:
            # Clear cache before starting
            if self.config.DEVICE == "mps":
                self._clear_mps_cache()

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

            # Save current configuration for reproducibility
            run_config_path = (
                self.config.PROJECT_PATH / self.config.RUN_NAME / "config.yaml"
            )
            run_config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.save_config(run_config_path)

            # MPS-specific setup
            if self.config.DEVICE == "mps":
                self.logger.info("Applying MPS-specific optimizations...")

                # Set environment variables
                import os

                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
                    "1" if self.config.USE_CPU_FALLBACK else "0"
                )

                # Additional MPS warnings
                if training_params.get("workers", 0) > 0:
                    self.logger.warning(
                        "Workers > 0 may cause issues with MPS. Consider setting workers=0"
                    )

            # Custom training callback for MPS memory management
            def on_train_epoch_end(trainer):
                """Callback to manage MPS memory"""
                if self.config.DEVICE == "mps":
                    epoch = trainer.epoch
                    if epoch % self.config.EMPTY_CACHE_FREQUENCY == 0:
                        self._clear_mps_cache()
                        self.logger.info(f"Cleared MPS cache at epoch {epoch}")

            # Add callback if using MPS
            if self.config.DEVICE == "mps":
                model.add_callback("on_train_epoch_end", on_train_epoch_end)

            results = model.train(**training_params)

            # Final memory cleanup
            if self.config.DEVICE == "mps":
                self._clear_mps_cache()

            self.logger.info("Training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")

            # Try to clear cache on error
            if self.config.DEVICE == "mps":
                try:
                    self._clear_mps_cache()
                except:
                    pass

            raise
