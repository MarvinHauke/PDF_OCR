# ===========================================
# File: training_project/src/predictor.py (Updated)
# ===========================================
import logging
from pathlib import Path

from config.settings import default_config
from ultralytics import YOLO

from src.utils import setup_logging


class YOLOPredictor:
    """YOLO model predictor class"""

    def __init__(self, config=None, model_path=None):
        self.logger = setup_logging()
        self.config = config or default_config

        # Use custom model path or default weights path
        self.model_path = model_path or self.config.get_weights_path()

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"No trained model found at {self.model_path}")

    def predict(self, source, save_dir=None, conf=None, show=False, save=None):
        """Run prediction on source"""
        model = YOLO(str(self.model_path))

        # Default save directory to main output folder if not specified
        if save_dir is None:
            save_dir = self.config.OUTPUT_PATH

        # Live sources (e.g. a webcam) default to not saving every frame unless asked
        if save is None:
            save = not show

        predict_params = {
            "source": source,
            "save": save,
            "show": show,
            "conf": conf or self.config.CONFIDENCE_THRESHOLD,
            "project": str(save_dir),
            "name": "yolo_predictions",
        }

        self.logger.info(f"Running prediction on: {source}")
        if save:
            self.logger.info(f"Results will be saved to: {save_dir}/yolo_predictions")

        results = model.predict(**predict_params)
        self.logger.info("Prediction completed!")
        return results

    def predict_images_folder(self, source_dir=None, output_dir=None):
        """Run prediction on all images in a folder"""
        # Use the unlabeled training folder by default
        if source_dir is None:
            source_dir = self.config.UNLABELED_PATH

        if not Path(source_dir).exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        return self.predict(source_dir, save_dir=output_dir)
