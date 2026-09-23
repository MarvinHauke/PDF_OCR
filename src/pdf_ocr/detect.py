"""Run the training project's YOLO detector on page images."""

from pathlib import Path

from pdf_ocr.training import dataset_config

from config.settings import default_config  # noqa: E402  (importable via pdf_ocr.training)
from src.predictor import YOLOPredictor  # noqa: E402


class Detector:
    def __init__(self, model_path=None, conf=None):
        self.predictor = YOLOPredictor(model_path=model_path)
        self.model_path = Path(self.predictor.model_path)
        self.conf = conf or default_config.CONFIDENCE_THRESHOLD

    def detect(self, image_paths: list[Path]):
        """One ultralytics Results per image, in the same order."""
        return self.predictor.predict(
            [str(p) for p in image_paths], save=False, conf=self.conf
        )


def training_paths(dataset: str = "pages") -> tuple[Path, Path]:
    """(sources, unlabeled) folders of a training dataset (pages or subcircuits)."""
    config = dataset_config(dataset)
    return config.SOURCES_PATH, config.UNLABELED_PATH
