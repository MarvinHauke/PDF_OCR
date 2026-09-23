"""Run the training project's YOLO detector on page images.

training_project/ is not an installed package: its modules import each other
as `config.settings` / `src.*`, so it has to sit at the front of sys.path
(the same approach its own scripts/ use).
"""

import sys
from pathlib import Path

TRAINING_PROJECT = Path(__file__).resolve().parents[2] / "training_project"
if str(TRAINING_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAINING_PROJECT))

from config.settings import default_config  # noqa: E402
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


def training_paths() -> tuple[Path, Path]:
    """(sources, unlabeled) folders of the training project."""
    return default_config.SOURCES_PATH, default_config.UNLABELED_PATH
