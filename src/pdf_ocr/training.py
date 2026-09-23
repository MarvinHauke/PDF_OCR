"""Access to the training project's configs from the application package.

training_project/ is not an installed package: its modules import each other
as `config.*` / `src.*`, so it has to sit at the front of sys.path (the same
approach its own scripts/ use).
"""

import sys
from pathlib import Path

TRAINING_PROJECT = Path(__file__).resolve().parents[2] / "training_project"
if str(TRAINING_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAINING_PROJECT))

from config.settings import Config  # noqa: E402

DATASET_CONFIGS = {"pages": None, "subcircuits": "config/subcircuits.yaml"}


def dataset_config(dataset: str = "pages") -> Config:
    config_file = DATASET_CONFIGS[dataset]
    return Config(config_file=config_file) if config_file else Config()
