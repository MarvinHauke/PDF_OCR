"""train.py resume: a crashed run must not count as complete (training_project/config/settings.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "training_project"))
from config.settings import Config  # noqa: E402


def state(tmp_path, checkpoint):
    last = tmp_path / "last.pt"
    if checkpoint is not None:
        torch.save(checkpoint, last)
    return Config.is_run_complete(SimpleNamespace(get_last_checkpoint_path=lambda: last))


def test_run_states(tmp_path):
    assert state(tmp_path / "a", None) is False                                   # never started
    (tmp_path / "b").mkdir(); (tmp_path / "c").mkdir()
    assert state(tmp_path / "b", {"epoch": 37, "optimizer": {"lr": 1}}) is False  # crashed -> resume
    assert state(tmp_path / "c", {"epoch": -1, "optimizer": None}) is True        # finished (stripped)
