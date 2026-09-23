#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Rename a class in trained YOLO weights without retraining.

Label .txt files only store class ids, so a class name lives only in
data.yaml and in the `.names` of the model objects saved inside each
checkpoint. This rewrites those names; the learned weights are untouched.
A backup of each file is written to <name>.pt.bak first.
"""

import argparse
import shutil
import sys
from pathlib import Path

import argcomplete
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import default_config


def rename_in_checkpoint(path: Path, old: str, new: str) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    changed = False
    for key in ("model", "ema"):
        model = ckpt.get(key)
        if model is None or not hasattr(model, "names"):
            continue
        names = dict(model.names)
        for cls_id, name in names.items():
            if name == old:
                names[cls_id] = new
                changed = True
        model.names = names

    if changed:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        torch.save(ckpt, path)
    return changed


def main():
    parser = argparse.ArgumentParser(description="Rename a class in trained YOLO weights")
    parser.add_argument("old", help="Current class name, e.g. schemtaic")
    parser.add_argument("new", help="New class name, e.g. schematic")
    parser.add_argument(
        "weights",
        nargs="*",
        help="Weight files (default: best.pt and last.pt of the configured run)",
    ).completer = argcomplete.completers.FilesCompleter()

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    weights_dir = default_config.PROJECT_PATH / default_config.RUN_NAME / "weights"
    paths = [Path(p) for p in args.weights] or [weights_dir / "best.pt", weights_dir / "last.pt"]

    for path in paths:
        if not path.exists():
            print(f"Not found: {path}")
            continue
        changed = rename_in_checkpoint(path, args.old, args.new)
        print(f"{path}: {'renamed' if changed else f'no class named {args.old!r}'}")

    print(f"\nRemember to update names in {default_config.YAML_PATH} as well.")


if __name__ == "__main__":
    main()
