#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Move human-reviewed images from the Label Studio review queue into train/ or val/.

Takes a Label Studio JSON export (Export -> JSON, not the YOLO zip: JSON keeps
our own image paths, so there's no need to copy renamed images out of Label
Studio). For every task with a submitted annotation, the image is moved from
review_queue/images/ to <split>/images/ and its boxes are written as a YOLO
label. A submitted annotation without boxes becomes an empty label, i.e. a
confirmed background image. The split is decided by a hash of the filename
(autolabel.val_fraction), so it's stable across runs.

Images that are already in train/ or val/ (queued by review_existing.py, e.g.
after adding classes) keep their place: only their label file is rewritten.

Tasks without an annotation, or whose annotation was skipped/cancelled, stay
in the review queue.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import argcomplete
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config, default_config

from src.utils import config_file_completer


def load_class_ids(yaml_path: Path) -> dict[str, int]:
    names = yaml.safe_load(yaml_path.read_text())["names"]
    if isinstance(names, dict):
        return {name: int(cls_id) for cls_id, name in names.items()}
    return {name: cls_id for cls_id, name in enumerate(names)}


def image_path_from_task(task: dict, repo_root: Path) -> Path:
    """'/data/local-files/?d=<repo-relative path>' -> absolute path."""
    url = task["data"]["image"]
    rel = parse_qs(urlparse(url).query).get("d")
    if not rel:
        raise ValueError(f"Not a local-files image URL: {url}")
    return repo_root / rel[0]


def latest_annotation(task: dict) -> dict | None:
    annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not annotations:
        return None
    return max(annotations, key=lambda a: a.get("updated_at") or a.get("created_at") or "")


def to_yolo_lines(annotation: dict, class_ids: dict[str, int]) -> list[str]:
    lines = []
    for result in annotation.get("result", []):
        if result.get("type") != "rectanglelabels":
            continue
        value = result["value"]
        for label in value["rectanglelabels"]:
            if label not in class_ids:
                raise ValueError(
                    f"Label {label!r} is not in data.yaml names {list(class_ids)} -- "
                    "fix the Label Studio labeling config"
                )
            # Label Studio stores percentages of the image size, top-left corner
            w, h = value["width"] / 100, value["height"] / 100
            cx, cy = value["x"] / 100 + w / 2, value["y"] / 100 + h / 2
            lines.append(f"{class_ids[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def pick_split(filename: str, val_fraction: float) -> str:
    bucket = int(hashlib.sha256(filename.encode()).hexdigest(), 16) % 1000
    return "val" if bucket < val_fraction * 1000 else "train"


def main():
    parser = argparse.ArgumentParser(
        description="Import a Label Studio JSON export into train/ and val/"
    )
    parser.add_argument(
        "export", help="Label Studio JSON export file"
    ).completer = argcomplete.completers.FilesCompleter()
    parser.add_argument(
        "--config", help="Configuration YAML (default: config/config.yaml)"
    ).completer = config_file_completer
    parser.add_argument(
        "--val-fraction",
        type=float,
        help="Share of images that go to val/ (default: autolabel.val_fraction)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print what would happen")

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = Config(config_file=args.config) if args.config else default_config
    val_fraction = args.val_fraction if args.val_fraction is not None else config.VAL_FRACTION
    repo_root = config.PROJECT_ROOT.parent
    review_dir = config.TRAINING_DATA_PATH / "review_queue"
    tasks_path = review_dir / "label_studio_tasks.json"

    try:
        class_ids = load_class_ids(config.YAML_PATH)
        exported = json.loads(Path(args.export).read_text())

        # Convert everything first, so a bad label aborts before any file moves
        imports = []
        pending = 0
        for task in exported:
            annotation = latest_annotation(task)
            if annotation is None:
                pending += 1
                continue
            image_path = image_path_from_task(task, repo_root)
            lines = to_yolo_lines(annotation, class_ids)
            imports.append((task["data"]["image"], image_path, lines))
    except (ValueError, KeyError) as e:
        print(f"Import aborted, nothing was moved: {e}")
        sys.exit(1)

    counts = {"train": 0, "val": 0, "relabeled": 0, "missing": 0}
    existing_splits = {
        (config.TRAINING_DATA_PATH / s / "images").resolve(): s for s in ("train", "val")
    }
    imported_urls = set()
    for url, image_path, lines in imports:
        if not image_path.exists():
            # Already imported earlier, or moved by hand
            print(f"missing, skipped: {image_path.name}")
            counts["missing"] += 1
            continue

        in_split = existing_splits.get(image_path.resolve().parent)
        if in_split:
            # Re-reviewed image (review_existing.py): rewrite the label, keep the image
            print(f"{image_path.name}: relabeled in {in_split}/ ({len(lines)} box(es))")
            counts["relabeled"] += 1
            imported_urls.add(url)
            if not args.dry_run:
                label = config.TRAINING_DATA_PATH / in_split / "labels" / f"{image_path.stem}.txt"
                label.write_text("\n".join(lines) + "\n" if lines else "")
            continue

        split = pick_split(image_path.name, val_fraction)
        images_dir = config.TRAINING_DATA_PATH / split / "images"
        labels_dir = config.TRAINING_DATA_PATH / split / "labels"
        print(f"{image_path.name} -> {split}/ ({len(lines)} box(es))")
        counts[split] += 1
        imported_urls.add(url)

        if args.dry_run:
            continue
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (labels_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + "\n" if lines else ""
        )
        shutil.move(image_path, images_dir / image_path.name)

    if not args.dry_run and imported_urls and tasks_path.exists():
        remaining = [
            t for t in json.loads(tasks_path.read_text()) if t["data"]["image"] not in imported_urls
        ]
        tasks_path.write_text(json.dumps(remaining, indent=2))

    prefix = "Would import" if args.dry_run else "Imported"
    print(
        f"\n{prefix}: {counts['train']} -> train/, {counts['val']} -> val/, "
        f"{counts['relabeled']} relabeled in place"
        f"\nMissing images: {counts['missing']}"
        f"\nNot yet reviewed (left in the queue): {pending}"
    )


if __name__ == "__main__":
    main()
