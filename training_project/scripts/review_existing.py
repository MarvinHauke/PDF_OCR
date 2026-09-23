#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Send images that are already in train/ or val/ back to Label Studio for review.

Used when the label set changes (e.g. new classes like block_diagram or pcb):
each task shows the image where it is, with its current boxes pre-drawn. After
review, import_reviewed.py rewrites the label file in place and leaves the
image in its split.

    review_existing.py train/images/foo.png val/images/bar.jpg
    review_existing.py --all            # every labeled image of the dataset
"""

import argparse
import json
import sys
from pathlib import Path

import argcomplete
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config, default_config

from src.autolabeler import IMAGE_SUFFIXES, local_files_url, to_label_studio_result
from src.utils import config_file_completer


def main():
    parser = argparse.ArgumentParser(description="Re-review images already in train/ or val/")
    parser.add_argument("images", nargs="*", help="Paths relative to training_data/, e.g. train/images/x.png")
    parser.add_argument("--all", action="store_true", help="All images in train/ and val/")
    parser.add_argument(
        "--config", help="Configuration YAML (default: config/config.yaml)"
    ).completer = config_file_completer

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = Config(config_file=args.config) if args.config else default_config
    data = config.TRAINING_DATA_PATH
    names = yaml.safe_load(config.YAML_PATH.read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)

    if args.all:
        images = [p for s in ("train", "val") for p in sorted((data / s / "images").iterdir())]
    else:
        images = [data / p for p in args.images]
    images = [p for p in images if p.suffix.lower() in IMAGE_SUFFIXES]
    missing = [p for p in images if not p.exists()]
    if missing or not images:
        print(f"Not found: {[str(p) for p in missing]}" if missing else "No images given")
        sys.exit(1)

    tasks_path = data / "review_queue" / "label_studio_tasks.json"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = {t["data"]["image"]: t for t in json.loads(tasks_path.read_text())} if tasks_path.exists() else {}

    for image_path in images:
        with Image.open(image_path) as image:
            w_img, h_img = image.size
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        results = []
        for i, line in enumerate(label_path.read_text().splitlines() if label_path.exists() else []):
            if not line.strip():
                continue
            cls, cx, cy, w, h = line.split()
            cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
            xyxy = ((cx - w / 2) * w_img, (cy - h / 2) * h_img, (cx + w / 2) * w_img, (cy + h / 2) * h_img)
            results.append(
                to_label_studio_result(
                    f"existing_{i}", xyxy, names[int(cls)], w_img, h_img,
                    config.LABEL_STUDIO_FROM_NAME, config.LABEL_STUDIO_TO_NAME,
                )
            )
        url = local_files_url(image_path, config.PROJECT_ROOT.parent)
        task = {"data": {"image": url}}
        if results:
            task["predictions"] = [{"model_version": "existing-labels", "score": 1.0, "result": results}]
        tasks[url] = task
        print(f"{image_path.relative_to(data)}: {len(results)} existing box(es)")

    tasks_path.write_text(json.dumps(list(tasks.values()), indent=2))
    print(f"\n{len(images)} task(s) in {tasks_path}")
    print("Next: uv run python training_project/scripts/setup_label_studio.py")


if __name__ == "__main__":
    main()
