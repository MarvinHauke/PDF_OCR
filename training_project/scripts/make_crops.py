#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Cut labeled schematics out of the page dataset for the subcircuit dataset.

Stage 2 of the detector finds functional blocks (power_supply, amplifier,
filter, oscillator) inside a schematic, not on the whole page, so small
details keep their resolution. Only `schematic` boxes are cropped (block
diagrams have no subcircuits). Crops come only from labels in the page
dataset's train/ and val/ (human-verified or confidently accepted), are
written to the subcircuit dataset's unlabeled/ folder, and are recorded in
crops_manifest.jsonl (parent image + box) so they can be traced back to the
page. Re-running only adds crops that aren't in the manifest yet.

Next step: autolabel.py --config config/subcircuits.yaml
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

from config.settings import Config

from src.autolabeler import IMAGE_SUFFIXES
from src.utils import config_file_completer

SUBCIRCUIT_CLASSES = ["power_supply", "amplifier", "filter", "oscillator"]
CROP_CLASS = "schematic"


def class_id(yaml_path: Path, name: str) -> int:
    names = yaml.safe_load(yaml_path.read_text())["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    return names.index(name)


def ensure_dataset_yaml(config: Config):
    """Create the subcircuit data.yaml on first use (same layout as the page dataset's)."""
    if config.YAML_PATH.exists():
        return
    config.YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.YAML_PATH.write_text(
        yaml.safe_dump(
            {
                "train": str(config.TRAINING_DATA_PATH / "train"),
                "val": str(config.TRAINING_DATA_PATH / "val"),
                "nc": len(SUBCIRCUIT_CLASSES),
                "names": SUBCIRCUIT_CLASSES,
            },
            sort_keys=False,
        )
    )
    print(f"Created {config.YAML_PATH}")


def yolo_to_pixels(line: str, w_img: int, h_img: int, pad: float):
    """YOLO 'cls cx cy w h' (normalized) -> padded, clamped pixel box."""
    _, cx, cy, w, h = (float(v) for v in line.split())
    x1, x2 = (cx - w / 2 - pad) * w_img, (cx + w / 2 + pad) * w_img
    y1, y2 = (cy - h / 2 - pad) * h_img, (cy + h / 2 + pad) * h_img
    return (
        max(0, round(x1)),
        max(0, round(y1)),
        min(w_img, round(x2)),
        min(h_img, round(y2)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Crop labeled schematics from the page dataset into the subcircuit dataset"
    )
    parser.add_argument(
        "--pages-config", help="Page dataset configuration (default: config/config.yaml)"
    ).completer = config_file_completer
    parser.add_argument(
        "--config",
        default="config/subcircuits.yaml",
        help="Subcircuit dataset configuration (default: config/subcircuits.yaml)",
    ).completer = config_file_completer
    parser.add_argument(
        "--pad", type=float, default=0.02, help="Padding around each box, as a fraction of the page"
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    pages = Config(config_file=args.pages_config) if args.pages_config else Config()
    crops = Config(config_file=args.config)
    ensure_dataset_yaml(crops)
    crop_class = class_id(pages.YAML_PATH, CROP_CLASS)

    out_dir = crops.UNLABELED_PATH
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = crops.TRAINING_DATA_PATH / "crops_manifest.jsonl"
    seen = set()
    if manifest_path.exists():
        seen = {json.loads(line)["crop"] for line in manifest_path.read_text().splitlines() if line}

    created = skipped = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for split in ("train", "val"):
            images_dir = pages.TRAINING_DATA_PATH / split / "images"
            labels_dir = pages.TRAINING_DATA_PATH / split / "labels"
            for image_path in sorted(images_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                # Number crops by their line in the label file, so relabeling another
                # box (e.g. as block_diagram) doesn't rename existing crops
                lines = [l for l in label_path.read_text().splitlines() if l.strip()]
                numbered = [(n, l) for n, l in enumerate(lines, start=1) if int(l.split()[0]) == crop_class]
                if not numbered:
                    continue

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    for n, line in numbered:
                        crop_name = f"{image_path.stem}-s{n}.png"
                        if crop_name in seen:
                            skipped += 1
                            continue
                        box = yolo_to_pixels(line, image.width, image.height, args.pad)
                        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                            continue  # degenerate label
                        image.crop(box).save(out_dir / crop_name)
                        manifest.write(
                            json.dumps(
                                {
                                    "crop": crop_name,
                                    "parent": str(image_path.relative_to(pages.TRAINING_DATA_PATH)),
                                    "box_px": list(box),
                                    "parent_size_px": [image.width, image.height],
                                }
                            )
                            + "\n"
                        )
                        seen.add(crop_name)
                        created += 1

    print(f"Created {created} crop(s) in {out_dir}, skipped {skipped} already made")
    print("Next: uv run python training_project/scripts/autolabel.py --config config/subcircuits.yaml")


if __name__ == "__main__":
    main()
