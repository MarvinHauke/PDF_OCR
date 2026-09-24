#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Generate the symbol-detection dataset from crawled KiCad projects.

For every sheet (.kicad_sch) of every crawled project: render it with kicad-cli
(black and white, 200 DPI), compute each symbol's box from the schematic
itself (pdf_ocr/circuit/kicad_sch.py), map it to a symbol class, and cut the
sheet into overlapping 640 px tiles with YOLO labels. No manual labeling: the
schematic is the ground truth.

The train/val split is per project, so tiles of one sheet never end up on both
sides. Projects are assigned greedily in a fixed (hash) order: to val until it
holds ~val_fraction of the symbols and every class that occurs somewhere, the
rest to train (a plain random project split left val without op-amps).
Delete training_data/symbols/{train,val,symbols_manifest.jsonl} to regenerate.

Symbols without a class (connectors, switches, unknown-polarity transistors,
...) are left unlabeled; tiles whose area is mostly such symbols are dropped.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import argcomplete
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

from src.utils import config_file_completer

from pdf_ocr.circuit.components import classify
from pdf_ocr.circuit.kicad_sch import placed_symbols, render_sheet
from pdf_ocr.circuit.netlist import Component

CLASSES = [
    "resistor", "capacitor", "capacitor_polarized", "inductor", "potentiometer",
    "diode", "zener", "led", "bjt_npn", "bjt_pnp", "mosfet_n", "mosfet_p",
    "opamp", "ic", "ground", "supply", "junction_dot",
]
# circuit component type -> symbol class (visual appearance)
TO_CLASS = {c: c for c in CLASSES} | {
    "comparator_ic": "opamp", "timer_555": "ic", "linear_regulator": "ic",
}
DPI = 200
TILE, STRIDE = 640, 512
JUNCTION_MM = 0.6  # half size of a junction dot box
MIN_INSIDE = 0.6  # keep a box in a tile if at least this share of it is inside


def symbol_class(sym) -> str | None:
    if not sym.has_body:
        return None  # pin-only unit, e.g. the power unit of a quad op-amp
    lib = sym.lib_id.lower()
    if lib.startswith("power:"):
        if "pwr_flag" in lib:
            return None
        return "ground" if re.search(r"gnd|earth|vss|0v", lib) else "supply"
    lib_name, _, part = sym.lib_id.partition(":")
    comp = Component(sym.ref, lib_name, part, sym.value, sym.description, sym.keywords)
    return TO_CLASS.get(classify(comp))


def tiles_for_sheet(sheet: Path):
    """Yield (tile image, [(class id, x1, y1, x2, y2) in tile px]) for one sheet."""
    symbols, junctions, _ = placed_symbols(sheet)
    image = render_sheet(sheet, DPI)
    if image is None:
        print(f"  render failed: {sheet.name}")
        return
    px_per_mm = DPI / 25.4
    boxes = []
    for sym in symbols:
        cls = symbol_class(sym)
        if cls is None:
            continue
        x1, y1, x2, y2 = (v * px_per_mm for v in sym.box)
        boxes.append((CLASSES.index(cls), x1, y1, x2, y2))
    for x, y in junctions:
        boxes.append((CLASSES.index("junction_dot"), *((v * px_per_mm) for v in
                      (x - JUNCTION_MM, y - JUNCTION_MM, x + JUNCTION_MM, y + JUNCTION_MM))))

    w, h = image.size
    xs = list(range(0, max(w - TILE, 0) + 1, STRIDE)) + ([w - TILE] if w > TILE else [])
    ys = list(range(0, max(h - TILE, 0) + 1, STRIDE)) + ([h - TILE] if h > TILE else [])
    for ty in sorted(set(ys)):
        for tx in sorted(set(xs)):
            inside = []
            for cls, x1, y1, x2, y2 in boxes:
                cx1, cy1 = max(x1, tx), max(y1, ty)
                cx2, cy2 = min(x2, tx + TILE), min(y2, ty + TILE)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                share = (cx2 - cx1) * (cy2 - cy1) / max((x2 - x1) * (y2 - y1), 1e-6)
                if share >= MIN_INSIDE:
                    inside.append((cls, cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty))
            if inside:
                yield image.crop((tx, ty, tx + TILE, ty + TILE)), inside, (tx, ty)


def project_classes(project_dir: Path) -> dict[int, int]:
    """Symbol class counts of a project, from the schematics alone (no rendering)."""
    counts = {}
    for sheet in project_dir.rglob("*.kicad_sch"):
        symbols, junctions, _ = placed_symbols(sheet)
        for sym in symbols:
            cls = symbol_class(sym)
            if cls:
                counts[CLASSES.index(cls)] = counts.get(CLASSES.index(cls), 0) + 1
    return counts


def assign_splits(projects: dict[str, dict[int, int]], val_fraction: float) -> dict[str, str]:
    """Val first gets, for each class (rarest first), the smallest project containing it,
    then is topped up in hash order until it holds ~val_fraction of all symbols."""
    size = {p: sum(c.values()) for p, c in projects.items()}
    total = sum(size.values())
    order = sorted(projects, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    totals = {}
    for c in projects.values():
        for k, n in c.items():
            totals[k] = totals.get(k, 0) + n
    # a project that is the only source of some class must stay in training
    sole = {ps[0] for k in totals if len(ps := [p for p in projects if k in projects[p]]) == 1}
    val, val_classes = set(), set()
    for k in sorted(totals, key=totals.get):  # rarest class first
        if k in val_classes:
            continue
        having = [p for p in projects if k in projects[p] and p not in sole]
        if not having:
            continue
        pick = min(having, key=lambda p: size[p])
        val.add(pick)
        val_classes |= set(projects[pick])
    for p in order:
        if sum(size[v] for v in val) >= val_fraction * total:
            break
        if p not in val and p not in sole and size[p] < 0.5 * val_fraction * total:
            val.add(p)
    # every class needs training examples: move its largest val holder back if needed
    for k in totals:
        holders = [p for p in projects if k in projects[p]]
        if holders and all(p in val for p in holders):
            val.discard(max(holders, key=lambda p: projects[p][k]))
    missing = [CLASSES[k] for k in sorted(set(totals) - {k for p in val for k in projects[p]})]
    if missing:
        print("classes without val examples:", missing)
    print(f"val holds {sum(size[v] for v in val)} of {total} symbols")
    return {p: "val" if p in val else "train" for p in projects}


def ensure_dataset_yaml(config: Config):
    if config.YAML_PATH.exists():
        return
    config.YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.YAML_PATH.write_text(yaml.safe_dump({
        "train": str(config.TRAINING_DATA_PATH / "train"),
        "val": str(config.TRAINING_DATA_PATH / "val"),
        "nc": len(CLASSES), "names": CLASSES,
    }, sort_keys=False))
    print(f"Created {config.YAML_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Symbol dataset from crawled KiCad projects")
    parser.add_argument("--config", default="config/symbols.yaml").completer = config_file_completer
    parser.add_argument("--sources", type=Path,
                        default=PROJECT_ROOT / "training_data" / "subcircuits" / "sources" / "crawled" / "kicad_github",
                        help="Folder with <project>.kicad/ directories from the crawler")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config = Config(config_file=args.config)
    ensure_dataset_yaml(config)
    manifest_path = config.TRAINING_DATA_PATH / "symbols_manifest.jsonl"
    done = set()
    if manifest_path.exists():
        done = {json.loads(l)["project"] for l in manifest_path.read_text().splitlines() if l.strip()}

    project_dirs = {d.name.removesuffix(".kicad"): d for d in sorted(args.sources.glob("*.kicad"))}
    todo = {p: d for p, d in project_dirs.items() if p not in done}
    splits = assign_splits({p: project_classes(d) for p, d in todo.items()}, config.VAL_FRACTION)

    counts = {"train": 0, "val": 0}
    class_counts = [0] * len(CLASSES)
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for project, project_dir in todo.items():
            split = splits[project]
            images_dir = config.TRAINING_DATA_PATH / split / "images"
            labels_dir = config.TRAINING_DATA_PATH / split / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            n_tiles = 0
            for sheet in sorted(project_dir.rglob("*.kicad_sch")):
                sheet_id = f"{project}__{sheet.stem}"
                for tile, boxes, (tx, ty) in tiles_for_sheet(sheet):
                    name = f"{sheet_id}__{tx}_{ty}"
                    tile.save(images_dir / f"{name}.png")
                    (labels_dir / f"{name}.txt").write_text("".join(
                        f"{c} {(x1 + x2) / 2 / TILE:.6f} {(y1 + y2) / 2 / TILE:.6f} "
                        f"{(x2 - x1) / TILE:.6f} {(y2 - y1) / TILE:.6f}\n"
                        for c, x1, y1, x2, y2 in boxes))
                    for c, *_ in boxes:
                        class_counts[c] += 1
                    n_tiles += 1
            manifest.write(json.dumps({"project": project, "split": split, "tiles": n_tiles}) + "\n")
            counts[split] += n_tiles
            print(f"{project}: {n_tiles} tiles -> {split}/")

    print(f"\nTiles: {counts['train']} train, {counts['val']} val")
    print("Boxes per class:", {c: n for c, n in zip(CLASSES, class_counts) if n})


if __name__ == "__main__":
    main()
