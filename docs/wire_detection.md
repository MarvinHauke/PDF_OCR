# Wire detection (M3): from schematic image to nets

Status: plan (2026-09-25). Part of the circuit analysis track in [`roadmap.md`](roadmap.md).

## Why

The YOLO models find *what* is on a schematic (page model: where; symbol model: resistor,
op-amp, junction dot, …). The circuit graph needs *how it is connected*: which pins share a
net. Without that, only KiCad projects (where the netlist is the graph) reach the rules,
the KiCanvas review and the graph dataset; PDFs stop at a list of symbols. Wire detection is
the missing link.

**Goal:** schematic image → nets, in the same graph format as `CircuitGraph`
(`src/pdf_ocr/circuit/graph.py`), so the pattern library, `--review`/`--import-review`,
`--evaluate` and the graph dataset work unchanged on circuits from PDFs.

## Ground truth for free

Like the symbol dataset (`training_project/scripts/kicad_symbol_labels.py`), wire labels come
from the crawled KiCad projects without manual labeling. A `.kicad_sch` sheet contains:

- `wire` segments (end points in mm) and `junction` points,
- `label`, `global_label`, `hierarchical_label` and power symbols (nets joined by name),
- symbols with pin positions in `lib_symbols` (pin `at` + length, transformed like the box).

Reuse: `parse_sexpr`, `placed_symbols` and `_transform` in `src/pdf_ocr/circuit/kicad_sch.py`
(already parses symbols and junctions), `render_sheet()` for the images, and the mm → pixel
mapping and per-project train/val split from `kicad_symbol_labels.py`. The symbol model
already has a `junction_dot` class.

## Steps

Each step ends with a number, measured against the true netlist.

### M3.0 Ground truth
- Extend `kicad_sch.py`: wires, labels, pin positions (same transform as the symbol boxes).
- Per rendered sheet: a wire mask (rasterized segments) and a JSON with segments, junctions,
  labels and pins in pixels.
- **Sanity check:** rebuild the nets from the geometry alone (segments touching, junctions,
  labels, pins on segment ends) and compare with the kicad-cli netlist. Expect 100 % on the
  crawled projects; anything less is a parser bug, not a detection problem.

### M3.1 Classical extractor, clean renders, true symbol boxes
- Binarize, mask out the symbol boxes, skeletonize, trace line segments; merge collinear
  pieces.
- Snap segment ends to symbol pins (where a wire enters the box edge).
- Junction dots connect; a crossing without a dot does not.
- Net labels with the same name connect (names from the ground truth for now).
- Using the true boxes isolates wire detection from symbol-detection errors.

### M3.2 End to end on renders
- Predicted symbols (symbol model) instead of true boxes; OCR for refs, values and net labels.
- Shows how errors of the stages add up; the gap to M3.1 is the symbol/OCR share.

### M3.3 Scans and photos
- Degraded renders first (blur, noise, skew, JPEG, line thickness), then real service-manual
  schematics from the page dataset.
- Where the classical extractor breaks: train a wire segmentation model on the rasterized
  masks (ultralytics segmentation, same framework and AGPL note as the detectors, or a small
  U-Net), then skeletonize → nets as in M3.1.

## Metrics

- **Pin-pair connectivity F1:** for every pair of pins, "same net?", predicted vs. true.
- **Nets:** count, and share of nets recovered exactly.
- **End to end:** run the pattern library on the extracted graph and on the true graph; compare
  the subcircuits (reuse the gold files and `--evaluate`).

## Output

`pdf-ocr circuit <image or PDF page>` writes the same `<name>.circuit.json` as for netlists,
so the rules, the KiCanvas review (where a KiCad source exists) and the graph dataset apply.

## Open questions and risks

- Hierarchical sheets: sheet pins and hierarchical labels join nets across sheets.
- Buses and bus entries.
- Crossings in scans where a junction dot is faint or missing.
- Unconnected pins and no-connect flags.
- Pin positions without KiCad: estimated from symbol class + box geometry.
