# Experiment log

Lab notebook for training runs and findings: what was tried, on which data, what came out,
and what it means. Entries go stale as the data grows; that's expected. Lasting decisions
move to `docs/roadmap.md`, everything else stays here.

Rules:
- One entry per run or experiment, newest at the bottom, never rewrite old numbers.
- Always note the **data state** (train/val counts): metrics are only comparable on the same
  validation set.
- Change one thing per run where possible, so the effect can be attributed.
- Open ideas go to the backlog at the end; move them into an entry once tried.

Runs live in `training_data/runs/<name>/` (gitignored): `results.csv`, `args.yaml`, weights.

## Entries

### 2026-09-23 · First real-page test (`run`, CEM3340 datasheet)
- `pdf-ocr analyse` on 6 datasheet pages, 200 DPI, conf 0.25, `last.pt`: 7 detections.
- Found the large diagrams (pages 1, 3, 6). False positive on the page 3 header (0.31),
  three overlapping boxes on one page 5 figure, **missed the small figures** on pages 5/6
  (a full page is shrunk to 640 px).

### 2026-09-23 · `run` was validated on its own training data
- All 10 `val/` images were byte-identical copies of `train/` images, plus one duplicate pair
  inside `train/`. The reported mAP50 0.995 / mAP50-95 0.867 is meaningless.
- Fix: duplicates removed from `train/` → 47 train / 10 val (backup:
  `training_data/backup_2026-09-23_before_split_fix/`).

### 2026-09-23 · `run2`: first honest baseline
- Data: 47 train / 10 val, 1 class. Settings: yolo11n, 640 px, 100 epochs, AdamW lr0 0.01.
- best.pt (epoch 92): **mAP50 0.686, mAP50-95 0.541, P 0.74, R 0.70**. Last epoch only
  mAP50 0.442 → switched inference to `best.pt` (`use_best_weights: true`).
- Almost nothing learned before epoch ~40.

### 2026-09-24 · `run2_imgsz1280`: resolution test
- Same data as `run2` (frozen 1-class data.yaml), only `image_size: 1280`.
- val: mAP50 0.580, mAP50-95 0.278, P 0.76, R 0.50 (best epoch 64): worse than `run2`.
- On the CEM datasheet it **does find the small schematics** 640 px misses (page 5: 0.55–0.99),
  but produces ~30 false positives on 6 pages (tables, text columns, margins, blank edges).
- Reading: too few negative examples to learn "text/table ≠ figure" at 1280. Retest (and
  960) once more figure-free pages are in the dataset.

### 2026-09-24 · Import round 1 and `run3`: 3 page classes
- Data: 81 train / 18 val (25 figure-free pages: 19 train, 6 val). Boxes train: schematic 147,
  block_diagram 5, pcb 20; val: schematic 13, block_diagram 2, pcb 1.
- Settings as `run2`, 3 classes (`schematic`, `block_diagram`, `pcb`), 100 epochs.
- best.pt (epoch 88) per class mAP50: schematic 0.429, block_diagram 0.529, pcb 0.497
  (P for schematic only 0.13). last.pt: 0.467 / 0.517 / 0.332.
- Comparison on the **new** val set, schematic only: `run2` 0.557, `run` 0.713 (inflated: 10 of
  the 18 val images were in its training data).
- Training curve (mAP50, all classes): ep50 0.08 → ep60 0.37 → ep88 0.44 → ep100 0.44, still
  rising → undertrained. The block_diagram/pcb values rest on 1–2 val boxes each.

### 2026-09-24 · `run4`: 200 epochs (running)
- Same data and settings as `run3`, only `--epochs 200` (patience 50).

## Backlog (ideas, not tried yet)

Planned after `run4`, one change per run:
- **`run5`: learning rate.** Config forces AdamW with `lr0: 0.01`, which is the SGD value
  (ultralytics default.yaml: "SGD=1E-2, Adam=1E-3"). Ultralytics' own `optimizer: auto`
  would pick AdamW with lr0 = 0.002·5/(4+nc) ≈ 0.0014 for 3 classes. Likely cause of the slow
  start in every run so far.
- **`run6`: augmentations for phone photos of books.** `degrees: 3–5`, `perspective: 0.0005`
  (both 0 now); question `fliplr: 0.5` (mirrors text) and `mosaic` for documents.
- **`run7`: model size** `yolo11s` (~9 M params) instead of `yolo11n` (2.6 M).

Later:
- 960 px / 1280 px again once there are more figure-free pages; or page tiling.
- More `block_diagram` examples: `pdf-ocr crawl wikimedia_blocks`.
- More `pcb` examples: export PCB layouts from the crawled KiCad projects (`kicad-cli pcb
  export`), and PCB assembly drawings from service manuals.
- Grow `val/` with human-reviewed pages; per-class numbers for rare classes are noise today.
- Subcircuits: first manual labeling round (13 classes), then a first model.
