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

### 2026-09-24 · `run4`: 200 epochs
- Same data and settings as `run3`, only `--epochs 200` (patience 50); early-stopped at
  epoch 135, best epoch 85.
- best.pt per class mAP50: **schematic 0.745** (P 0.89, R 0.62), block_diagram 0.513,
  pcb 0.995 (1 val box). All classes: mAP50 0.751, mAP50-95 0.369.
- Curve (mAP50): ep40 0.05 → ep60 0.13 → ep80 0.31 → ep120 0.45; still a very slow start.
- Beats `run2` on schematic (0.557 on the same val set) → inference switched to `run4`
  (`run_name: "run4"`).

### 2026-09-24 · Decision: subcircuits become graph-based
- Building blocks (current mirror, voltage divider, …) are topologies, not shapes; YOLO
  learns appearance. New direction: symbol detection (YOLO) + connectivity graph + graph
  pattern matching, with Jev arbitrating ambiguous matches. YOLO subcircuit labeling
  (project #5) paused before any labels were made. Details: `docs/roadmap.md` → Circuit analysis.

### 2026-09-24 · `run5`: learning rate 0.0014
- Same data and settings as `run4` (200 epochs), only `lr0: 0.0014` instead of 0.01 (AdamW with
  the SGD default was the suspect for the slow start). Early-stopped at 137, best epoch 87.
- Learns from the start: mAP50 0.24 at epoch 20 (`run4`: 0.00).
- best.pt per class: schematic mAP50 0.724 / **mAP50-95 0.554** (`run4`: 0.745 / 0.408),
  block_diagram 0.551 / 0.380, pcb 0.995 / 0.796 (1 val box). All: mAP50 0.756, **mAP50-95 0.577**
  (`run4`: 0.751 / 0.369) → same hit rate, much tighter boxes.
- Adopted: `lr0: 0.0014` in `config.yaml`, inference switched to `run5`.

### 2026-09-24 · Symbol dataset from KiCad projects (`symbols_run1` running)
- 24 crawled KiCad projects → `scripts/kicad_symbol_labels.py`: sheets rendered black-and-white
  at 200 DPI, symbol boxes computed from the `.kicad_sch` (overlay checked on 3 sheets: boxes sit
  on rotated/mirrored symbols and junction dots), cut into 640 px tiles (stride 512).
- 775 tiles, 17 classes; train 708 / val 67 tiles, split per project so that every class is in
  both (a plain random project split left val without op-amps; a class occurring in only val
  projects left training without zeners and P-MOSFETs).
- Boxes train/val: resistor 1019/84, capacitor 676/103, junction_dot 2794/409, ground 1195/127,
  opamp 53/3, bjt_npn 12/1, bjt_pnp 9/3, mosfet_p 5/2, zener 4/2 → transistor and zener classes
  are too thin to learn yet.
- `symbols_run1`: yolo11n, 640 px tiles, 200 epochs, lr0 0.0014, degrees 3, flipud 0.5.
  (First attempt crashed on MPS in the first batch with "size of tensor a must match the size of
  tensor b"; not reproducible, the restart ran through. Newer ultralytics has MPS fixes in the
  assigner, PR #24222.)

### 2026-09-24 · `symbols_run1` results
- 200 epochs (5.8 h, no early stop, still improving slowly). Val (67 render tiles, 871 boxes):
  **mAP50 0.852, mAP50-95 0.801**, P 0.76, R 0.92.
- Per class mAP50: resistor 0.981, capacitor 0.995, junction_dot 0.994, ground 0.939, supply
  0.963, ic 0.836, diode 0.823, led/zener/opamp/mosfet 0.995 (1–13 boxes each), bjt_pnp 0.746,
  **bjt_npn 0.249** (1 box), **inductor 0.000** (5 boxes, none found).
- Real crops (conf 0.3): clean digital schematics (EAGLE screenshot, electricdruid datasheet)
  are good for resistors, capacitors, op-amps, junction dots, supply; misses EAGLE-style ground
  symbols and some ICs, calls a P-MOSFET `ic`. Book photos (DIN symbols) are weak: some DIN
  resistors and junction dots found, the DIN op-amp triangle (∞) not, one false `led`.
- Reading: renders are learned well; the gap is symbol *style* (DIN, EAGLE ground) and photo
  quality. Next: pre-label the 145 real crops and correct them (B3), more symbol styles.

### 2026-09-24 · Import round 2 (service manuals) and `run6`
- `pdf-ocr crawl archive_org --limit 10` → `ingest --max-pages 8` (pages now spread evenly over
  each manual) → 76 pages; `run5` pre-labeled, 2 auto-accepted, 74 reviewed in Label Studio:
  34 with figures (32 schematic, 8 block_diagram, 22 pcb boxes), 41 without.
- Data: **141 train / 34 val** (48 / 18 figure-free). Boxes train: schematic 175,
  block_diagram 13, pcb 38; val: schematic 16, block_diagram 2, pcb 8.
- `run6`: same settings as `run5` (lr0 0.0014, 200 epochs, no early stop, best epoch 169).
- Both on the new val set (34 images, 26 boxes):

  | | all mAP50 | all mAP50-95 | schematic mAP50 | block_diagram | pcb | recall |
  |---|---|---|---|---|---|---|
  | `run5` | 0.642 | 0.428 | 0.672 | 0.543 | 0.712 | 0.59 |
  | `run6` | **0.928** | **0.722** | **0.866** | 0.995 (2 boxes) | **0.923** | **0.77** |

- Reading: more (and more varied) data is the biggest lever so far. Part of the gap is that
  `run5` had never seen a service manual. Inference switched to `run6`.

### 2026-09-24 · Import round 3 (service manuals) and `run7`
- 10 more archive.org manuals, 8 pages each → 80 pages; `run6` pre-labeled, 5 auto-accepted,
  74 reviewed: 28 with figures (33 schematic, 12 block_diagram, 10 pcb boxes), 46 without.
- Data: **204 train / 50 val** (83 / 29 figure-free). Boxes train: schematic 210, block_diagram 22,
  pcb 47; val: schematic 18, block_diagram 5, pcb 10.
- `run7`: same settings as `run6`, early-stopped at 179, best epoch 129. Long plateau (ep 60–100
  around 0.6) before improving again.
- Both on the new val set (50 images, 33 boxes):

  | | all mAP50 | all mAP50-95 | schematic mAP50 | block_diagram | pcb mAP50 / R | P |
  |---|---|---|---|---|---|---|
  | `run6` | 0.699 | 0.549 | **0.885** | 0.401 | 0.812 / **0.61** | 0.74 |
  | `run7` | **0.813** | **0.580** | 0.806 | **0.810** | 0.823 / 0.42 | **0.90** |

- Reading: mixed. Block diagrams gain a lot (9 new examples), precision rises; schematic and pcb
  recall drop. With 33 val boxes (18 schematic) one figure moves a class by ~5 points, so part of
  this is noise; val needs to grow further before small differences mean anything.
- Decision (user): inference switched to `run7` (better overall and on block diagrams, most data).

### 2026-09-24 · Import round 4 and `run8`
- Round 4: 10 archive.org documents via a new broader query (7 turned out to be declassified CIA
  reading-room files: some tube schematics, many forms/charts) plus 3 Roland manuals (user's own),
  8 pages each; 17 pages were auto-accepted by `run7` (≥ 0.75) and at least 3 were wrong (a chart,
  an archive form, a revision sheet) → all 17 re-reviewed. 95 tasks reviewed: 31 schematic,
  34 block_diagram, 35 pcb boxes, 49 empty pages.
- Data: **283 train / 70 val**. Boxes train: schematic 245, block_diagram 52, pcb 81; val: 19, 9, 12.
- `run8`: same settings as `run7`; early-stopped at 115, best epoch 65 (by fitness = mostly
  mAP50-95). mAP50 was still rising at the stop (ep 90: 0.52, ep 105: 0.54), train loss still falling.
- Both on the new val set (70 images, 40 boxes):

  | | all mAP50 | all mAP50-95 | schematic | block_diagram | pcb | P | R |
  |---|---|---|---|---|---|---|---|
  | `run7` | **0.645** | **0.469** | **0.729** | 0.477 | **0.728** | **0.86** | 0.54 |
  | `run8` | 0.563 | 0.412 | 0.690 | 0.483 | 0.515 | 0.51 | 0.60 |

- Reading: `run8` worse, likely stopped too early (run7 only broke out of its plateau after
  epoch 120); label inconsistencies in the new material are the other suspect. Inference stays
  on `run7`. Auto-accept at 0.75 is not safe on new document types.

### 2026-09-25 · Import round 5 (`run9` still training)
- Auto-accept off (`accept_threshold: 1.01`): all 88 new pages went to review (63 with boxes
  from `run7`, 25 without detections). Sources: the missing 33 pages of the Roland RE-101/201 and
  RE-301 manuals (`ingest --refill`) and 55 pages from 8 archive.org synth service manuals
  (service-manual query only, no CIA hits). Plus 1 leftover round-4 task.
- Data: **355 train / 87 val**. Boxes train: schematic 274, block_diagram 75, pcb 97 (146 empty
  pages); val: 22, 18, 15 (51 empty). block_diagram in val doubled (9 → 18).
- `run9` (run8 data, 300 epochs, patience 100) was started before the import. It crashed once
  on MPS at epoch 38 (tensor shape mismatch in the loss, same family as the symbol-model crash)
  and was resumed from `last.pt` via ultralytics directly: `train.py`'s auto-resume never
  triggers, because `is_run_complete()` counts any existing weights file as finished
  (fixed 2026-09-25: complete = last.pt stripped by ultralytics; resume always from last.pt).
- Compare `run7`/`run8`/`run9` on the round-4 val set (list in
  `backup_2026-09-24_before_import5/val_images_run9.txt`), then `run10` on the round-5 data.

## Backlog (ideas, not tried yet)

Next up:
- **Finish `run9`** and compare as above; then **`run10`** on 355/87.
- **Label check of round 4:** run8's biggest errors on val, consistency of block_diagram vs
  schematic on the new pages (CIA documents, Roland manuals).

Page model, one change per run (`run5` learning rate done, see above):
- **`run6`: augmentations for phone photos of books.** `degrees: 3–5`, `perspective: 0.0005`
  (both 0 now); question `fliplr: 0.5` (mirrors text) and `mosaic` for documents.
- **`run7`: model size** `yolo11s` (~9 M params) instead of `yolo11n` (2.6 M).

Later:
- 960 px / 1280 px again once there are more figure-free pages; or page tiling.
- More `block_diagram` examples: `pdf-ocr crawl wikimedia_blocks`.
- More `pcb` examples: export PCB layouts from the crawled KiCad projects (`kicad-cli pcb
  export`), and PCB assembly drawings from service manuals.
- Grow `val/` with human-reviewed pages; per-class numbers for rare classes are noise today.
- Symbols: evaluate on real schematic crops (scans, book photos), not only on KiCad renders;
  pre-label the 145 crops with the symbol model and correct them (domain gap).
- Symbols: more transistor/zener examples (crawl more analog projects: eurorack, synth, audio).
