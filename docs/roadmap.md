# Roadmap

Consolidated from the TODOs already scattered across the root `README.md` and
`training_project/`, organized by horizon. This isn't a commitment — it's a planning
surface to revisit and reprioritize.

## Suggested starting point

1. **Reconcile the uncommitted `training_project/scripts/` changes first** — it's a small,
   contained cleanup (finish or drop the `argcomplete` migration) and gets you to a clean
   working tree before building on top of it.
2. ~~Then connect the YOLO model to the PDF pipeline.~~ **Done (2026-09-23)**: `uv run pdf-ocr`
   (`src/pdf_ocr/`). See "First real-page results" below.
3. **Only after that, invest in more/better training data.** The current model hits ~0.99
   mAP50 on a 10-image validation set whose images are all duplicates of training images (see
   below), so it says nothing about real-world generalization — testing it against actual rendered datasheet pages
   from step 2 will make it obvious whether more data (root README's KiCAD-scraping /
   autolabeling TODOs) is actually the bottleneck, rather than guessing upfront.

## First real-page results (2026-09-23)

`uv run pdf-ocr` on `input/CEM33403345-VCO.pdf` (6 pages, 200 DPI, conf 0.25, `last.pt`):
7 detections.

- **Found:** the large block/connection diagrams on pages 1 and 3 and a mid-size figure on
  page 6.
- **False positive:** the page 3 header strip (0.31).
- **Duplicates:** three overlapping boxes on one figure on page 5.
- **Missed:** the small figures at the top of pages 5 and 6. A full page is shrunk to 640px
  for YOLO, so small schematics become tiny.

Next experiments, cheapest first:
1. Raise `imgsz` for inference (e.g. 1280), or tile pages.
2. Try `best.pt` instead of `last.pt`.
3. Feed more real datasheet pages through the labeling workflow (`pdf-ocr ingest` →
   `autolabel.py` → Label Studio → `import_reviewed.py`) to grow the dataset.

## Near-term (unblock the core pipeline)

- [x] **Dependency cleanup (2026-09-23):** dropped `pypdf2` for `pypdf`, dropped `pymupdf`
      and `opencv-contrib-python` (both unused directly, and the latter conflicted with
      transitive `opencv-python`). See git log for the full detail of what broke and how it
      was fixed.
  - **RF-DETR evaluated and deferred (2026-09-23):** investigated as an Apache-2.0 alternative
    to `ultralytics` YOLO11 (AGPL-3.0). Dataset compatibility is fine (RF-DETR auto-detects
    this project's existing YOLO-format `training_data/`, no conversion needed), but **Apple
    Silicon/MPS support is not solid** — an open upstream GitHub issue reports MPS training
    making a Mac unresponsive followed by a CoreML crash on retry, with no official fix at
    time of writing; a required op silently falls back to CPU even when it doesn't crash.
    Decision: stay on YOLO (already fast and working via MPS on this Mac) and revisit RF-DETR
    once upstream Apple Silicon support matures, rather than risk training stability now.
  - `docling` (MIT) has no strong reason to switch away from; `marker` is a faster
    Surya-OCR-based alternative worth knowing about if conversion speed becomes a bottleneck.
- [x] Reconcile the in-progress, uncommitted changes in `training_project/scripts/` — migrated
      from custom bash/zsh completion scripts to standard `argcomplete` global registration.
- [x] **PDF → YOLO pipeline (2026-09-23):** `src/pdf_ocr/` package. `pdf-ocr analyse` takes
      a PDF, an image, or a folder (default `input/`) and writes `output/<name>/` with
      rendered pages, annotated pages and `detections.json` (boxes in px and PDF points).
- [x] **Labeling loop closed (2026-09-23):** `pdf-ocr ingest` (`training_data/sources/` →
      `unlabeled/`, deduplicated by hash). `autolabel.py` now routes whole images (no more
      partially labeled training images), sends no-detection pages to review so misses get
      labeled, and moves files so reruns can't duplicate. `import_reviewed.py` brings Label
      Studio JSON exports back into `train/`/`val/`; `val/` only gets human-reviewed images.
      Workflow documented in `training_project/README.md`.
- [x] **Class name typo fixed (2026-09-23):** `schemtaic` → `schematic` in `data.yaml` and in
      the `best.pt`/`last.pt` weights via `scripts/rename_class.py` (no retraining needed:
      label files store only class ids).
- [x] **Label Studio environment (2026-09-23):** `scripts/start_label_studio.sh`,
      `scripts/setup_label_studio.py` (projects, Local Files storage, backend, deduplicated
      task import via the API) and `labelstudio/ml_backend.py` (own implementation of the ML
      backend protocol, since `label-studio-ml` needs `opencv-python-headless`). Two projects:
      schematics on pages, subcircuits on crops.
- [x] **Subcircuit stage started (2026-09-23):** `config/subcircuits.yaml` (`power_supply`,
      `amplifier`, `filter`, `oscillator`), `scripts/make_crops.py` (156 crops from the
      labeled pages). First round is manual, since there's no subcircuit model yet.
- [ ] **Fix the page dataset's validation split.** All 10 `val/` images are identical copies of
      `train/` images (found by `make_crops.py`: 10 duplicate crop names). The ~0.99 mAP50 is
      therefore measured on training images and says little. Move or replace them with
      unseen, human-reviewed pages (e.g. from `import_reviewed.py`), then retrain.
- [ ] Plug the subcircuit model into `pdf-ocr analyse`: crop detected schematics and add
      subcircuits to `detections.json` (in page and PDF coordinates).
- [ ] **Image crawler** writing into `training_data/sources/crawled/<site>/`, recording URL,
      license and date for every file. Start with clearly licensed sources (KiCad libraries,
      Wikimedia Commons); manufacturer datasheets are copyrighted, so check each site's terms.
- [ ] Bring docling (`src/pdf_ocr/docling_convert.py`, still a standalone smoke test) into
      the pipeline, and extend `detections.json` into the intermediate-representation
      contract for the steps that follow (structure analysis, NLP enrichment).
- [ ] Wire in OCRmyPDF for scanned/annotated PDFs (already a dependency, not yet called
      from any pipeline code) — `ocrmypdf input.pdf output.pdf --deskew --clean --rotate-pages`.
- [ ] Implement `training_project/scripts/evaluate.py` (currently an empty stub) so trained
      models can be scored against the validation set without a one-off script.

## Mid-term (structure analysis + data generation)

- [ ] Analyse PDF structure with PyMuPDF/opencv2/YOLO (root README step 2) — layout regions,
      tables, figures — building on the existing trained YOLO model in `training_data/runs/run/`.
- [ ] Add NLP enrichment with Spacy/EasyOCR (root README step 3) — `spacy-layout` is already
      a dependency and pairs naturally with the docling output.
- [ ] Generate more training data automatically using docling for datasheet conversion
      (root README TODO).
- [ ] Evaluate ImageMagick for cropping images out of PDFs for training data (root README TODO).
- [ ] Scrape a KiCAD database/library for additional training data (root README TODO).
- [x] **Autolabeling image pipeline (scaffolded 2026-09-23)** — root README TODO, built as
      `training_project/src/{features,decisions,autolabeler}.py` +
      `scripts/autolabel.py`. Uses a threshold-stub decider until a real `TYPESAFE_API_KEY`
      is configured; see [`typesafe-integration.md`](./typesafe-integration.md#3-autolabeling-pre-filter-for-yolo-training-data--scaffolded)
      for the corrected design (Jev is text/JSON-only, arbitrates over extracted evidence, not
      the raw crop). Still needs: a real API key, and more source material (see the crawler item above).
      Flagged candidates route to a Label Studio pre-annotated task queue — run Label Studio
      via `uvx label-studio start` (isolated; installing it as a project dependency conflicts
      with the opencv version already required by easyocr/ultralytics, confirmed by testing).

## Long-term (schematic analysis + LLM hand-off)

- [ ] Analyse schematics for subcircuits with YOLO and other tools (root README step 4).
- [ ] Feed an LLM with the generated context via MCP (root README step 5) — see
      [`typesafe-integration.md`](./typesafe-integration.md) for using cheap Jev judgments
      as a routing/gating layer in front of this step to cut down on LLM calls.
- [ ] Output additional information back to the electronic engineer to help build/repair PCBs
      (root README step 6).

## Process

- [ ] Turn this repo's TODOs into a GitHub Project (root README TODO) once the near-term
      items above settle.
