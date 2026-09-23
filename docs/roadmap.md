# Roadmap

Consolidated from the TODOs already scattered across the root `README.md` and
`training_project/`, organized by horizon. This isn't a commitment — it's a planning
surface to revisit and reprioritize.

## Suggested starting point

1. **Reconcile the uncommitted `training_project/scripts/` changes first** — it's a small,
   contained cleanup (finish or drop the `argcomplete` migration) and gets you to a clean
   working tree before building on top of it.
2. **Then connect the YOLO model to the PDF pipeline.** This is the highest-leverage next
   step: the two subsystems (docling conversion in `src/main.py`, the trained detector in
   `training_project/`) are fully built but never talk to each other. Concretely: render a
   PDF's pages to images (`pypdfium2`/`pdf2image`, both already dependencies), feed each page
   through `training_project/src/predictor.py:YOLOPredictor`, and see what the current
   "schematic" detector actually finds on real datasheet pages. This turns root README step 2
   from a TODO into a working (if rough) pipeline, and gives concrete, real-world detections
   to decide whether the dataset/model needs more work before investing further.
3. **Only after that, invest in more/better training data.** The current model hits ~0.99
   mAP50 on an 11-image, single-class validation set, which is easy to hit and doesn't say
   much about real-world generalization — testing it against actual rendered datasheet pages
   from step 2 will make it obvious whether more data (root README's KiCAD-scraping /
   autolabeling TODOs) is actually the bottleneck, rather than guessing upfront.

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
- [ ] Turn `src/main.py` from a docling smoke test into an actual pipeline step: run it
      over more than one hardcoded PDF, and decide on an output/intermediate-representation
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
- [ ] Build an autolabeling image pipeline from the currently trained model (root README TODO)
      — see [`typesafe-integration.md`](./typesafe-integration.md) for a proposed pre-filter
      using Jev's `Score`/`Noul` primitives to sanity-check auto-generated labels before they
      enter the training set.

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
