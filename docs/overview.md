# Project Overview

## What this is

PDF_OCR analyses PDF datasheets for electronic engineers, combining OCR, layout/structure
analysis, and object detection (YOLO) so the extracted context can eventually be fed to an
LLM (via MCP) to help engineers build or repair circuits and PCBs.

The repo currently has two subsystems at different levels of maturity.

## Subsystem 1: PDF pipeline (`src/pdf_ocr/`)

Status: **early**. An installable package with two commands:

- `pdf-ocr analyse [path]` (`pipeline.py`): takes a PDF, an image, or a folder of both
  (default `input/`). Renders PDF pages to PNG with `pypdfium2` (default 200 DPI,
  `sources.py`) and runs the training project's `YOLOPredictor` on every page (`detect.py`).
  Writes `output/<name>/{pages,annotated}/` and `detections.json` with boxes in pixels and
  PDF points.
- `pdf-ocr ingest [path]` (`ingest.py`): renders training material from
  `training_data/sources/` (recursively) into `training_data/unlabeled/`, skipping files
  already listed by hash in `training_data/ingest_manifest.jsonl`. No detection happens here.
  `--dataset subcircuits` targets the crop dataset, `--max-pages` caps pages per PDF.
- `pdf-ocr crawl <source>` (`crawl/`): downloads training material (Wikimedia Commons, KiCad
  projects on GitHub, archive.org service manuals, a curated URL list) with license tiers,
  host allow/block lists and a 1 GB cap; settings in `training_project/config/crawl.yaml`.

`docling_convert.py` is the old `src/main.py` docling smoke test (writes
`tmp/markdown/file.md`). It isn't wired into the pipeline yet. OCRmyPDF, Spacy/EasyOCR,
subcircuit analysis and the MCP hand-off aren't wired in either.

`training_project/` isn't an installed package: its modules import each other as
`config.*` / `src.*`. So `detect.py` puts `training_project/` at the front of `sys.path`,
the same way `training_project/scripts/` does.

## Subsystem 2: YOLO training framework (`training_project/`)

Status: **mature**. A self-contained, well-documented framework for training YOLO models,
optimized for Apple Silicon (MPS):

- `config/` — YAML-based config system (`config.yaml`, `mps_optimized.yaml`,
  `cpu_fallback.yaml`, JSON schema) plus a central `settings.py` Config class.
- `scripts/train.py` — CLI for training (device auto-select, config overrides), wraps
  `src/trainer.py:YOLOTrainer`.
- `scripts/predict.py` — CLI for running predictions, wraps `src/predictor.py:YOLOPredictor`.
- `scripts/autolabel.py` — routes each image in `training_data/unlabeled/` to `train/`, the
  Label Studio review queue, or `skipped/`.
- `scripts/import_reviewed.py` — turns a Label Studio JSON export into `train/`/`val/` labels.
- `scripts/rename_class.py` — renames a class inside trained weights (no retraining).
- `scripts/make_crops.py` + `config/subcircuits.yaml` — second dataset: functional subcircuit
  blocks inside schematic crops.
- `scripts/start_label_studio.sh`, `scripts/setup_label_studio.py`, `labelstudio/` — Label
  Studio setup, labeling configs and our own ML backend (see `training_project/README.md`).
- `scripts/evaluate.py` — **empty stub**, not implemented yet.
- `training_data/` — labeled dataset plus at least one completed training run
  (`runs/run/` with `best.pt`/`last.pt`, PR/F1/confusion-matrix curves, `results.csv`).

See [`../training_project/README.md`](../training_project/README.md) for full usage docs
(installation, training commands, troubleshooting).

## Tech stack

- Package/dependency management: `uv` (`pyproject.toml` + `uv.lock`), Python `>=3.12`.
- PDF/OCR: `ocrmypdf`, `pypdf`, `pypdfium2`, `pdf2image`, `pytesseract`, `reportlab`,
  `docling`.
- CV/ML: `ultralytics` (YOLO, pulls in `opencv-python`), `numpy`. `torch`/`torchvision` are
  pulled in transitively via `ultralytics`.
- NLP: `spacy`, `spacy-layout`.
- Dev/misc: `jupyterlab`, `notebook`, `matplotlib`, `argcomplete`, `typer`, `pyyaml`.
- No `requirements.txt`/`environment.yml`/`package.json` — `uv` is the single source of truth.

## Known gaps / in-progress work (flag before building on top)

- `training_project/scripts/evaluate.py` is an empty stub.
- `training_project/README.md` referenced a `training_project/docs/` folder for detailed guides
  that never existed; docs now live at the repo root instead (this folder).
- `.serena/` had no prior memories — Serena was just (re)registered as an MCP server for this
  project; its onboarding will populate `.serena/memories/` once its tools are loaded in a new
  session.

## See also

- [`roadmap.md`](./roadmap.md) for consolidated next steps.
- [`typesafe-integration.md`](./typesafe-integration.md) for a proposal on using TypeSafe AI's
  Jev model for structured decision points in the pipeline.
