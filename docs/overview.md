# Project Overview

## What this is

PDF_OCR analyses PDF datasheets for electronic engineers, combining OCR, layout/structure
analysis, and object detection (YOLO) so the extracted context can eventually be fed to an
LLM (via MCP) to help engineers build or repair circuits and PCBs.

The repo currently has two subsystems at different levels of maturity.

## Subsystem 1: PDF-OCR pipeline (repo root)

Status: **prototype**. `src/main.py` is currently just a smoke test of `docling`'s
`DocumentConverter`:

- Loads a hardcoded sample PDF (`input/CEM33403345-VCO.pdf`)
- Converts it to markdown + dict via docling
- Writes markdown to `tmp/markdown/file.md`

None of the other planned steps from the root `README.md` (OCRmyPDF annotation, PyMuPDF/opencv2
structure analysis, Spacy/EasyOCR enrichment, YOLO schematic/subcircuit analysis, MCP hand-off
to an LLM) are wired into this pipeline yet, despite the relevant packages already being
project dependencies.

## Subsystem 2: YOLO training framework (`training_project/`)

Status: **mature**. A self-contained, well-documented framework for training YOLO models,
optimized for Apple Silicon (MPS):

- `config/` — YAML-based config system (`config.yaml`, `mps_optimized.yaml`,
  `cpu_fallback.yaml`, JSON schema) plus a central `settings.py` Config class.
- `scripts/train.py` — CLI for training (device auto-select, config overrides), wraps
  `src/trainer.py:YOLOTrainer`.
- `scripts/predict.py` — CLI for running predictions, wraps `src/predictor.py:YOLOPredictor`.
- `scripts/evaluate.py` — **empty stub**, not implemented yet.
- `training_data/` — labeled dataset plus at least one completed training run
  (`runs/run/` with `best.pt`/`last.pt`, PR/F1/confusion-matrix curves, `results.csv`).

See [`../training_project/README.md`](../training_project/README.md) for full usage docs
(installation, training commands, troubleshooting).

## Tech stack

- Package/dependency management: `uv` (`pyproject.toml` + `uv.lock`), Python `>=3.12`.
- PDF/OCR: `ocrmypdf`, `pymupdf`, `pypdf2`, `pypdfium2`, `pdf2image`, `pytesseract`,
  `reportlab`, `docling`.
- CV/ML: `opencv-contrib-python`, `ultralytics` (YOLO), `numpy`. `torch`/`torchvision` are
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
