# PDF_OCR

I want to analyse PDF Datasheets for electronic engineers to build better circuits with
the help of llms and AI.

I would like to use the following steps and tools:

1. Use OcrmyPdf for adding annotation to old scanned PDF files.
2. analyse the structure of PDFs using PyMuPDF, opencv2 and yolo.
3. use Spacy and easy_ocr to add additonal information with nlp.
4. analyse Schematics for subcurcuits with yolo and other tools.
5. use a MCP to feed a LLM with the generated context
6. output additonal infomation for the electronic engineer to build better PCBs

This shall help by building complex curcuits or make repair work of old PCBs and curcuits way easier.
In general it shall improve the understanding of complex curcuits.

## 🚧 Current State

This repo currently has two subsystems at different levels of maturity, and **they are not
yet connected to each other**:

- **PDF pipeline** (`src/pdf_ocr/`, `pdf-ocr` command) — early stage. It renders PDF pages to
  images and runs the trained YOLO detector on them (step 2 above, detection only). A
  standalone [docling](https://github.com/docling-project/docling) conversion lives in
  `src/pdf_ocr/docling_convert.py` but isn't part of the pipeline yet. Steps 1 and 3–6 aren't
  wired in yet.
- **YOLO training framework** (`training_project/`) — mature. A trained single-class
  ("schematic") detector exists at `training_project/training_data/runs/run/weights/`, with a
  full CLI for training (`scripts/train.py`) and prediction (`scripts/predict.py`). See
  [`training_project/README.md`](training_project/README.md) for usage.

See [`docs/roadmap.md`](docs/roadmap.md) for the full breakdown, and
[`docs/overview.md`](docs/overview.md) for a more detailed current-state writeup (including
known in-progress/uncommitted work).

[OcrmyPDF ](https://github.com/ocrmypdf/OCRmyPDF)

```Bash
# use this commmand if you just want to annotate your PDF file (needs unpaper -> install with brew)
OcrmyPdf input.pdf output.pdf --deskew --clean --rotate-pages
```

Run a local server with label studio to label data for a yolo model

```Bash
# runns a local server with label-studio
label-studio start
```

Analyse documents (application):

```Bash
# Every PDF and image in input/ (the default)
uv run pdf-ocr analyse

# A single PDF or image, or another folder
uv run pdf-ocr analyse input/CEM33403345-VCO.pdf --dpi 200 --conf 0.25
```

Each input gets its own folder, which is overwritten on rerun:
`output/<name>/pages/` (rendered pages), `output/<name>/annotated/` (boxes drawn), and
`output/<name>/detections.json`. The JSON gives boxes in pixels and, for PDFs, in PDF points
(`bbox_pt`, origin top-left).

Add training material (training): drop PDFs/images into
`training_project/training_data/sources/manual/` and run `uv run pdf-ocr ingest`, or let
`uv run pdf-ocr crawl <source>` fetch license-checked material (Wikimedia Commons, KiCad
projects, archive.org service manuals; see `training_project/config/crawl.yaml`). The full
labeling workflow (ingest → autolabel → Label Studio → import → train) is described in
[`training_project/README.md`](training_project/README.md) ("Labeling workflow").

Folder roles:

| Folder | Used by | Contents |
|---|---|---|
| `input/` | application | PDFs and images to analyse (only the samples are tracked) |
| `output/` | application | per-document results |
| `training_project/training_data/sources/` | training | raw training material (`manual/`, later `crawled/<site>/`) |
| `training_project/training_data/unlabeled/` | training | ingested pages waiting for `autolabel.py` |
| `training_project/training_data/review_queue/` | training | images + Label Studio tasks waiting for human review |
| `training_project/training_data/train/`, `val/` | training | labeled dataset (`val/` only gets human-reviewed images) |

TODOs:

- [x] label data with label-studio
- [x] train a yolo model on the data --> take a look at training_data/
- [x] connect the YOLO model to the PDF pipeline (rasterize PDF pages -> run detection)
- [ ] generate more training data automaticaly -> use docling for datasheet conversion.
- [ ] checkout imagemagic for croping images
- [ ] scrape your kiCAD database library for training data.
- [x] Build a autolabeling image pipeline from the current trained model (scaffolded, see docs/)
- [ ] Generate a Github Project out of this Repository

See [`docs/`](docs/) for a fuller planning surface (current-state overview, roadmap, and a
proposal for integrating [TypeSafe AI](https://docs.typesafe.ai/introduction) for structured
decision points in the pipeline).

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

### Third-Party Dependencies

This project uses [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF), which is licensed under the [Mozilla Public License 2.0 (MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/).

OCRmyPDF is not included in this repository, but may be installed by users as a dependency. Its license terms apply independently and do not affect the MIT license of this project.
