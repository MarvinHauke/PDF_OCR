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

- **PDF-OCR pipeline** (`src/main.py`) — prototype stage. Right now it only converts one
  hardcoded PDF via [docling](https://github.com/docling-project/docling) and writes the
  result to `tmp/markdown/file.md`. None of steps 1, 3–6 above are wired in yet, despite the
  relevant packages already being dependencies.
- **YOLO training framework** (`training_project/`) — mature. A trained single-class
  ("schematic") detector exists at `training_project/training_data/runs/run/weights/`, with a
  full CLI for training (`scripts/train.py`) and prediction (`scripts/predict.py`). See
  [`training_project/README.md`](training_project/README.md) for usage.

The biggest open gap (step 2 above) is that nothing currently takes a PDF, rasterizes its
pages, and runs them through the trained YOLO model — that hand-off is the natural next
build target. See [`docs/roadmap.md`](docs/roadmap.md) for the full breakdown, and
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

Try the trained YOLO model manually:

```Bash
# Run detection over the sample images in img/
uv run python training_project/scripts/predict.py --img-folder

# Or on a specific image, with a chosen confidence threshold
uv run python training_project/scripts/predict.py img/test.png --conf 0.25
```

Annotated results are saved to `output/yolo_predictions/`. See
[docs/roadmap.md](docs/roadmap.md#how-to-test-the-yolo-model-manually) for more options
(picking a specific weights file, running over a whole PDF's rendered pages, etc.).

TODOs:

- [x] label data with label-studio
- [x] train a yolo model on the data --> take a look at training_data/
- [ ] connect the YOLO model to the PDF pipeline (rasterize PDF pages -> run detection)
- [ ] generate more training data automaticaly -> use docling for datasheet conversion.
- [ ] checkout imagemagic for croping images
- [ ] scrape your kiCAD database library for training data.
- [ ] Build a autolabeling image pipeline from the current trained model
- [ ] Generate a Github Project out of this Repository

See [`docs/`](docs/) for a fuller planning surface (current-state overview, roadmap, and a
proposal for integrating [TypeSafe AI](https://docs.typesafe.ai/introduction) for structured
decision points in the pipeline).

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

### Third-Party Dependencies

This project uses [OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF), which is licensed under the [Mozilla Public License 2.0 (MPL-2.0)](https://www.mozilla.org/en-US/MPL/2.0/).

OCRmyPDF is not included in this repository, but may be installed by users as a dependency. Its license terms apply independently and do not affect the MIT license of this project.
