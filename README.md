# Annotated PDF Generator from Scanned Datasheets

This repository provides a local toolchain to analyze and annotate old scanned datasheets (e.g., service manuals or engineering documents). It uses OCR to extract content like text, headings, tables, and schematics, and overlays annotations on the original pages. The project is divided into two main components:

1. **CLI Pipeline Tool** – Runs end-to-end from PDF input to annotated PDF output.
2. **Jupyter Notebook** – For visual analysis, OCR tuning, and experimentation.

---

## 📁 Project Structure

```
PDF_OCR/
├── cli/                            # CLI entrypoint and Typer logic
│   └── main.py                     # Typer CLI tool
├── core/                           # Business logic modules
│   ├── pdf_utils.py                # PDF to image conversion, PDF writing
│   ├── ocr_pipeline.py             # OCR execution and annotation
│   └── toc_generator.py            # TOC generation from OCR output
├── notebook/                       # Jupyter notebooks for exploration
│   └── analysis.ipynb              # Interactive OCR and annotation tuning
├── assets/                         # Sample data and outputs
├── requirements.txt
├── README.md
└── setup.py                        # Optional package setup
```

---

## 🚀 Features

### CLI Pipeline (Typer)

- Input: scanned PDF (e.g. old datasheet)
- Output: annotated searchable PDF and JSON-formatted table of contents
- OCR engines supported:

  - pytesseract
  - easyocr
  - keras_ocr

- Modular language support (default: English)
- Retains original page images with overlay annotations
- Uses `Typer` for CLI with autocomplete and help
- Displays progress bars during processing

### Jupyter Notebook

- PDF-to-image conversion
- OCR visualization using:

  - spaCy (for heading/entity parsing)
  - pytesseract, easyocr, keras_ocr

- Visual overlays of text regions
- Adjustable OCR thresholds and parameters
- Non-destructive preview of annotations

---

## ⚖️ Installation

### Prerequisites

- Python >= 3.8
- Tesseract OCR installed ([https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract))
- Poppler for `pdf2image`

### Install

```bash
# Clone repo
$ git clone https://github.com/yourusername/annotated-datasheets.git
$ cd annotated-datasheets

# Create virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt

# Download spaCy model
$ python -m spacy download en_core_web_sm
```

---

## 🔍 Usage

### 1. CLI Pipeline

```bash
# Run annotation pipeline
$ python cli/main.py annotate \
    --input input/datasheet.pdf \
    --output output/annotated.pdf \
    --toc-json output/toc.json \
    --language en
```

Arguments:

- `--input`: path to scanned PDF
- `--output`: path to output annotated PDF
- `--toc-json`: path to output JSON TOC file
- `--language`: language code (default: en)

### 2. Jupyter Notebook

```bash
# Start Jupyter Lab or Notebook
$ jupyter lab
```

Open `notebook/analysis.ipynb` and follow these steps:

1. Convert PDF pages to images
2. Run OCR engines and tweak thresholds
3. Display bounding boxes and recognized text
4. Export updated annotations if needed

---

## ⚙️ Future Improvements

- Multilingual support (pluggable OCR language models)
- Table and schematic detection improvements
- Better layout detection and heading classification

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🌟 Example Output (TOC JSON)

```json
[
  { "page": 1, "title": "General Description" },
  { "page": 3, "title": "Electrical Characteristics" },
  { "page": 5, "title": "Schematic Diagram" }
]
```

---

Happy annotating! 📔🔍📏
