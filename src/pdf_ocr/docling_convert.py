"""Docling conversion smoke test (not wired into the detection pipeline yet).

Run from the repo root: uv run python -m pdf_ocr.docling_convert
"""

import os

from docling.document_converter import DocumentConverter


def main():
    converter = DocumentConverter()
    PDF = "./input/CEM33403345-VCO.pdf"
    result = converter.convert(PDF)
    document = result.document
    markdown_output = document.export_to_markdown()
    print(markdown_output)
    os.makedirs("./tmp/markdown/", exist_ok=True)

    with open("./tmp/markdown/file.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)


if __name__ == "__main__":
    main()
