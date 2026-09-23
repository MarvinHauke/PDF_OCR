"""Collect input documents and turn each one into a folder of page images."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

PDF_SUFFIX = ".pdf"
# Same set training_project/src/autolabeler.py accepts
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Page:
    number: int  # 1-based
    image_path: Path
    size_px: tuple[int, int]
    size_pt: tuple[float, float] | None = None  # only set for PDF pages


@dataclass
class Document:
    source: Path
    kind: str  # "pdf" or "image"
    dpi: int | None = None
    pages: list[Page] = field(default_factory=list)


def collect_sources(path: Path, recursive: bool = False) -> list[Path]:
    """A single file, or every PDF/image in a folder (top level unless recursive)."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES | {PDF_SUFFIX}:
            raise ValueError(f"Unsupported input type: {path}")
        return [path]
    return sorted(
        p
        for p in (path.rglob("*") if recursive else path.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | {PDF_SUFFIX}
    )


def page_filename(number: int) -> str:
    return f"page-{number:03d}.png"


def prepare_document(source: Path, pages_dir: Path, dpi: int, max_pages: int = 0) -> Document:
    """Render a PDF's pages (or copy a single image) into pages_dir as page-NNN.png.

    max_pages > 0 renders only the first max_pages pages."""
    pages_dir.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == PDF_SUFFIX:
        doc = Document(source=source, kind="pdf", dpi=dpi)
        pdf = pdfium.PdfDocument(str(source))
        try:
            count = len(pdf) if max_pages <= 0 else min(len(pdf), max_pages)
            for index in range(count):
                page = pdf[index]
                width_pt, height_pt = page.get_size()
                image = page.render(scale=dpi / 72).to_pil()
                image_path = pages_dir / page_filename(index + 1)
                image.save(image_path)
                doc.pages.append(
                    Page(index + 1, image_path, image.size, (width_pt, height_pt))
                )
        finally:
            pdf.close()
        return doc

    from PIL import Image

    image_path = pages_dir / page_filename(1)
    if source.suffix.lower() == ".png":
        shutil.copy2(source, image_path)
    else:
        Image.open(source).save(image_path)
    with Image.open(image_path) as image:
        size = image.size
    return Document(source=source, kind="image", pages=[Page(1, image_path, size)])
