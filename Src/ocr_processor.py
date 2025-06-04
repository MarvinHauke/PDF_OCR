from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger
from pytesseract import Output
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def is_heading(text: str) -> bool:
    stripped = text.strip()
    return stripped.isupper() and len(stripped.split()) <= 10 and len(stripped) > 3


def process_pdf(pdf_path: Path, temp_dir: Path, output_dir: Path, output_pdf: Path):
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    print(f"Converting PDF pages to images from: [cyan]{pdf_path}[/cyan] ...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Converting pages...", total=None)
        images = convert_from_path(str(pdf_path), dpi=300, output_folder=str(temp_dir))
        progress.update(task, completed=len(images))
        progress.stop()

    print("Performing OCR and annotating pages...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Annotating pages...", total=len(images))
        for i, img in enumerate(images):
            text_data = pytesseract.image_to_data(img, output_type=Output.DICT)
            pdf_page_path = output_dir / f"page_{i+1}.pdf"
            c = canvas.Canvas(str(pdf_page_path), pagesize=letter)
            width, height = letter

            for j, word in enumerate(text_data["text"]):
                if word.strip() == "":
                    continue

                x = text_data["left"][j] * 0.75
                y = height - text_data["top"][j] * 0.75

                font_size = 8
                c.setFont("Helvetica", font_size)

                if is_heading(word):
                    c.setFillColorRGB(1, 0, 0)
                    c.setFont("Helvetica-Bold", 10)
                else:
                    c.setFillColorRGB(0, 0, 0)

                c.drawString(x, y, word)

            c.save()
            progress.update(task, advance=1)

    print("Merging annotated pages into one PDF...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Merging pages...", total=len(images))
        merger = PdfMerger()
        for i in range(len(images)):
            pdf_page_path = output_dir / f"page_{i+1}.pdf"
            merger.append(str(pdf_page_path))
            progress.update(task, advance=1)
        merger.write(str(output_pdf))
        merger.close()

    print(f"✅ Output saved as: [green]{output_pdf}[/green]")
