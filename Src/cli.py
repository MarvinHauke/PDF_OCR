from pathlib import Path

import typer

from ocr_processor import process_pdf

app = typer.Typer()


@app.command()
def run(
    pdf_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the PDF file to process.",
    )
):
    process_pdf(
        pdf_path=pdf_path,
        temp_dir=Path("temp_pages"),
        output_dir=Path("output_pages"),
        output_pdf=Path("outputs/annotated_output.pdf"),
    )


if __name__ == "__main__":
    app()
