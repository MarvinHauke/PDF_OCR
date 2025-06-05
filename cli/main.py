import typer
from rich.progress import Progress

from core.ocr_pipeline import process_pdf

app = typer.Typer()


@app.command()
def annotate(
    input: str = typer.Option(..., help="Input PDF path"),
    output: str = typer.Option(..., help="Output annotated PDF path"),
    toc_json: str = typer.Option(..., help="TOC JSON path"),
    language: str = typer.Option("en", help="OCR language"),
):
    with Progress() as progress:
        task = progress.add_task("Processing PDF...", total=100)
        process_pdf(input, output, toc_json, language, progress, task)


if __name__ == "__main__":
    app()
