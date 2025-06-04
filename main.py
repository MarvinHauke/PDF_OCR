import os

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger
from pytesseract import Output
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

INPUT_PDF = "typed_input.pdf"
TEMP_DIR = "temp_pages"
OUTPUT_DIR = "output_pages"
OUTPUT_PDF = "annotated_output.pdf"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert PDF to images
images = convert_from_path(INPUT_PDF, dpi=300, output_folder=TEMP_DIR)


def is_heading(text):
    stripped = text.strip()
    return stripped.isupper() and len(stripped.split()) <= 10 and len(stripped) > 3


# OCR and annotate each page
for i, img in enumerate(images):
    text_data = pytesseract.image_to_data(img, output_type=Output.DICT)

    pdf_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Draw original text
    for j, word in enumerate(text_data["text"]):
        if word.strip() == "":
            continue

        x = text_data["left"][j] * 0.75  # scale to PDF units
        y = height - text_data["top"][j] * 0.75

        font_size = 8
        c.setFont("Helvetica", font_size)

        # Highlight headings in bold red
        if is_heading(word):
            c.setFillColorRGB(1, 0, 0)
            c.setFont("Helvetica-Bold", 10)
        else:
            c.setFillColorRGB(0, 0, 0)

        c.drawString(x, y, word)

    c.save()

# Merge all annotated pages
merger = PdfMerger()
for i in range(len(images)):
    pdf_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.pdf")
    merger.append(pdf_path)

merger.write(OUTPUT_PDF)
merger.close()

print(f"\n✅ Output saved as: {OUTPUT_PDF}")
