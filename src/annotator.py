# src/annotator.py

from io import BytesIO
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.colors import blue, red
from reportlab.pdfgen import canvas


def annotate_pdf(doc, input_pdf_path: Path, output_pdf_path: Path):
    """Annotate PDF pages with layout section bounding boxes and labels."""
    reader = PdfReader(str(input_pdf_path))
    writer = PdfWriter()

    for page_index, pdf_page in enumerate(reader.pages):
        layout_page = doc._.layout.pages[page_index]
        page_width = layout_page.width
        page_height = layout_page.height

        # Create overlay
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))

        for section in doc._.pages[page_index][1]:
            box = section._.layout

            x, y, w, h = box.x, box.y, box.width, box.height

            # Draw rectangle and label
            can.setStrokeColor(blue)
            can.setLineWidth(1)
            can.rect(x, y, w, h, stroke=1, fill=0)

            can.setFillColor(red)
            can.setFont("Helvetica", 8)
            can.drawString(x, y + h + 2, section.label_)

        can.save()
        packet.seek(0)

        # Merge overlay with original page
        overlay_pdf = PdfReader(packet)
        overlay_page = overlay_pdf.pages[0]
        pdf_page.merge_page(overlay_page)
        writer.add_page(pdf_page)

    # Write final output
    with open(output_pdf_path, "wb") as f:
        writer.write(f)
