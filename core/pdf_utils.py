import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image


def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)


def save_annotated_pdf(images, annotations, output_path):
    doc = fitz.open()
    for img, boxes in zip(images, annotations):
        pix = fitz.Pixmap(
            fitz.csRGB, fitz.open("png", img.tobytes()).get_page_pixmap(0)
        )
        page = doc.new_page(width=pix.width, height=pix.height)
        page.insert_image(page.rect, pixmap=pix)
        for box in boxes:
            rect = fitz.Rect(*box[:4])
            page.draw_rect(rect, color=(1, 0, 0), width=1)
            page.insert_text(rect.tl, box[4], fontsize=6, color=(0, 0, 1))
    doc.save(output_path)
