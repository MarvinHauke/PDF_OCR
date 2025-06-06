import os

import easyocr
import numpy as np
from IPython.display import display
from pdf2image import convert_from_path
from PIL import Image

# Path to your PDF
pdf_path = "examples/CEM33403345-VCO.pdf"

# Convert PDF to images
pages = convert_from_path(pdf_path, dpi=600)  # Higher DPI = better OCR accuracy

for i, page in enumerate(pages):
    print(f"--- Page {i+1} ---")
    display(page)


# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])  # You can add more languages, e.g. ['en', 'de']

# Process each page
for page_num, image in enumerate(pages):
    print(f"--- Page {page_num + 1} ---")
    results = reader.readtext(np.array(image))  # Convert PIL image to numpy array

    for bbox, text, confidence in results:
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Box: {bbox}\n")
