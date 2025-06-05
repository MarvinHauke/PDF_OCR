import json

import cv2
import numpy as np
import pytesseract
from PIL import Image

from core.pdf_utils import pdf_to_images, save_annotated_pdf
from core.toc_generator import generate_toc


def process_pdf(input_path, output_path, toc_path, language, progress, task):
    images = pdf_to_images(input_path)
    all_boxes = []
    toc_entries = []
    total = len(images)

    for i, image in enumerate(images):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        text_data = pytesseract.image_to_data(
            gray, lang=language, output_type=pytesseract.Output.DICT
        )

        boxes = []
        for j in range(len(text_data["text"])):
            if int(text_data["conf"][j]) > 60:
                (x, y, w, h) = (
                    text_data["left"][j],
                    text_data["top"][j],
                    text_data["width"][j],
                    text_data["height"][j],
                )
                word = text_data["text"][j]
                boxes.append([x, y, x + w, y + h, word])
                if word.isupper() and len(word) > 5:
                    toc_entries.append({"page": i + 1, "title": word})
        all_boxes.append(boxes)
        progress.update(task, advance=100 / total)

    save_annotated_pdf(images, all_boxes, output_path)
    with open(toc_path, "w") as f:
        json.dump(generate_toc(toc_entries), f, indent=2)
