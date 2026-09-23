"""Cheap, non-ML evidence extraction for a YOLO detection candidate.

Builds the structured `state` payload the autolabel decider (Jev or the
threshold stub) reasons over. No vision-model or LLM calls here -- just
geometry and classical CV descriptors, all already in the project's
dependency stack.
"""

import cv2
import numpy as np


def extract_evidence(image: np.ndarray, box_xyxy, yolo_confidence: float, yolo_class: str, nearby_text: str = "") -> dict:
    """
    image: full-size image as a numpy array (BGR, as returned by ultralytics'
        Results.orig_img)
    box_xyxy: (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    crop = image[y1:y2, x1:x2]

    w, h = x2 - x1, y2 - y1
    edge_density = 0.0
    line_count = 0
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)
        line_count = 0 if lines is None else len(lines)

    return {
        "yolo_confidence": round(float(yolo_confidence), 4),
        "yolo_class": yolo_class,
        "box_width_px": w,
        "box_height_px": h,
        "aspect_ratio": round(w / h, 3) if h else None,
        "area_fraction_of_image": round((w * h) / (w_img * h_img), 4) if w_img and h_img else None,
        "edge_density": round(edge_density, 4),
        "line_count": line_count,
        "nearby_text": nearby_text,
    }
