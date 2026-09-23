"""PDF/image input -> rendered pages -> YOLO detections -> per-document output.

Output layout, one folder per input document (overwritten on rerun):
    <output>/<stem>/pages/page-NNN.png      rendered pages
    <output>/<stem>/annotated/page-NNN.png  pages with detections drawn
    <output>/<stem>/detections.json
"""

import json
import shutil
from pathlib import Path

from pdf_ocr.detect import Detector
from pdf_ocr.sources import Document, collect_sources, prepare_document


def run(
    input_path: Path,
    output_dir: Path,
    dpi: int = 200,
    conf: float | None = None,
    model_path: str | None = None,
) -> list[dict]:
    sources = collect_sources(input_path)
    if not sources:
        print(f"No PDFs or images found in {input_path}")
        return []

    detector = Detector(model_path=model_path, conf=conf)
    summaries = []

    for source in sources:
        doc_dir = output_dir / source.stem
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

        doc = prepare_document(source, doc_dir / "pages", dpi)
        results = detector.detect([page.image_path for page in doc.pages])
        report = _write_outputs(doc, results, doc_dir, detector.model_path)

        n_detections = sum(len(p["detections"]) for p in report["pages"])
        print(f"{source.name}: {len(doc.pages)} page(s), {n_detections} detection(s) -> {doc_dir}")
        summaries.append(report)

    return summaries


def _write_outputs(doc: Document, results, doc_dir: Path, model_path: Path) -> dict:
    annotated_dir = doc_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    for page, result in zip(doc.pages, results):
        # Per-page ratio rather than 72/dpi: the rendered pixel size is rounded
        scale = (
            (page.size_pt[0] / page.size_px[0], page.size_pt[1] / page.size_px[1])
            if page.size_pt
            else None
        )
        result.save(filename=str(annotated_dir / page.image_path.name))

        detections = []
        for box in result.boxes:
            xyxy = [round(v, 2) for v in box.xyxy[0].tolist()]
            detection = {
                "class": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "bbox_px": xyxy,
            }
            if scale:
                # PDF points, origin top-left (same orientation as the image)
                detection["bbox_pt"] = [round(v * scale[i % 2], 2) for i, v in enumerate(xyxy)]
            detections.append(detection)

        pages.append(
            {
                "page": page.number,
                "image": str(page.image_path.relative_to(doc_dir)),
                "size_px": list(page.size_px),
                "size_pt": [round(v, 2) for v in page.size_pt] if page.size_pt else None,
                "detections": detections,
            }
        )

    report = {
        "source": str(doc.source),
        "kind": doc.kind,
        "dpi": doc.dpi,
        "model": str(model_path),
        "pages": pages,
    }
    with open(doc_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

