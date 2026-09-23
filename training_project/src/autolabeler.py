"""Autolabeling pipeline: run the trained YOLO model over a folder of
unlabeled images, gate each candidate detection through an AutolabelDecider
(Jev or the threshold stub), and route each *image* (never single boxes, a
partially labeled image would teach YOLO that the missing box is background):

- every candidate accepted  -> train/ (image moved, YOLO label written)
- anything uncertain        -> review_queue/images/ + a Label Studio task
- no candidates at all      -> review too (sampled by review_no_detection_rate,
                               the rest go to skipped/), so schematics the
                               model misses still reach a human

Images are moved, not copied: whatever is left in the source folder has not
been processed yet, so re-running never produces duplicates.

Flagged candidates are written as Label Studio pre-annotated tasks
(review_queue/label_studio_tasks.json) rather than isolated crops, so a
human reviews them with full page context and can drag-correct the box
instead of just accepting/rejecting a cropped image. Import that file into
Label Studio (Apache-2.0, self-hosted) as "predictions"; the labeling
config's rectangle-label control tag names must match
config.LABEL_STUDIO_FROM_NAME / config.LABEL_STUDIO_TO_NAME (defaults:
"label" / "image", Label Studio's standard Object Detection template).

Run Label Studio via `uvx` (an isolated, ephemeral environment) rather than
installing it into this project's own venv -- it pulls in opencv-python-headless,
which conflicts with the opencv-python already required by easyocr/ultralytics/
docling-ibm-models (both provide the same top-level cv2/ package; having both
installed in one venv silently breaks cv2). Local file serving must be enabled
to load the referenced images, e.g.:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo root> uvx label-studio start
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from config.settings import default_config
from PIL import Image
from ultralytics import YOLO

from src.decisions import get_decider
from src.features import extract_evidence
from src.utils import setup_logging

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def local_files_url(image_path: Path, document_root: Path) -> str:
    """Label Studio local-files URL for an image under the document root (repo root)."""
    rel_path = Path(image_path).resolve().relative_to(document_root)
    return f"/data/local-files/?d={rel_path}"


def to_label_studio_result(
    result_id: str, xyxy, label: str, w_img: int, h_img: int, from_name: str, to_name: str
) -> dict:
    """One rectanglelabels region; Label Studio stores boxes as percentages."""
    x1, y1, x2, y2 = xyxy
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": from_name,
        "to_name": to_name,
        "original_width": w_img,
        "original_height": h_img,
        "image_rotation": 0,
        "value": {
            "x": x1 / w_img * 100,
            "y": y1 / h_img * 100,
            "width": (x2 - x1) / w_img * 100,
            "height": (y2 - y1) / h_img * 100,
            "rotation": 0,
            "rectanglelabels": [label],
        },
    }


class AutolabelPipeline:
    """Composes YOLO candidate generation, evidence extraction, and an
    AutolabelDecider into an end-to-end autolabeling run."""

    def __init__(self, config=None, decider=None, model_path=None):
        self.logger = setup_logging()
        self.config = config or default_config
        self.decider = decider or get_decider(self.config)

        weights = model_path or self.config.get_weights_path()
        if Path(weights).exists():
            self.model = YOLO(str(weights))
            self.class_names = self.model.names
        else:
            # First round of a new dataset (e.g. subcircuits): there is nothing to
            # pre-label with yet, so every image goes to review without boxes
            self.model = None
            self.class_names = {}
            self.logger.warning(f"No trained model at {weights}: sending all images to review")

        # Document root Label Studio's local-files serving must be pointed at
        self.document_root = self.config.PROJECT_ROOT.parent

        self.train_images_dir = self.config.TRAINING_DATA_PATH / "train" / "images"
        self.train_labels_dir = self.config.TRAINING_DATA_PATH / "train" / "labels"
        self.review_dir = self.config.TRAINING_DATA_PATH / "review_queue"
        self.review_images_dir = self.review_dir / "images"
        self.skipped_dir = self.config.TRAINING_DATA_PATH / "skipped"
        for d in (self.train_images_dir, self.train_labels_dir, self.review_images_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.review_no_detection_rate = getattr(self.config, "REVIEW_NO_DETECTION_RATE", 1.0)

        self.label_studio_from_name = getattr(self.config, "LABEL_STUDIO_FROM_NAME", "label")
        self.label_studio_to_name = getattr(self.config, "LABEL_STUDIO_TO_NAME", "image")

        self.logger.info(f"Autolabel decider: {type(self.decider).__name__}")

    def run(self, source_dir, candidate_conf: float = 0.1) -> dict:
        """candidate_conf is intentionally low: we want YOLO's raw candidates,
        the decider (not a hard confidence cutoff) makes the accept/reject
        call."""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        images = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        summary = {"to_train": 0, "to_review": 0, "no_detections_to_review": 0, "skipped": 0}
        review_tasks = []
        evidence_log = []
        run_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        if self.model is None:
            for image_path in images:
                moved = self._move(image_path, self.review_images_dir)
                with Image.open(moved) as img:
                    shape = (img.height, img.width)
                review_tasks.append(self._to_label_studio_task(moved, shape, []))
                summary["to_review"] += 1
            if review_tasks:
                self._write_review_tasks(review_tasks)
            return summary

        for image_path in images:
            results = self.model.predict(source=str(image_path), conf=candidate_conf, verbose=False)
            result = results[0]
            image = result.orig_img

            if len(result.boxes) == 0:
                if self._sample_no_detection(image_path):
                    moved = self._move(image_path, self.review_images_dir)
                    review_tasks.append(self._to_label_studio_task(moved, image.shape, []))
                    summary["no_detections_to_review"] += 1
                    self.logger.info(f"{image_path.name}: no detections -> review")
                else:
                    self._move(image_path, self.skipped_dir)
                    summary["skipped"] += 1
                continue

            verdicts = []
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]

                evidence = extract_evidence(image, xyxy, conf, cls_name)
                decision = self.decider.decide(evidence)
                verdicts.append((xyxy, cls_id, cls_name, decision))
                evidence_log.append(
                    {
                        "run_at": run_at,
                        "source_image": image_path.name,
                        "box_xyxy": xyxy,
                        "evidence": evidence,
                        "decision": {
                            "verdict": decision.verdict,
                            "score": decision.score,
                            "reason": decision.reason,
                        },
                    }
                )
                self.logger.info(
                    f"{image_path.name}: {cls_name} conf={conf:.2f} -> "
                    f"{decision.verdict} (score={decision.score:.2f}, {decision.reason})"
                )

            if all(d.verdict == "accept" for _, _, _, d in verdicts):
                yolo_lines = [
                    self._to_yolo_line(cls_id, xyxy, image.shape) for xyxy, cls_id, _, _ in verdicts
                ]
                self._accept_image(image_path, yolo_lines)
                summary["to_train"] += 1
            else:
                # Pre-draw everything that wasn't rejected; the reviewer confirms,
                # corrects or adds boxes for the whole image
                drawn = [
                    (xyxy, cls_name, decision)
                    for xyxy, _, cls_name, decision in verdicts
                    if decision.verdict != "reject"
                ]
                moved = self._move(image_path, self.review_images_dir)
                review_tasks.append(self._to_label_studio_task(moved, image.shape, drawn))
                summary["to_review"] += 1

        if evidence_log:
            self._write_evidence_log(evidence_log)
        if review_tasks:
            self._write_review_tasks(review_tasks)

        return summary

    def _sample_no_detection(self, image_path: Path) -> bool:
        """Deterministic sampling by filename, so reruns decide the same way."""
        bucket = int(hashlib.sha256(image_path.name.encode()).hexdigest(), 16) % 1000
        return bucket < self.review_no_detection_rate * 1000

    @staticmethod
    def _move(image_path: Path, target_dir: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / image_path.name
        shutil.move(image_path, dest)
        return dest

    @staticmethod
    def _to_yolo_line(cls_id: int, xyxy, image_shape) -> str:
        h_img, w_img = image_shape[:2]
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
        w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
        return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def _accept_image(self, image_path: Path, yolo_lines: list[str]):
        dest_image = self._move(image_path, self.train_images_dir)
        dest_label = self.train_labels_dir / f"{image_path.stem}.txt"
        dest_label.write_text("\n".join(yolo_lines) + "\n")
        self.logger.info(f"Accepted into training set: {dest_image.name}")

    def _to_label_studio_task(self, image_path: Path, image_shape, flagged: list) -> dict:
        """Build one Label Studio pre-annotated task for an image, covering
        every flagged box on it. See module docstring for the local-files
        serving setup this depends on."""
        h_img, w_img = image_shape[:2]
        task = {"data": {"image": local_files_url(image_path, self.document_root)}}
        if not flagged:
            # No pre-drawn boxes: the reviewer draws any schematic the model missed,
            # or submits empty to confirm the page as a background sample
            return task

        results = []
        avg_score = 0.0
        for i, (xyxy, cls_name, decision) in enumerate(flagged):
            results.append(
                to_label_studio_result(
                    f"result_{i}", xyxy, cls_name, w_img, h_img,
                    self.label_studio_from_name, self.label_studio_to_name,
                )
            )
            avg_score += decision.score
        avg_score /= len(flagged)

        task["predictions"] = [
            {
                "model_version": type(self.decider).__name__,
                "score": avg_score,
                "result": results,
            }
        ]
        return task

    def _write_review_tasks(self, review_tasks: list[dict]):
        """One task per image: new tasks replace existing ones for the same image."""
        tasks_path = self.review_dir / "label_studio_tasks.json"
        tasks = {}
        if tasks_path.exists():
            tasks = {t["data"]["image"]: t for t in json.loads(tasks_path.read_text())}
        for task in review_tasks:
            tasks[task["data"]["image"]] = task
        tasks_path.write_text(json.dumps(list(tasks.values()), indent=2))
        self.logger.info(f"Wrote {len(review_tasks)} review task(s) to {tasks_path}")

    def _write_evidence_log(self, evidence_log: list[dict]):
        evidence_path = self.review_dir / "evidence_log.jsonl"
        with evidence_path.open("a") as f:
            for entry in evidence_log:
                f.write(json.dumps(entry) + "\n")
