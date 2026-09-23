"""Autolabeling pipeline: run the trained YOLO model over a folder of
unlabeled images, gate each candidate detection through an AutolabelDecider
(Jev or the threshold stub), and route it to the training set, a Label
Studio review queue, or discard it.

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

import json
import shutil
from pathlib import Path

from config.settings import default_config
from ultralytics import YOLO

from src.decisions import get_decider
from src.features import extract_evidence
from src.utils import setup_logging

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class AutolabelPipeline:
    """Composes YOLO candidate generation, evidence extraction, and an
    AutolabelDecider into an end-to-end autolabeling run."""

    def __init__(self, config=None, decider=None, model_path=None):
        self.logger = setup_logging()
        self.config = config or default_config
        self.decider = decider or get_decider(self.config)

        weights = model_path or self.config.get_weights_path()
        if not Path(weights).exists():
            raise FileNotFoundError(f"No trained model found at {weights}")
        self.model = YOLO(str(weights))
        self.class_names = self.model.names

        # Document root Label Studio's local-files serving must be pointed at
        self.document_root = self.config.PROJECT_ROOT.parent

        self.train_images_dir = self.config.TRAINING_DATA_PATH / "train" / "images"
        self.train_labels_dir = self.config.TRAINING_DATA_PATH / "train" / "labels"
        self.review_dir = self.config.TRAINING_DATA_PATH / "review_queue"
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)

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
        summary = {"accept": 0, "flag_for_review": 0, "reject": 0, "no_detections": 0}
        review_tasks = []
        evidence_log = []

        for image_path in images:
            results = self.model.predict(source=str(image_path), conf=candidate_conf, verbose=False)
            result = results[0]
            image = result.orig_img

            if len(result.boxes) == 0:
                summary["no_detections"] += 1
                continue

            yolo_lines = []
            flagged = []
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names[cls_id]

                evidence = extract_evidence(image, xyxy, conf, cls_name)
                decision = self.decider.decide(evidence)
                summary[decision.verdict] += 1
                evidence_log.append(
                    {
                        "source_image": str(image_path),
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

                if decision.verdict == "accept":
                    yolo_lines.append(self._to_yolo_line(cls_id, xyxy, image.shape))
                elif decision.verdict == "flag_for_review":
                    flagged.append((xyxy, cls_name, decision))
                # reject: discarded, nothing written

            if yolo_lines:
                self._accept_image(image_path, yolo_lines)
            if flagged:
                review_tasks.append(self._to_label_studio_task(image_path, image.shape, flagged))

        if evidence_log:
            self._write_evidence_log(evidence_log)
        if review_tasks:
            self._write_review_tasks(review_tasks)

        return summary

    @staticmethod
    def _to_yolo_line(cls_id: int, xyxy, image_shape) -> str:
        h_img, w_img = image_shape[:2]
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
        w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
        return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def _accept_image(self, image_path: Path, yolo_lines: list[str]):
        dest_image = self.train_images_dir / image_path.name
        shutil.copy2(image_path, dest_image)
        dest_label = self.train_labels_dir / f"{image_path.stem}.txt"
        dest_label.write_text("\n".join(yolo_lines) + "\n")
        self.logger.info(f"Accepted into training set: {dest_image.name}")

    def _to_label_studio_task(self, image_path: Path, image_shape, flagged: list) -> dict:
        """Build one Label Studio pre-annotated task for an image, covering
        every flagged box on it. See module docstring for the local-files
        serving setup this depends on."""
        h_img, w_img = image_shape[:2]
        rel_path = image_path.resolve().relative_to(self.document_root)

        results = []
        avg_score = 0.0
        for i, (xyxy, cls_name, decision) in enumerate(flagged):
            x1, y1, x2, y2 = xyxy
            results.append(
                {
                    "id": f"result_{i}",
                    "type": "rectanglelabels",
                    "from_name": self.label_studio_from_name,
                    "to_name": self.label_studio_to_name,
                    "original_width": w_img,
                    "original_height": h_img,
                    "image_rotation": 0,
                    "value": {
                        "x": x1 / w_img * 100,
                        "y": y1 / h_img * 100,
                        "width": (x2 - x1) / w_img * 100,
                        "height": (y2 - y1) / h_img * 100,
                        "rotation": 0,
                        "rectanglelabels": [cls_name],
                    },
                }
            )
            avg_score += decision.score
        avg_score /= len(flagged)

        return {
            "data": {"image": f"/data/local-files/?d={rel_path}"},
            "predictions": [
                {
                    "model_version": type(self.decider).__name__,
                    "score": avg_score,
                    "result": results,
                }
            ],
        }

    def _write_review_tasks(self, review_tasks: list[dict]):
        tasks_path = self.review_dir / "label_studio_tasks.json"
        existing = []
        if tasks_path.exists():
            existing = json.loads(tasks_path.read_text())
        tasks_path.write_text(json.dumps(existing + review_tasks, indent=2))
        self.logger.info(f"Wrote {len(review_tasks)} review task(s) to {tasks_path}")

    def _write_evidence_log(self, evidence_log: list[dict]):
        evidence_path = self.review_dir / "evidence_log.jsonl"
        with evidence_path.open("a") as f:
            for entry in evidence_log:
                f.write(json.dumps(entry) + "\n")
