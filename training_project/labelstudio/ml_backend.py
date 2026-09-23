#!/usr/bin/env python3
"""Minimal Label Studio ML backend serving this project's YOLO models.

Label Studio calls it to pre-label tasks. It implements the small HTTP
protocol Label Studio's ML client speaks (label_studio/ml/api_connector.py):

    GET  /health    -> {"status": "UP"}
    POST /setup     -> {"model_version": ...}
    POST /predict   {tasks, label_config, ...} -> {"results": [one prediction per task]}
    POST /webhook   (training events) -> ignored, training stays manual (train.py)

Why not the official label-studio-ml package: it depends on label-studio-sdk,
which requires opencv-python-headless, and that breaks cv2 next to the
opencv-python ultralytics needs. This server only needs the stdlib and
ultralytics, so it runs in the project's own venv (with MPS):

    uv run python training_project/labelstudio/ml_backend.py --port 9090

The model is picked per project from its labeling config: the page model for
<Label value="schematic"/>, the subcircuit model (config/subcircuits.yaml)
for the functional block labels. Images are read straight from disk: task
URLs are /data/local-files/?d=<repo-relative path>, and this server runs on
the same machine as the repo.
"""

import argparse
import json
import logging
import sys
import threading
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.autolabeler import to_label_studio_result  # noqa: E402

REPO_ROOT = PROJECT_ROOT.parent
DATASET_CONFIGS = [None, "config/subcircuits.yaml"]  # None = config/config.yaml

logger = logging.getLogger("ml_backend")


class ModelRegistry:
    """Loads each dataset's weights on first use and reloads them after retraining."""

    def __init__(self):
        self.configs = [Config(config_file=c) if c else Config() for c in DATASET_CONFIGS]
        self._cache = {}  # weights path -> (mtime, YOLO)
        self._lock = threading.Lock()  # ultralytics models aren't thread-safe

    def pick(self, labels: set[str]):
        """(config, weights) of the dataset whose class names overlap the labels most."""
        best, best_overlap = None, 0
        for config in self.configs:
            names = set(self._class_names(config))
            overlap = len(names & labels)
            if overlap > best_overlap:
                best, best_overlap = config, overlap
        if best is None:
            return None, None
        weights = Path(best.get_weights_path())
        return (best, weights) if weights.exists() else (best, None)

    @staticmethod
    def _class_names(config) -> list[str]:
        import yaml

        if not config.YAML_PATH.exists():
            return []
        names = yaml.safe_load(config.YAML_PATH.read_text()).get("names", [])
        return list(names.values()) if isinstance(names, dict) else list(names)

    def model(self, weights: Path) -> YOLO:
        mtime = weights.stat().st_mtime
        cached = self._cache.get(weights)
        if cached is None or cached[0] != mtime:
            logger.info(f"Loading {weights}")
            self._cache[weights] = (mtime, YOLO(str(weights)))
        return self._cache[weights][1]

    def predict(self, weights: Path, image_path: Path, conf: float, device: str):
        with self._lock:
            return self.model(weights).predict(
                source=str(image_path), conf=conf, device=device, verbose=False
            )[0]


def model_version(weights: Path | None) -> str:
    if weights is None:
        return "no-model"
    return f"{weights.parent.parent.name}/{weights.name}@{int(weights.stat().st_mtime)}"


def parse_label_config(label_config: str):
    """(from_name, to_name, image data key, label values) of the first RectangleLabels."""
    root = ET.fromstring(label_config)
    rect = root.find(".//RectangleLabels")
    if rect is None:
        return None
    to_name = rect.get("toName")
    image = next((e for e in root.iter("Image") if e.get("name") == to_name), None)
    data_key = image.get("value", "$image").lstrip("$") if image is not None else "image"
    labels = {label.get("value") for label in rect.iter("Label")}
    return rect.get("name"), to_name, data_key, labels


def resolve_image(url: str) -> Path | None:
    """/data/local-files/?d=<repo-relative path> -> file in the repo, if it exists."""
    rel = parse_qs(urlparse(url).query).get("d")
    if not rel:
        return None
    path = (REPO_ROOT / rel[0]).resolve()
    if REPO_ROOT not in path.parents or not path.exists():
        return None
    return path


class Handler(BaseHTTPRequestHandler):
    registry: ModelRegistry

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._send({"status": "UP", "model_class": "YOLO"})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        route = urlparse(self.path).path.rstrip("/")

        if route == "/setup":
            parsed = parse_label_config(body.get("schema") or "<View/>")
            weights = self.registry.pick(parsed[3])[1] if parsed else None
            self._send({"model_version": model_version(weights)})
        elif route == "/predict":
            self._send({"results": self._predict(body)})
        elif route in ("/webhook", "/train"):
            self._send({})  # training is done with train.py, not from Label Studio
        else:
            self._send({"error": "not found"}, status=404)

    def _predict(self, body: dict) -> list[dict]:
        tasks = body.get("tasks", [])
        parsed = parse_label_config(body.get("label_config") or "<View/>")
        if parsed is None:
            return [{"result": []} for _ in tasks]
        from_name, to_name, data_key, labels = parsed
        config, weights = self.registry.pick(labels)
        version = model_version(weights)

        results = []
        for task in tasks:
            image_path = resolve_image(task.get("data", {}).get(data_key, ""))
            if weights is None or image_path is None:
                # Label Studio needs one entry per task, even if it's empty
                results.append({"result": [], "model_version": version})
                continue

            detection = self.registry.predict(
                weights, image_path, config.CONFIDENCE_THRESHOLD, config.DEVICE
            )
            h_img, w_img = detection.orig_shape
            regions, scores = [], []
            for i, box in enumerate(detection.boxes):
                label = detection.names[int(box.cls[0])]
                if label not in labels:
                    continue
                regions.append(
                    to_label_studio_result(
                        f"yolo_{i}", box.xyxy[0].tolist(), label, w_img, h_img, from_name, to_name
                    )
                )
                scores.append(float(box.conf[0]))
            results.append(
                {
                    "result": regions,
                    "score": sum(scores) / len(scores) if scores else 0.0,
                    "model_version": version,
                }
            )
            logger.info(f"{image_path.name}: {len(regions)} box(es) with {version}")
        return results

    def _send(self, payload: dict, status: int = 200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        logger.debug(fmt % args)


def main():
    parser = argparse.ArgumentParser(description="Label Studio ML backend for the YOLO models")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    Handler.registry = ModelRegistry()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(f"ML backend listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
