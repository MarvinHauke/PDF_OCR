#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Prediction script for YOLO model"""

import argparse
import sys
from pathlib import Path

import argcomplete

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config, default_config

from src.predictor import YOLOPredictor
from src.utils import config_file_completer


def main():
    parser = argparse.ArgumentParser(description="Run YOLO predictions")
    parser.add_argument(
        "source", nargs="?", help="Source image or directory, use '0' for local webcam"
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--conf",
        type=float,
        help="Confidence threshold (default: from config)",
    )
    parser.add_argument(
        "--config", help="Configuration YAML (default: config/config.yaml)"
    ).completer = config_file_completer
    parser.add_argument(
        "--model", help="Path to model weights"
    ).completer = argcomplete.completers.FilesCompleter()
    parser.add_argument(
        "--img-folder", action="store_true", help="Use training_data/unlabeled/ (unlabeled images) as source"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show a live annotated window (needed for webcam)"
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        config = Config(config_file=args.config) if args.config else default_config
        predictor = YOLOPredictor(config=config, model_path=args.model)

        # Determine source
        if args.img_folder:
            source = config.UNLABELED_PATH
        elif args.source:
            # Ultralytics expects an int index for webcam sources (e.g. "0")
            source = int(args.source) if args.source.isdigit() else args.source
        else:
            print("Please specify a source or use --img-folder flag")
            sys.exit(1)

        predictor.predict(source, save_dir=args.output, conf=args.conf, show=args.show)

    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
