#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Autolabel a folder of images using the trained YOLO model plus a Jev (or
threshold-stub) decider, routing candidates into the training set, a human
review queue, or discarding them.

Uses the threshold stub decider unless TYPESAFE_API_KEY is set in the
environment -- see training_project/src/decisions.py.
"""

import argparse
import sys
from pathlib import Path

import argcomplete

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

from src.autolabeler import AutolabelPipeline
from src.utils import config_file_completer


def main():
    parser = argparse.ArgumentParser(
        description="Autolabel a folder of images using YOLO + Jev (or the threshold stub)"
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Folder of unlabeled images to autolabel (default: training_data/unlabeled/)",
    )
    parser.add_argument(
        "--config", help="Configuration YAML (default: config/config.yaml)"
    ).completer = config_file_completer
    parser.add_argument(
        "--model", help="Path to model weights"
    ).completer = argcomplete.completers.FilesCompleter()
    parser.add_argument(
        "--candidate-conf",
        type=float,
        default=0.1,
        help="Minimum YOLO confidence to consider a candidate at all "
        "(kept low on purpose -- the decider does the real filtering)",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    try:
        config = Config(config_file=args.config) if args.config else None
        pipeline = AutolabelPipeline(config=config, model_path=args.model)
        source = args.source or pipeline.config.UNLABELED_PATH
        summary = pipeline.run(source, candidate_conf=args.candidate_conf)

        print("\n=== Autolabel summary ===")
        for verdict, count in summary.items():
            print(f"  {verdict}: {count}")
        print(f"\nAccepted images/labels: {pipeline.train_images_dir}")
        print(f"Review queue: {pipeline.review_dir}")

    except Exception as e:
        print(f"Autolabeling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
