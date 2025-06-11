# ===========================================
# File: training_project/scripts/predict.py (Updated)
# ===========================================
#!/usr/bin/env python3
"""Prediction script for YOLO model"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

from src.predictor import YOLOPredictor


def main():
    parser = argparse.ArgumentParser(description="Run YOLO predictions")
    parser.add_argument(
        "source", nargs="?", help="Source image or directory, use '0' for local webcam"
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument(
        "--conf",
        type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help="Confidence threshold",
    )
    parser.add_argument("--model", help="Path to model weights")
    parser.add_argument(
        "--img-folder", action="store_true", help="Use main img folder as source"
    )

    args = parser.parse_args()

    try:
        predictor = YOLOPredictor(model_path=args.model)

        # Determine source
        if args.img_folder:
            source = Config.PROJECT_ROOT.parent / "img"
        elif args.source:
            source = args.source
        else:
            print("Please specify a source or use --img-folder flag")
            sys.exit(1)

        predictor.predict(source, save_dir=args.output, conf=args.conf)

    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
