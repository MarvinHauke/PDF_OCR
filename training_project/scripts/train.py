# ===========================================
# File: training_project/scripts/train.py (Updated)
# ===========================================
#!/usr/bin/env python3
"""Training script for YOLO model"""

import argparse
import sys
from pathlib import Path

# Add project root to path - adjusted for your structure
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

from src.trainer import YOLOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument(
        "--epochs", type=int, default=Config.EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=Config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--device", type=str, default=Config.DEVICE, help="Device to use"
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume")
    parser.add_argument(
        "--model", type=str, default=Config.MODEL_ARCH, help="Model architecture"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=Config.RUN_NAME,
        help="Optional name for this training run",
    )

    args = parser.parse_args()

    # Override config with command line arguments
    class RuntimeConfig(Config):
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        DEVICE = args.device
        MODEL_ARCH = args.model
        RUN_NAME = args.run_name

    try:
        trainer = YOLOTrainer(config=RuntimeConfig)
        trainer.train(resume_if_possible=not args.no_resume)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
