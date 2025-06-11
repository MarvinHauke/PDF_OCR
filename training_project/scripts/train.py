#!/usr/bin/env python3
"""Training script for YOLO model with autocompletion support"""

import argparse
import sys
from pathlib import Path

import torch

# Add argcomplete import
try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import Config

from src.trainer import YOLOTrainer


def config_file_completer(prefix, parsed_args, **kwargs):
    """Completer for configuration files"""
    config_dir = PROJECT_ROOT / "config"
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml"))
        return [
            f"config/{f.name}"
            for f in yaml_files
            if f.name.startswith(prefix.split("/")[-1])
        ]
    return []


def model_completer(prefix, parsed_args, **kwargs):
    """Completer for model architectures"""
    models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    return [m for m in models if m.startswith(prefix)]


def run_name_completer(prefix, parsed_args, **kwargs):
    """Completer for run names based on existing runs"""
    runs_dir = PROJECT_ROOT / "training_data" / "runs"
    if runs_dir.exists():
        existing_runs = [d.name for d in runs_dir.iterdir() if d.is_dir()]
        return [name for name in existing_runs if name.startswith(prefix)]
    return ["run", "experiment", "test"]


def check_mps_availability():
    """Check and report MPS availability"""
    print("=== MPS Availability Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print("✅ MPS is available and ready for training")
    else:
        if not torch.backends.mps.is_built():
            print("❌ MPS not available: PyTorch not built with MPS support")
        else:
            print("❌ MPS not available: macOS version < 12.3 or no MPS-enabled device")
    print("=" * 30)


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model with MPS optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check-mps
  %(prog)s --config config/mps_optimized.yaml
  %(prog)s --device auto --optimize-for-mps
  %(prog)s --model yolo11n.pt --epochs 10 --device mps
        """,
    )

    parser.add_argument(
        "--config", type=str, help="Path to configuration YAML file"
    ).completer = config_file_completer

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        metavar="EPOCHS",
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (-1 for auto)",
        metavar="BATCH",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device to use (auto will select best available)",
    )

    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume")

    parser.add_argument("--model", type=str, help="Model architecture").completer = (
        model_completer
    )

    parser.add_argument(
        "--run-name", type=str, help="Name for this training run"
    ).completer = run_name_completer

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of dataloader workers",
        metavar="WORKERS",
    )

    parser.add_argument(
        "--amp", action="store_true", help="Enable Automatic Mixed Precision"
    )

    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision",
    )

    parser.add_argument(
        "--check-mps", action="store_true", help="Check MPS availability and exit"
    )

    parser.add_argument(
        "--optimize-for-mps",
        action="store_true",
        help="Apply aggressive MPS optimizations",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Enable argcomplete if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # Check MPS availability if requested
    if args.check_mps:
        check_mps_availability()
        return

    try:
        # Load configuration
        config = Config(config_file=args.config) if args.config else Config()

        # Auto-select device if requested
        if args.device == "auto":
            if torch.backends.mps.is_available():
                args.device = "mps"
                print("🍎 Auto-selected MPS device (Apple Silicon)")
            elif torch.cuda.is_available():
                args.device = "cuda"
                print("🚀 Auto-selected CUDA device")
            else:
                args.device = "cpu"
                print("💻 Auto-selected CPU device")

        # Apply MPS optimizations if requested
        if args.optimize_for_mps and (args.device == "mps" or config.DEVICE == "mps"):
            print("🔧 Applying aggressive MPS optimizations...")
            # Override config for maximum MPS compatibility
            config.WORKERS = 0
            config.BATCH_SIZE = -1  # Auto batch size
            config.AMP = True
            config.HALF = False
            config.CACHE = False
            config.PIN_MEMORY = False
            config.PERSISTENT_WORKERS = False
            config.MULTI_SCALE = False
            config.DETERMINISTIC = False

        # Override with command line arguments
        if args.epochs is not None:
            config.EPOCHS = args.epochs
        if args.batch is not None:
            config.BATCH_SIZE = args.batch
        if args.device is not None:
            config.DEVICE = args.device
        if args.model is not None:
            config.MODEL_ARCH = args.model
        if args.run_name is not None:
            config.RUN_NAME = args.run_name
        if args.workers is not None:
            config.WORKERS = args.workers
        if args.amp is not None:
            config.AMP = args.amp

        # Print final configuration
        if args.verbose:
            check_mps_availability()
            print("\n=== Final Configuration ===")
            print(f"Device: {config.DEVICE}")
            print(f"Model: {config.MODEL_ARCH}")
            print(f"Epochs: {config.EPOCHS}")
            print(f"Batch Size: {config.BATCH_SIZE}")
            print(f"Workers: {config.WORKERS}")
            print(f"AMP: {config.AMP}")
            print(f"Run Name: {config.RUN_NAME}")
            print("=" * 30)

        print(f"🚀 Starting YOLO training...")
        print(f"   Model: {config.MODEL_ARCH}")
        print(f"   Device: {config.DEVICE}")
        print(f"   Epochs: {config.EPOCHS}")
        print(f"   Batch: {config.BATCH_SIZE}")
        print(f"   Run: {config.RUN_NAME}")

        trainer = YOLOTrainer(config=config)
        results = trainer.train(resume_if_possible=not args.no_resume)

        if results:
            print("✅ Training completed successfully!")
        else:
            print("ℹ️ Training skipped (already completed or resumed)")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        # Clean up MPS cache if needed
        if torch.backends.mps.is_available():
            try:
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                print("🧹 Cleaned up MPS cache")
            except:
                pass
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
