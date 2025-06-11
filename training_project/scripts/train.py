# ===========================================
# File: training_project/scripts/train.py (MPS Enhanced)
# ===========================================
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
        choices=[10, 50, 100, 200, 300],
        metavar="EPOCHS",
    )

    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (-1 for auto)",
        choices=[-1, 8, 16, 32, 64],
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
        choices=[0, 2, 4, 8],
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

    # Enable argcomplete if available
    if ARGCOMPLETE_AVAILABLE:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # Rest of your main function...
    # [Previous main function code here]


if __name__ == "__main__":
    main()
