# ===========================================
# File: training_project/scripts/train.py (Updated)
# ===========================================
#!/usr/bin/env python3
#!/usr/bin/env python3
"""Training script for YOLO model with stability optimizations"""
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

    # Core training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs (reduced for stability)",
    )
    parser.add_argument(
        "--batch", type=int, default=Config.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--device", type=str, default=Config.DEVICE, help="Device to use"
    )
    parser.add_argument(
        "--imgsz", type=int, default=320, help="Image size (reduced for speed)"
    )

    # Learning rate and optimization
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0005,
        help="Initial learning rate (reduced for stability)",
    )
    parser.add_argument(
        "--lrf", type=float, default=0.1, help="Final learning rate factor"
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        default=True,
        help="Use cosine learning rate scheduling",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=5.0,
        help="Warmup epochs (increased for stability)",
    )

    # Regularization parameters
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay for regularization",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Early stopping and monitoring
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (reduced)"
    )
    parser.add_argument(
        "--save-period", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Speed optimizations
    parser.add_argument(
        "--cache", action="store_true", default=Config.CACHE, help="Cache dataset"
    )
    parser.add_argument(
        "--rect", action="store_true", default=True, help="Rectangular training"
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic augmentation (disabled for speed)",
    )

    # Model and run configuration
    parser.add_argument(
        "--model", type=str, default=Config.MODEL_ARCH, help="Model architecture"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=Config.RUN_NAME,
        help="Optional name for this training run",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable auto-resume")

    # Advanced options
    parser.add_argument(
        "--val", action="store_true", default=True, help="Run validation"
    )
    parser.add_argument(
        "--plots", action="store_true", default=True, help="Generate training plots"
    )
    parser.add_argument(
        "--amp", action="store_true", default=True, help="Automatic mixed precision"
    )
    parser.add_argument(
        "--workers", type=int, default=Config.WORKERS, help="Number of workers"
    )

    # Quick presets for different training modes
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast experimentation mode (very small settings)",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Stable training mode (conservative settings)",
    )

    args = parser.parse_args()

    # Apply presets
    if args.fast:
        print("🚀 Fast experimentation mode enabled")
        args.epochs = 10
        args.imgsz = 320
        args.batch = 64
        args.lr0 = 0.0001
        args.patience = 3

    if args.stable:
        print("🎯 Stable training mode enabled")
        args.epochs = 20
        args.lr0 = 0.0001
        args.warmup_epochs = 8.0
        args.patience = 8
        args.weight_decay = 0.002

    # Override config with command line arguments
    class RuntimeConfig(Config):
        EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        DEVICE = args.device
        MODEL_ARCH = args.model
        RUN_NAME = args.run_name
        IMGSZ = args.imgsz
        LR0 = args.lr0
        LRF = args.lrf
        COS_LR = args.cos_lr
        WARMUP_EPOCHS = args.warmup_epochs
        WEIGHT_DECAY = args.weight_decay
        DROPOUT = args.dropout
        PATIENCE = args.patience
        SAVE_PERIOD = args.save_period
        CACHE = args.cache
        RECT = args.rect
        MOSAIC = args.mosaic
        VAL = args.val
        PLOTS = args.plots
        AMP = args.amp
        WORKERS = args.workers

    # Print configuration summary
    print("=" * 50)
    print("🔧 TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Model: {RuntimeConfig.MODEL_ARCH}")
    print(f"Image Size: {RuntimeConfig.IMGSZ}px")
    print(f"Batch Size: {RuntimeConfig.BATCH_SIZE}")
    print(f"Epochs: {RuntimeConfig.EPOCHS}")
    print(f"Learning Rate: {RuntimeConfig.LR0}")
    print(f"Device: {RuntimeConfig.DEVICE}")
    print(f"Cache Enabled: {RuntimeConfig.CACHE}")
    print(f"Patience: {RuntimeConfig.PATIENCE}")
    if args.fast:
        print("⚡ Mode: FAST EXPERIMENTATION")
    elif args.stable:
        print("🎯 Mode: STABLE TRAINING")
    else:
        print("⚙️  Mode: CUSTOM")
    print("=" * 50)

    try:
        # Validate paths before starting
        RuntimeConfig.validate_paths()

        # Create trainer with optimized config
        trainer = YOLOTrainer(config=RuntimeConfig)

        # Start training
        print(f"🚀 Starting training run: {RuntimeConfig.RUN_NAME}")
        trainer.train(resume_if_possible=not args.no_resume)

        print("✅ Training completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💡 You can resume training by running the same command again")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Tips for troubleshooting:")
        print("   - Check your data.yaml file")
        print("   - Try reducing batch size or image size")
        print("   - Use --stable mode for more conservative settings")
        sys.exit(1)


if __name__ == "__main__":
    main()
