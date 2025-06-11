#!/usr/bin/env python3
"""Debug version of train.py with detailed logging"""

print("🔍 DEBUG: Script started")

try:
    print("🔍 DEBUG: Importing modules...")

    import argparse

    print("🔍 DEBUG: argparse imported")

    import sys

    print("🔍 DEBUG: sys imported")

    import torch

    print("🔍 DEBUG: torch imported")

    from pathlib import Path

    print("🔍 DEBUG: pathlib imported")

    # Add project root to path
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.append(str(PROJECT_ROOT))
    print(f"🔍 DEBUG: Added path: {PROJECT_ROOT}")

    print("🔍 DEBUG: Importing custom modules...")
    from config.settings import Config

    print("🔍 DEBUG: Config imported")

    from src.trainer import YOLOTrainer

    print("🔍 DEBUG: YOLOTrainer imported")

    def check_mps_availability():
        """Check and report MPS availability"""
        print("🔍 DEBUG: check_mps_availability function called")
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
                print(
                    "❌ MPS not available: macOS version < 12.3 or no MPS-enabled device"
                )
        print("=" * 30)

    def main():
        print("🔍 DEBUG: main() function called")

        parser = argparse.ArgumentParser(description="Debug training script")
        print("🔍 DEBUG: ArgumentParser created")

        parser.add_argument(
            "--check-mps", action="store_true", help="Check MPS availability"
        )
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--batch", type=int, default=4, help="Batch size")
        parser.add_argument("--device", type=str, default="mps", help="Device")

        print("🔍 DEBUG: About to parse arguments...")
        args = parser.parse_args()
        print(f"🔍 DEBUG: Arguments parsed: {args}")

        if args.check_mps:
            print("🔍 DEBUG: MPS check requested")
            check_mps_availability()
            print("🔍 DEBUG: MPS check completed, returning")
            return

        print("🔍 DEBUG: Creating config...")
        try:
            config = Config()
            print("🔍 DEBUG: Config created successfully")
        except Exception as e:
            print(f"🔍 DEBUG: Config creation failed: {e}")
            raise

        print("🔍 DEBUG: Creating trainer...")
        try:
            trainer = YOLOTrainer(config=config)
            print("🔍 DEBUG: Trainer created successfully")
        except Exception as e:
            print(f"🔍 DEBUG: Trainer creation failed: {e}")
            raise

        print("🔍 DEBUG: Starting training...")
        try:
            results = trainer.train()
            print(f"🔍 DEBUG: Training completed: {results}")
        except Exception as e:
            print(f"🔍 DEBUG: Training failed: {e}")
            raise

    if __name__ == "__main__":
        print("🔍 DEBUG: __main__ block entered")
        try:
            main()
            print("🔍 DEBUG: main() completed successfully")
        except Exception as e:
            print(f"🔍 DEBUG: Exception in main: {e}")
            import traceback

            traceback.print_exc()
        except KeyboardInterrupt:
            print("🔍 DEBUG: KeyboardInterrupt caught")
        finally:
            print("🔍 DEBUG: Script ending")

except Exception as e:
    print(f"🔍 DEBUG: Exception during imports: {e}")
    import traceback

    traceback.print_exc()

print("🔍 DEBUG: End of script reached")
