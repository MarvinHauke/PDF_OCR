#!/usr/bin/env python3
"""Fixed debug script for correct directory structure"""

import os
import sys
import traceback
from pathlib import Path

print("🔍 FIXED DEBUG SCRIPT")
print("=" * 50)


def check_file_structure():
    """Check if all required files exist"""
    print("\n📁 Checking file structure...")

    # Assuming we're running from PDF_OCR/training_project/
    required_files = [
        "src/__init__.py",
        "config/__init__.py",
        "config/config.yaml",
        "scripts/train.py",
        "src/trainer.py",
        "src/utils.py",
        "config/settings.py",
    ]

    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            missing_files.append(file_path)

    return missing_files


def test_basic_imports():
    """Test basic Python imports"""
    print("\n🐍 Testing basic imports...")

    try:
        import argparse

        print("✅ argparse")
    except Exception as e:
        print(f"❌ argparse: {e}")
        return False

    try:
        import sys

        print("✅ sys")
    except Exception as e:
        print(f"❌ sys: {e}")
        return False

    try:
        import torch

        print(f"✅ torch {torch.__version__}")
        print(f"   MPS available: {torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"❌ torch: {e}")
        return False

    try:
        import yaml

        print("✅ yaml")
    except Exception as e:
        print(f"❌ yaml: {e}")
        return False

    try:
        from ultralytics import YOLO

        print("✅ ultralytics")
    except Exception as e:
        print(f"❌ ultralytics: {e}")
        return False

    return True


def test_custom_imports():
    """Test custom module imports"""
    print("\n🔧 Testing custom module imports...")

    # Add current directory to path (we should be in training_project/)
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")

    # Check if we're in the right place
    if not (current_dir / "src").exists():
        print(f"❌ src directory not found at {current_dir / 'src'}")
        print("Make sure you're running this from PDF_OCR/training_project/")
        return False

    sys.path.insert(0, str(current_dir))
    print(f"📍 Added to path: {current_dir}")

    try:
        from config.settings import Config

        print("✅ Config imported")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.trainer import YOLOTrainer

        print("✅ YOLOTrainer imported")
    except Exception as e:
        print(f"❌ YOLOTrainer import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from src.utils import setup_logging

        print("✅ utils imported")
    except Exception as e:
        print(f"❌ utils import failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_config_creation():
    """Test config object creation"""
    print("\n⚙️ Testing config creation...")

    try:
        current_dir = Path.cwd()
        sys.path.insert(0, str(current_dir))
        from config.settings import Config

        config = Config()
        print("✅ Config object created")
        print(f"   Device: {getattr(config, 'DEVICE', 'Not set')}")
        print(f"   Model: {getattr(config, 'MODEL_ARCH', 'Not set')}")
        print(f"   Epochs: {getattr(config, 'EPOCHS', 'Not set')}")
        return True

    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        traceback.print_exc()
        return False


def test_trainer_creation():
    """Test trainer object creation"""
    print("\n🏋️ Testing trainer creation...")

    try:
        current_dir = Path.cwd()
        sys.path.insert(0, str(current_dir))
        from config.settings import Config

        from src.trainer import YOLOTrainer

        config = Config()
        trainer = YOLOTrainer(config=config)
        print("✅ Trainer object created")
        return True

    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        traceback.print_exc()
        return False


def test_script_execution():
    """Test if the train.py script can at least be imported"""
    print("\n📜 Testing train.py script...")

    try:
        script_path = Path("scripts/train.py")
        if not script_path.exists():
            print(f"❌ Script not found at {script_path}")
            return False

        # Try to read the script
        with open(script_path, "r") as f:
            content = f.read()

        print(f"✅ Script readable ({len(content)} characters)")

        # Check if it has main execution
        if '__name__ == "__main__"' in content:
            print("✅ Script has main execution block")
        else:
            print("⚠️  Script missing main execution block")

        return True

    except Exception as e:
        print(f"❌ Script test failed: {e}")
        return False


def test_minimal_yolo():
    """Test minimal YOLO functionality"""
    print("\n🎯 Testing minimal YOLO...")

    try:
        from ultralytics import YOLO

        # Just load a model, don't train
        model = YOLO("yolo11n.pt")
        print("✅ YOLO model loaded successfully")

        # Test device detection
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"✅ Device selected: {device}")

        return True

    except Exception as e:
        print(f"❌ Minimal YOLO test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print(f"Current directory: {Path.cwd()}")
    print(f"Python version: {sys.version}")
    print(f"Expected structure: PDF_OCR/training_project/[src,scripts,config]")

    tests = [
        ("File Structure", check_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Custom Imports", test_custom_imports),
        ("Config Creation", test_config_creation),
        ("Trainer Creation", test_trainer_creation),
        ("Script Execution", test_script_execution),
        ("Minimal YOLO", test_minimal_yolo),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "File Structure":
                missing = test_func()
                results[test_name] = len(missing) == 0
                if missing:
                    print(f"\n❌ Missing files: {missing}")
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print(f"{'='*50}")

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")

    failed_tests = [name for name, passed in results.items() if not passed]

    if failed_tests:
        print(f"\n🚨 FAILED TESTS: {', '.join(failed_tests)}")
        print("\n🔧 NEXT STEPS:")
        if "File Structure" in failed_tests:
            print("1. Create missing __init__.py files:")
            print("   touch src/__init__.py")
            print("   touch config/__init__.py")
            print("2. Create config.yaml file in config/")
        if "Basic Imports" in failed_tests:
            print("3. Install missing dependencies: uv sync")
        if "Custom Imports" in failed_tests:
            print("4. Check that you're in PDF_OCR/training_project/ directory")
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your setup should work. Try running the training script again.")


if __name__ == "__main__":
    main()
