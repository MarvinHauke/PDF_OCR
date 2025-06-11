#!/usr/bin/env python3

print("Starting simple test...")

import sys
from pathlib import Path

# Add path (since we're in training_project, we need to go up one level and then into training_project)
# Or we can use relative paths to our modules
sys.path.append(".")  # Add current directory
sys.path.append("..")  # Add parent directory

print("Path added, testing imports...")

try:
    # Test argparse
    import argparse

    print("✅ argparse")

    # Test torch (this might be where it hangs)
    print("Testing torch import...")
    import torch

    print(f"✅ torch {torch.__version__}")

    # Test MPS
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Test our modules (since we're in training_project directory)
    print("Testing our modules...")
    from config.settings import Config

    print("✅ Config imported")

    from src.trainer import YOLOTrainer

    print("✅ YOLOTrainer imported")

    print("✅ All basic tests passed")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

print("Simple test completed")
