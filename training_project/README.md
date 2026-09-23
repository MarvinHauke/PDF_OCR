# 🚀 YOLO Training Project

A comprehensive YOLO training framework optimized for Apple Silicon (MPS) with YAML-based configuration management and advanced autocompletion support.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Labeling Workflow](#labeling-workflow)
- [Autocompletion Setup](#autocompletion-setup)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project provides a robust, production-ready framework for training YOLO models with special optimizations for Apple Silicon devices. It features a flexible YAML-based configuration system, comprehensive CLI tools, and intelligent autocompletion to streamline your machine learning workflow.

### Key Highlights

- **🍎 Apple Silicon Optimized**: Specially tuned for MacBook Pro/Air with M1/M2/M3/M4 chips
- **⚙️ YAML Configuration**: Centralized, version-controlled configuration management
- **🤖 Smart Autocompletion**: Shell completion for all CLI arguments and parameters
- **📊 Multiple Training Profiles**: Pre-configured setups for different scenarios
- **🔄 Auto-Resume**: Intelligent checkpoint detection and resumption
- **🛡️ Error Handling**: Robust error handling with memory cleanup

## ✨ Features

### Training Features

- **MPS Acceleration**: Full Metal Performance Shaders support for Apple Silicon
- **Automatic Mixed Precision**: Enhanced performance with AMP training
- **Memory Management**: Intelligent cache clearing and memory optimization
- **Batch Size Auto-tuning**: Automatic batch size adjustment for optimal GPU utilization
- **Multiple Model Support**: Support for all YOLO11 variants (nano to extra-large)

### Configuration Features

- **YAML-based Config**: Human-readable, version-controllable configuration files
- **Environment Profiles**: Separate configs for development, testing, and production
- **Runtime Overrides**: Command-line argument support to override any config parameter
- **Configuration Validation**: Schema validation with helpful error messages

### Developer Experience

- **Shell Autocompletion**: Bash and Zsh completion for all commands
- **IDE Integration**: VS Code schema support with IntelliSense
- **Comprehensive Logging**: Detailed logging with configurable verbosity
- **Progress Tracking**: Real-time training progress with ETA estimation

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/MarvinHauke/PDF_OCR.git
cd PDF_OCR
uv sync
```

### 2. Check MPS Availability

```bash
uv run python training_project/scripts/train.py --check-mps
```

### 3. Start Training

```bash
# Quick training with auto-optimizations
uv run python training_project/scripts/train.py --optimize-for-mps --device auto

# Or use a specific configuration
uv run python training_project/scripts/train.py --config training_project/config/mps_optimized.yaml
```

## 📦 Installation

### Prerequisites

- **Python 3.9+**
- **macOS 12.3+** (for MPS support)
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/MarvinHauke/PDF_OCR.git
cd PDF_OCR

# Install dependencies with uv (automatically creates virtual environment)
uv sync

# Or if you prefer to create virtual environment manually
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Dependencies

The project uses `pyproject.toml` for dependency management with uv:

```toml
[project]
dependencies = [
    "ultralytics>=8.0.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pyyaml>=6.0",
    "numpy>=1.21.0",
]

[project.optional-dependencies]
dev = [
    "argcomplete>=3.0.0",  # For autocompletion
    "tensorboard>=2.10.0", # For training visualization
    "black>=23.0.0",       # Code formatting
    "isort>=5.12.0",       # Import sorting
    "flake8>=6.0.0",       # Linting
]
```

Install with development dependencies:

```bash
uv sync --extra dev
```

## ⚙️ Configuration

### Configuration Files

The project uses YAML configuration files located in the `config/` directory:

- **`config.yaml`**: Default configuration with balanced settings
- **`mps_optimized.yaml`**: Aggressive MPS optimizations for maximum performance
- **`cpu_fallback.yaml`**: Safe CPU configuration as fallback

### Basic Configuration Structure

```yaml
# Model settings
model:
  architecture: "yolo11n.pt"
  use_best_weights: false

# Training parameters
training:
  device: "mps"
  batch_size: -1 # Auto-adjust
  epochs: 100
  workers: 0 # Required for MPS
  amp: true # Automatic Mixed Precision

# MPS-specific optimizations
mps:
  max_memory_fraction: 0.8
  empty_cache_frequency: 10
  use_cpu_fallback: true

# Paths
paths:
  training_data: "training_data"
  dataset_yaml: "training_data/data.yaml"
  project: "training_data/runs"
  sources: "training_data/sources"
  unlabeled: "training_data/unlabeled"
```

See the Labeling workflow section below for what the `sources/` and `unlabeled/` folders
are for.

### Creating Custom Configurations

1. **Copy base configuration:**

```bash
cp training_project/config/config.yaml training_project/config/my_experiment.yaml
```

2. **Modify parameters** as needed

3. **Use in training:**

```bash
uv run python training_project/scripts/train.py --config training_project/config/my_experiment.yaml
```

## 🏋️ Training

### Dataset Preparation

1. **Organize your dataset:**

```
training_project/training_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

2. **Create data.yaml:**

```yaml
path: /path/to/training_data
train: images/train
val: images/val

names:
  0: class1
  1: class2
  # ... add your classes
```

### Training Commands

#### Basic Training

```bash
# Start training with default settings
uv run python training_project/scripts/train.py

# Resume training automatically
uv run python training_project/scripts/train.py  # Will auto-resume if checkpoint exists
```

#### Advanced Training Options

```bash
# Use specific configuration
uv run python training_project/scripts/train.py --config training_project/config/mps_optimized.yaml

# Override specific parameters
uv run python training_project/scripts/train.py --epochs 200 --batch 32 --device mps

# Auto-select best device with optimizations
uv run python training_project/scripts/train.py --device auto --optimize-for-mps

# Train different model sizes
uv run python training_project/scripts/train.py --model yolo11s.pt --epochs 100
uv run python training_project/scripts/train.py --model yolo11m.pt --epochs 150
```

#### MPS-Specific Training

```bash
# Maximum MPS optimization
uv run python training_project/scripts/train.py --optimize-for-mps --device mps

# MPS with custom batch size
uv run python training_project/scripts/train.py --device mps --batch -1 --workers 0

# Monitor MPS performance
uv run python training_project/scripts/train.py --device mps --verbose
```

### Training Parameters

| Parameter    | Description               | MPS Recommended |
| ------------ | ------------------------- | --------------- |
| `--device`   | Training device           | `mps` or `auto` |
| `--batch`    | Batch size                | `-1` (auto)     |
| `--workers`  | Data loader workers       | `0`             |
| `--amp`      | Automatic Mixed Precision | `true`          |
| `--epochs`   | Number of epochs          | `100-300`       |
| `--patience` | Early stopping patience   | `50`            |

### Monitoring Training

- **Real-time logs**: Training progress is displayed in terminal
- **Weights & Biases**: Automatic experiment tracking (if configured)
- **TensorBoard**: Enable with `tensorboard: true` in config
- **Activity Monitor**: Check GPU utilization on macOS

## 🏷️ Labeling workflow

How new training data gets from a PDF to a labeled image:

```
training_data/sources/manual/        you drop PDFs/images here (crawler output: sources/crawled/<site>/)
        │  uv run pdf-ocr ingest
training_data/unlabeled/             rendered pages, not yet processed
        │  uv run python training_project/scripts/autolabel.py
        ├─▶ train/                   every box confidently accepted → labeled automatically
        ├─▶ review_queue/images/     anything uncertain, or no detections → human review
        └─▶ skipped/                 no detections and not sampled (review_no_detection_rate < 1)
        │  Label Studio: review, export JSON
        │  uv run python training_project/scripts/import_reviewed.py export.json
        └─▶ train/ or val/           human-verified (val_fraction goes to val/)
```

1. **Ingest.** `uv run pdf-ocr ingest` renders every PDF under `sources/` and copies images
   into `unlabeled/` as `<source path>__<name>-pNNN.png`. Each file is recorded by hash in
   `training_data/ingest_manifest.jsonl`, so re-running only processes new files.
2. **Autolabel.** `autolabel.py` decides per *image*. An image goes to `train/` only if every
   box on it is accepted. Otherwise it goes to review, with the non-rejected boxes pre-drawn.
   Pages without any detection go to review too, so schematics the model missed get labeled.
   Images are moved, so re-running is safe. Uses the threshold stub unless
   `TYPESAFE_API_KEY` is set.
3. **Review in Label Studio.** Run it in its own environment (installing it into this venv
   breaks `cv2`):

   ```bash
   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)" \
   uvx label-studio start
   ```

   Run this from the repo root. Create a project with the *Object Detection with Bounding
   Boxes* template and set its label to `schematic` (it must match `names` in
   `training_data/data.yaml`). Import `training_data/review_queue/label_studio_tasks.json`.
   Correct, add or delete boxes and submit each task. Submitting a task with no boxes
   confirms the page as a background image (no schematic).

   If images don't load, add a *Local Files* source storage in the project settings with
   the absolute path of `training_data/review_queue/images` (don't sync it; it only grants
   access). Label Studio's docs aren't clear on whether this is needed for imported tasks.
4. **Import.** Export the project as **JSON** (not YOLO) and run
   `import_reviewed.py <export.json> --dry-run`, then without `--dry-run`. Reviewed images
   move to `train/` or `val/` with YOLO labels, and their tasks are removed from the queue.
   Unreviewed tasks stay. An unknown label name aborts the import before anything moves.
5. **Retrain** with `scripts/train.py`.

## 🎯 Autocompletion Setup

Completion is powered by [`argcomplete`](https://github.com/kislyuk/argcomplete) directly on
`train.py` and `predict.py` (both carry a `# PYTHON_ARGCOMPLETE_OK` marker) — there are no
project-specific shell scripts to install or maintain.

### Global Setup (Recommended)

```bash
# One-time: registers completion for any argcomplete-enabled script on your PATH
activate-global-python-argcomplete --user

# Restart terminal or reload shell config
source ~/.bashrc  # or ~/.zshrc
```

### Per-script Setup

```bash
# Add to ~/.bashrc or ~/.zshrc
eval "$(register-python-argcomplete training_project/scripts/train.py)"
eval "$(register-python-argcomplete training_project/scripts/predict.py)"
```

### Debugging completion

```bash
# Checks argcomplete is installed, the magic comment is present, and registration works
./training_project/scripts/debug_completion.sh
```

### Using Autocompletion

After setup, you'll have intelligent completion for:

```bash
# CLI arguments
uv run python training_project/scripts/train.py --<TAB>
# Shows: --config --epochs --batch --device --model etc.

# Configuration files
uv run python training_project/scripts/train.py --config <TAB>
# Shows: training_project/config/config.yaml training_project/config/mps_optimized.yaml etc.

# Device options
uv run python training_project/scripts/train.py --device <TAB>
# Shows: auto cpu mps cuda

# Model architectures
uv run python training_project/scripts/train.py --model <TAB>
# Shows: yolo11n.pt yolo11s.pt yolo11m.pt etc.

# Common values
uv run python training_project/scripts/train.py --batch <TAB>
# Shows: -1 8 16 32 64
```

## 📝 Examples

### Example 1: Quick MPS Training

```bash
# Check system compatibility
uv run python training_project/scripts/train.py --check-mps

# Start optimized training
uv run python training_project/scripts/train.py --optimize-for-mps
```

### Example 2: Custom Experiment

```bash
# Create custom config
cp training_project/config/mps_optimized.yaml training_project/config/my_experiment.yaml

# Edit config file with your parameters
# nano training_project/config/my_experiment.yaml

# Train with custom config
uv run python training_project/scripts/train.py --config training_project/config/my_experiment.yaml --run-name my_experiment_v1
```

### Example 3: Hyperparameter Sweep

```bash
# Train different model sizes
for model in yolo11n.pt yolo11s.pt yolo11m.pt; do
    uv run python training_project/scripts/train.py --model $model --run-name "sweep_$model" --epochs 50
done
```

### Example 4: Production Training

```bash
# Long training run with all optimizations
uv run python training_project/scripts/train.py \
    --config training_project/config/mps_optimized.yaml \
    --epochs 300 \
    --patience 100 \
    --run-name production_v1 \
    --optimize-for-mps
```

## 🔧 Troubleshooting

### Common Issues

#### MPS Not Available

```bash
# Check MPS status
uv run python training_project/scripts/train.py --check-mps

# Solutions:
# 1. Update to macOS 12.3+
# 2. Install PyTorch with MPS support: uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 3. Use CPU fallback: --device cpu
```

#### Training Slow/Freezing

```bash
# Apply MPS optimizations
uv run python training_project/scripts/train.py --optimize-for-mps

# Or manually set safe parameters
uv run python training_project/scripts/train.py --workers 0 --batch -1 --device mps
```

#### Memory Errors

```bash
# Reduce batch size
uv run python training_project/scripts/train.py --batch 8

# Enable aggressive memory management
uv run python training_project/scripts/train.py --config training_project/config/mps_optimized.yaml
```

#### "GPU_mem: 0G" Showing

This is normal for MPS! Check Activity Monitor → GPU tab to see actual GPU usage (should be 60-90%).

### Performance Tips

1. **Use Auto Batch Size**: `--batch -1` for optimal memory usage
2. **Disable Workers**: `--workers 0` for MPS compatibility
3. **Enable AMP**: `--amp` for better performance
4. **Monitor Activity Monitor**: Check GPU utilization
5. **Clear Cache Regularly**: Built into MPS optimization

### Getting Help

- **Check logs**: Training logs show detailed error information
- **Verbose mode**: Add `--verbose` for detailed output
- **Configuration validation**: YAML schema will catch config errors
- **MPS status**: Use `--check-mps` to verify system compatibility

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/MarvinHauke/PDF_OCR.git
cd PDF_OCR

# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks (if configured)
uv run pre-commit install
```

### Code Style

- **Black** for Python formatting
- **isort** for import sorting
- **flake8** for linting
- **YAML** validation for config files

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Ensure all checks pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLO implementation
- **PyTorch Team**: For MPS support in PyTorch
- **Apple**: For Metal Performance Shaders framework
- **Contributors**: Thank you to all contributors and users

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MarvinHauke/PDF_OCR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MarvinHauke/PDF_OCR/discussions)
- **Documentation**: Check the [`docs/`](../docs/) folder at the repo root for project-wide planning and guides

---

**Happy Training! 🎯🚀**
