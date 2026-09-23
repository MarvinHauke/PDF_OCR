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
  - [Label Studio setup](#label-studio-setup)
  - [Subcircuits (stage 2)](#subcircuits-stage-2)
  - [Crawling training material](#crawling-training-material)
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
3. **Review in Label Studio.** See [Label Studio setup](#label-studio-setup) below. Correct,
   add or delete boxes and submit each task. Submitting a task with no boxes confirms the
   image as a background image.
4. **Import.** Export the project as **JSON** (not YOLO) and run
   `import_reviewed.py <export.json> --dry-run`, then without `--dry-run`. Reviewed images
   move to `train/` or `val/` with YOLO labels, and their tasks are removed from the queue.
   Unreviewed tasks stay. An unknown label name aborts the import before anything moves.
5. **Retrain** with `scripts/train.py`.

### Label Studio setup

Three processes, each in its own terminal, all from the repo root:

```bash
# 1. Label Studio (own uvx environment; installing it into this venv breaks cv2)
training_project/scripts/start_label_studio.sh

# 2. ML backend: pre-labels tasks with our YOLO models (runs in this venv, uses MPS)
uv run python training_project/labelstudio/ml_backend.py

# 3. Once: create/update projects, storage, backend connection and import new tasks.
#    Re-run after every autolabel.py / make_crops.py run; it only imports new tasks.
uv run python training_project/scripts/setup_label_studio.py
```

`setup_label_studio.py` needs a personal access token (*Account & Settings → Personal Access
Token*) in `LABEL_STUDIO_API_KEY`. Put `export LABEL_STUDIO_API_KEY="<token>"` in the repo's
gitignored `.envrc` and run `direnv allow`; never commit it.

It sets up two projects:

| Project | Labeling config | Tasks from | Pre-labels from |
|---|---|---|---|
| Schematics (pages) | `labelstudio/schematics.xml` | `training_data/review_queue/` | page model |
| Subcircuits (crops) | `labelstudio/subcircuits.xml` | `training_data/subcircuits/review_queue/` | subcircuit model, once trained |

Things that are easy to get wrong, and why the script handles them:
- **Images only load with a Local Files storage.** Label Studio (checked in 1.23) serves
  `/data/local-files/?d=…` only if the project has a Local Files storage whose path contains
  the file; otherwise you get "There was an issue loading URL from $image value". The script
  adds one for `training_data/` without syncing it (syncing would create a task per file).
- **The document root must be the repo root**, because task URLs are repo-relative.
  `start_label_studio.sh` sets it.
- **Label names must match `data.yaml`.** The XML configs do.

The ML backend (`labelstudio/ml_backend.py`) implements Label Studio's ML backend protocol
directly instead of using the `label-studio-ml` package, whose SDK dependency needs
`opencv-python-headless` (the same `cv2` conflict). Label Studio asks it for predictions when
you open a task that has none. It picks the model whose class names match the project's
labels and reloads weights after retraining.

### Subcircuits (stage 2)

Functional blocks (`power_supply`, `amplifier`, `filter`, `oscillator`) are labeled on
schematic crops, not full pages, so small details stay readable. The dataset lives in
`training_data/subcircuits/` and is configured by `config/subcircuits.yaml`:

```bash
# Crop every labeled schematic from the page dataset (train/ + val/) into subcircuits/unlabeled/
uv run python training_project/scripts/make_crops.py
# Same workflow as the page dataset, with --config
uv run python training_project/scripts/autolabel.py --config config/subcircuits.yaml
uv run python training_project/scripts/setup_label_studio.py --project subcircuits
uv run python training_project/scripts/import_reviewed.py export.json --config config/subcircuits.yaml
uv run python training_project/scripts/train.py --config config/subcircuits.yaml
```

Until the first subcircuit model is trained, `autolabel.py` sends every crop to review
without boxes and the ML backend returns no pre-labels: the first round is manual.
`crops_manifest.jsonl` records the parent image and box of each crop.

### Crawling training material

`pdf-ocr crawl <source>` downloads raw material into `sources/crawled/<source>/` of the
matching dataset; from there it's the normal workflow (`pdf-ocr ingest [--dataset …]` →
`autolabel.py` → Label Studio). All limits, sources and license rules are in
`config/crawl.yaml`.

| Source | What | Dataset | Tier |
|---|---|---|---|
| `wikimedia` | circuit diagrams from Commons categories, with a class hint per category | subcircuits | `free` |
| `kicad_github` | open-hardware KiCad projects (`topic:kicad license:<key>`), rendered with `kicad-cli` | subcircuits | `free` |
| `archive_org` | synthesizer service manual PDFs | pages | mostly `restricted` |
| `urls` | your list in `training_data/sources/datasheet_urls.txt` | pages | depends |

```bash
uv run pdf-ocr crawl wikimedia --limit 20 --dry-run   # list what would be downloaded
uv run pdf-ocr crawl wikimedia --limit 20
uv run pdf-ocr ingest --dataset subcircuits          # crawled images/sheets -> subcircuits/unlabeled/
uv run pdf-ocr ingest --max-pages 30                 # crawled PDFs -> unlabeled/ (first 30 pages each)
```

Rules (enforced in `src/pdf_ocr/crawl/base.py`):
- **Tiers.** `free` = a license on `free_licenses` (no NC/ND). `restricted` = no usable license,
  but the host is in `reviewed_hosts` (terms checked by hand) and doesn't opt out of text and
  data mining in a machine-readable way (`tdm-reservation` header, `/.well-known/tdmrep.json`).
  Everything else is skipped. `restricted` files are for local training only: `training_data/`
  is gitignored, and the tier is recorded in `sources.jsonl` and in the ingest manifest.
- **Blocked hosts** (`blocked_hosts`) are never fetched. elektronik-kompendium.de is there
  because its imprint reserves text and data mining (§ 44b UrhG) in plain text, which
  robots.txt doesn't show.
- **Politeness.** robots.txt for file downloads, documented APIs follow their own usage
  policies, 1 request/s per host, serial requests, `maxlag` for Wikimedia. Set `contact` in
  `crawl.yaml`, since Wikimedia asks for contact info in the User-Agent.
- **Limits.** Per-run counts, per-file size limits, and a **1 GB total cap** for all
  `sources/crawled/` folders; crawling stops cleanly when it's reached.
- **Provenance.** Every file gets a `sources.jsonl` record: URL, license, author, tier, hint.
- GitHub without a valid `GITHUB_TOKEN` allows ~20 repos per hour; an invalid token is ignored.

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
