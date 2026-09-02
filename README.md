<div align="center">

# 🍎 Fruit Recognition System

**Deep Learning-Based 20-Fruit Classification with ResNet18 + CBAM Attention**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-RoL1n%2F20--Fruit__Recognition-yellow)](https://huggingface.co/RoL1n/20-Fruit_Recognition)
[![GitHub Release](https://img.shields.io/badge/Release-v1.0.0-brightgreen.svg)](https://github.com/RolinShmily/20-Fruit_Recognition_System/releases/latest)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Test Accuracy: 91.78%** · GUI Application · ONNX Inference · PyInstaller Packaging

English · [中文文档](README_CN.md)

</div>

---

## Overview

A deep learning-based fruit classification system that recognizes **20 types of fruits** in natural background images. Built with **ResNet18** backbone enhanced by **CBAM** (Channel + Spatial) attention mechanism, trained with **CutMix** augmentation and a **two-stage progressive** training strategy.

| Component | Solution |
|-----------|----------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Attention | CBAM Channel + Spatial dual attention (injected after Layer2/3/4) |
| Augmentation | CutMix mixed-sample + multi-dimensional transform pipeline |
| Class Imbalance | Weighted random sampling + weighted cross-entropy loss |
| Training | Two-stage progressive training with differential learning rates |
| Brightness | Brightness normalization for domain adaptation |
| Deployment | PySide6 GUI + ONNX Runtime (~150MB packaged) |

---

## Key Features

- **CBAM Attention** — Channel + Spatial dual-dimension attention guides the model to focus on fruit regions and suppress background interference
- **CutMix Augmentation** — Mixed-sample augmentation forces the model to learn from partial regions, improving generalization
- **Two-Stage Progressive Training** — Phase 1: freeze backbone to train attention modules; Phase 2: unfreeze deep layers for fine-tuning
- **Class Imbalance Handling** — Dual mechanism of weighted sampling + weighted loss ensures minority class performance
- **GUI Application** — Single image and batch prediction with Top-5 confidence visualization
- **Batch Analysis** — Automatically generates 9 statistical charts including confusion matrix, error analysis, and low-confidence alerts
- **Lightweight Deployment** — ONNX Runtime inference, no full PyTorch environment required

---

## Models & Resources

All project assets, pre-trained weights, and documentation are available via GitHub Releases and Hugging Face:

- 🤗 **Hugging Face Model Hub**: [RoL1n/20-Fruit_Recognition](https://huggingface.co/RoL1n/20-Fruit_Recognition) — Model checkpoints (`.pt`) and ONNX models (`.onnx`)
- 📦 **GitHub Releases**: [Release v1.0.0](https://github.com/RolinShmily/20-Fruit_Recognition_System/releases/latest) provides downloadable packages:
  - 📄 **`report.pdf`** — Comprehensive technical report & experimental analysis
  - 🧠 **`models.zip`** — Pre-trained PyTorch checkpoint (`best_resnet18_cbam.pt`) and ONNX model (`best_resnet18_cbam.onnx`)
  - 📊 **`data.zip`** — Pre-split 20-class fruit dataset (`Training/`, `Validation/`, `Test/`)
  - 🖼️ **`ImagesForTesting.zip`** — Sample test images for instant evaluation
  - 💻 **`GUI_APP.zip`** — Standalone desktop GUI executable (ready-to-run, no Python required)

---

## Architecture

```
Input Image (224×224)
        │
        ▼
┌──────────────────────────────────────────┐
│    ResNet18 Backbone (ImageNet)           │
│    ├── Conv1 + BN + ReLU + MaxPool        │
│    ├── Layer1 (64ch)        [frozen]      │
│    ├── Layer2 (128ch)       [frozen]      │
│    │       └──→ CBAM(128)                 │
│    ├── Layer3 (256ch)     [unfrozen]      │
│    │       └──→ CBAM(256)                 │
│    ├── Layer4 (512ch)     [unfrozen]      │
│    │       └──→ CBAM(512)                 │
│    └── AdaptiveAvgPool2d(1,1)             │
└──────────────┬───────────────────────────┘
               │ 512-d feature vector
               ▼
┌──────────────────────────────────────────┐
│    Custom Classification Head             │
│    Dropout(0.5)                           │
│    Linear(512 → 256) + ReLU + BN          │
│    Dropout(0.3)                           │
│    Linear(256 → 20)                       │
└──────────────┬───────────────────────────┘
               │
        20-class logits
```

**Total Parameters:** ~11.39M (only ~213K trainable in Phase 1)

---

## Supported Fruits (20 Classes)

| # | Fruit | # | Fruit |
|:---:|-------|:---:|-------|
| 1 | 🍎 Apple | 11 | Jujube |
| 2 | 🍌 Banana | 12 | 🍐 Pear |
| 3 | 🍊 Orange | 13 | 🍒 Cherry |
| 4 | 🍇 Grape | 14 | 🥥 Coconut |
| 5 | 🍉 Watermelon | 15 | Pomegranate |
| 6 | 🍓 Strawberry | 16 | Papaya |
| 7 | 🍍 Pineapple | 17 | 🥑 Avocado |
| 8 | 🥭 Mango | 18 | Blueberry |
| 9 | 🍋 Lemon | 19 | Cantaloupe |
| 10 | 🥝 Kiwi | 20 | Dragonfruit |

---

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU recommended (CPU supported)

### Installation

```bash
# Clone the repository
git clone https://github.com/RolinShmily/20-Fruit_Recognition_System.git
cd 20-Fruit_Recognition_System

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

You can either download the pre-split `data.zip` directly from [Releases](https://github.com/RolinShmily/20-Fruit_Recognition_System/releases/latest) and extract it into the project root, or prepare from scratch:

1. Download the [fruits-262 dataset](https://www.kaggle.com/datasets/aelchimminut/fruits262)
2. Place the selected 20 fruit categories under `data/`
3. Split the dataset:

```bash
python split_dataset.py
```

Expected directory structure:

```
data/
├── Training/       # ~16,692 images
├── Validation/     # ~3,129 images
└── Test/           # ~1,042 images
```

### Training

```bash
python train.py
```

Two-stage training strategy:
- **Phase 1** (10 epochs): Freeze backbone, train CBAM + classifier head (lr=1e-3)
- **Phase 2** (15 epochs): Unfreeze Layer3+4, differential learning rates (backbone=1e-4, head=5e-4)

The best model is automatically saved to `models/best_resnet18_cbam.pt`. Alternatively, download pre-trained weights from [Hugging Face](https://huggingface.co/RoL1n/20-Fruit_Recognition) or [GitHub Releases](https://github.com/RolinShmily/20-Fruit_Recognition_System/releases/latest).

### Evaluation

```bash
python evaluate.py
```

Outputs: classification report (Precision / Recall / F1), confusion matrix heatmap, per-class accuracy.

### Prediction

```bash
# Single image
python predict.py images/apple.jpg

# Batch prediction with accuracy analysis
python batch_predict.py images/
```

Batch prediction automatically generates:
- **9 statistical charts** — Success rate pie chart, class distribution, confidence distribution, confusion matrix, etc.
- **Detailed text report** — Per-class accuracy, error case analysis, low-confidence warnings
- **Real-time terminal output** — Per-image prediction result and confidence score

### GUI Application

```bash
# Export ONNX model first (required for GUI)
python export_onnx.py

# Launch GUI
python gui_predictor.py
```

GUI features:
- Single image: Select/drag image, view Top-5 predictions with confidence
- Batch mode: Select image directory, browse results per image
- Save results: Export original image, annotated image, probability chart, text results

---

## Project Structure

```
├── config.py              # Global configuration (classes, hyperparameters, paths)
├── model.py               # ResNet18 + CBAM model definition
├── dataset.py             # Dataset loader with CutMix augmentation
├── train.py               # Two-stage training script
├── evaluate.py            # Model evaluation (metrics + confusion matrix)
├── predict.py             # Single image prediction
├── batch_predict.py       # Batch prediction with analysis report
├── gui_predictor.py       # PySide6 GUI application
├── export_onnx.py         # Export PyTorch model to ONNX
├── split_dataset.py       # Dataset splitting script
├── requirements.txt       # Python dependencies
├── data/                  # Dataset directory (available in Release data.zip)
├── models/                # Saved models (available in Release models.zip / Hugging Face)
├── results/               # Output results
└── images/                # Sample test images
```

---

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_CBAM` | `True` | Enable CBAM attention mechanism |
| `USE_CUTMIX` | `True` | Enable CutMix augmentation |
| `CUTMIX_PROB` | `0.5` | CutMix trigger probability |
| `PHASE1_EPOCHS` | `10` | Phase 1 training epochs |
| `PHASE2_EPOCHS` | `15` | Phase 2 training epochs |
| `BATCH_SIZE` | `32` | Training batch size |
| `EARLY_STOP_PATIENCE` | `7` | Early stopping patience |
| `USE_CLASS_WEIGHTS` | `True` | Weighted cross-entropy loss |
| `USE_WEIGHTED_SAMPLER` | `True` | Weighted random sampling |

---

## Packaging

Package the GUI as a standalone executable (no Python installation required):

```bash
# Install packaging tools
pip install pyinstaller onnxruntime

# Ensure ONNX model is exported
python export_onnx.py

# Package
pyinstaller --noconfirm --onedir --windowed ^
    --name "FruitRecognition" ^
    --add-data "models\best_resnet18_cbam.onnx;models" ^
    --hidden-import PySide6.QtWidgets ^
    --hidden-import PySide6.QtCore ^
    --hidden-import PySide6.QtGui ^
    --hidden-import onnxruntime ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    gui_predictor.py
```

> **macOS/Linux:** Replace `^` with `\` for line continuation, and `;` with `:` in `--add-data`.

| Package Method | Size | Notes |
|---------------|------|-------|
| PyTorch (GPU) | ~2 GB | Includes full CUDA libraries |
| PyTorch (CPU) | ~600 MB | CPU inference only |
| **ONNX Runtime** | **~150 MB** | This project's approach, no PyTorch needed |

---

## Innovation Highlights

| Aspect | Conventional | This Project |
|--------|-------------|--------------|
| Dataset | White background, severe domain gap | Natural background, eliminates domain difference at source |
| Attention | Pure CNN, no explicit attention | CBAM channel + spatial attention, focuses on fruit regions |
| Augmentation | Basic geometric transforms | CutMix + multi-dimensional augmentation pipeline |
| Class Imbalance | No handling | Dual mechanism: weighted sampling + weighted loss |
| Training | Single-stage | Two-stage progressive with differential learning rates |

---

## References

| Paper / Resource | Authors / Year | Contribution |
|------------------|---------------|--------------|
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He et al., 2016 | Proposed ResNet |
| [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) | Woo et al., 2018 | Proposed channel + spatial attention |
| [CutMix: Regularization Strategy](https://arxiv.org/abs/1905.04899) | Yun et al., 2019 | Proposed mixed-sample augmentation |
| [fruits-262 Dataset](https://www.kaggle.com/datasets/aelchimminut/fruits262) | — | Natural background fruit images |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Built with PyTorch · ResNet18 · CBAM · CutMix**

</div>
