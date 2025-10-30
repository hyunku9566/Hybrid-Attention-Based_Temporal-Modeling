# 🏠 Lightweight ADL Recognition with TCN-BiGRU-Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight yet highly accurate deep learning model for Activities of Daily Living (ADL) recognition in smart home environments.

## 🏆 Key Results

### Best Model: Local Attention Hybrid 🥇
- **97.52% Test Accuracy** (State-of-the-art)
- **0.9717 Macro F1-Score**
- **3.32M Parameters** (Real-time capable)
- **+2.12%p improvement** over baseline

### Baseline Model: TCN-BiGRU-Attention 🥈
- **95.40% Test Accuracy** (Production-ready)
- **0.9468 Macro F1-Score**
- **2.26M Parameters** (Most efficient)
- **Excellent parameter/performance ratio**

## 🎯 Highlights

We evaluated **5 different architectures** and achieved:
- ✅ **97.52% accuracy** with Local Attention Hybrid (best)
- ✅ **95.40% accuracy** with Baseline (most efficient)
- ✅ **>95% F1 on ALL classes** with Local Hybrid
- ✅ **Interpretable** through attention weight visualization
- ✅ **Real-time inference** on edge devices

## 📊 Model Comparison

| Rank | Model | Test Acc | F1 Score | Parameters | Status |
|------|-------|----------|----------|------------|--------|
| 🥇 | **Local Attention Hybrid** | **97.52%** | **0.9717** | 3.32M | ⭐ Best Overall |
| 🥈 | **Baseline (TCN-BiGRU)** | **95.40%** | **0.9468** | 2.26M | ⚡ Most Efficient |
| 🥉 | Deep TCN | 93.42% | 0.9174 | 2.43M | Good |
| 4 | Conformer | 92.08% | 0.9009 | 6.15M | Overparameterized |
| 5 | Transformer | 83.66% | 0.8042 | 3.22M | Not Recommended |

**Full comparison:** See [docs/COMPREHENSIVE_MODEL_COMPARISON.md](docs/COMPREHENSIVE_MODEL_COMPARISON.md)

## 🏗️ Architectures

### 1. Local Attention Hybrid 🥇 (Best Performance)

```
Input (T=100, F=114)
       ↓
[Feature Projection] Linear(114 → 256)
       ↓
[Temporal Encoding] TCN (5 blocks, dilation=1,2,4,8,16)
       ↓
[Local Self-Attention] Window=25, 2 heads
       ↓
[Sequential Modeling] BiGRU (hidden=256, bidirectional)
       ↓
[Global Attention] Additive Attention
       ↓
[Classification] FC → ReLU → Dropout → FC(5 classes)
       ↓
Output: [cooking, hand_washing, sleeping, medicine, eating]
```

**Key Components:**
- **5-block TCN** with exponential dilation [1,2,4,8,16]
- **Local Self-Attention** (window=25) for efficient local pattern capture
- **BiGRU** for bidirectional sequential modeling
- **Global Attention** for final temporal aggregation
- **Residual connections** + LayerNorm for stable training

**Performance:** 97.52% accuracy, 0.9717 F1

---

### 2. Baseline (TCN-BiGRU-Attention) 🥈 (Most Efficient)

```
Input (T=100, F=114)
       ↓
[Feature Projection] Linear(114 → 256)
       ↓
[Temporal Encoding] TCN (3 blocks, dilation=1,2,4)
       ↓
[Sequential Modeling] BiGRU (hidden=256, bidirectional)
       ↓
[Temporal Attention] Additive Attention (Bahdanau-style)
       ↓
[Classification] FC → ReLU → Dropout → FC(5 classes)
       ↓
Output: [cooking, hand_washing, sleeping, medicine, eating]
```

**Key Components:**
- **3-block TCN** with dilations [1,2,4] (receptive field: 13)
- **BiGRU** (25% fewer params than BiLSTM)
- **Additive Attention** (interpretable, single-focus)

**Performance:** 95.40% accuracy, 0.9468 F1, only 2.26M params

## 📂 Project Structure

```
baseline-adl-recognition/
├── data/                       # Data preprocessing
│   ├── preprocess.py          # Main preprocessing pipeline
│   ├── build_features.py      # Feature extraction from sensors
│   └── README.md              # Data format documentation
├── models/                     # Model architectures
│   ├── baseline_model.py      # TCN-BiGRU-Attention (baseline)
│   ├── local_attention_hybrid.py # Local Hybrid (best)
│   ├── deep_tcn_model.py      # Deep TCN with SE attention
│   ├── conformer_model.py     # Conformer architecture
│   ├── transformer_model.py   # Transformer encoder
│   ├── components.py          # Shared components (TCN, Attention, FocalLoss)
│   └── README.md              # Architecture details
├── train/                      # Training scripts
│   ├── train.py               # Baseline training script
│   ├── train_transformer.py   # Transformer training script
│   ├── config.py              # Hyperparameters and settings
│   └── utils.py               # Training utilities
├── evaluate/                   # Evaluation scripts
│   ├── evaluate.py            # Model evaluation
│   └── visualize.py           # Attention visualization
├── checkpoints/                # Model checkpoints
│   ├── best_baseline.pt       # Baseline checkpoint (95.40%)
│   └── ...
├── checkpoints_local_hybrid/   # Local Hybrid checkpoints
│   ├── best_local_hybrid.pt   # Best model (97.52%)
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── test_results.json
├── docs/                       # Documentation
│   ├── COMPREHENSIVE_MODEL_COMPARISON.md # Full 5-model comparison
│   ├── HYPERPARAMETER_TUNING.md # Baseline tuning results
│   ├── DATA_SETUP.md          # Data preparation guide
│   └── PREPROCESSING_PIPELINE.md # Preprocessing details
├── run_train_all.py            # Unified training script (all models)
├── create_comparison_plots.py  # Generate comparison visualizations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hyunku9566/Hybrid-Attention-Based_Temporal-Modeling.git
cd baseline-adl-recognition

# Create virtual environment
conda create -n adl python=3.8
conda activate adl

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

**Option 1: Use Existing Data (Recommended for reproduction)**

If you have the original ADL dataset:

```bash
# Setup data links (no copying, saves space!)
chmod +x setup_data.sh
./setup_data.sh
```

This links to your existing data:
- 📁 `adl_noerror/` - 120 normal activity files
- 📁 `adl_error/` - 90 error activity files  
- 📁 `processed_all/` - 440 preprocessed binary matrices
- 📊 `dataset_with_lengths_v3.npz` - Ready-to-use dataset (7,066 sequences)

**Option 2: Prepare Your Own Data**

```bash
# Build features from your preprocessed CSVs
python data/build_features.py \
    --data_dir data/raw/your_data \
    --output data/processed/dataset.npz \
    --T 100 --stride 5
```

**See [docs/DATA_SETUP.md](docs/DATA_SETUP.md) for detailed instructions.**

### Training

```bash
# Train with optimal hyperparameters (95.40% accuracy)
python train/train.py \
    --data_path data/processed/dataset_with_lengths_v3.npz \
    --epochs 50 --batch_size 32 --lr 3e-4 \
    --dropout 0.1 --hidden_dim 256 --focal_gamma 1.5 --patience 15

# Or train with your custom dataset
python train/train.py \
    --data_path data/processed/dataset.npz \
    --epochs 50 --batch_size 32 --lr 3e-4 \
    --dropout 0.1 --hidden_dim 256 --focal_gamma 1.5 --patience 15
```

### Evaluation

```bash
# Evaluate on test set
python evaluate/evaluate.py \
    --model_path checkpoints/best_baseline.pt \
    --data_path processed_data/dataset.npz \
    --output_dir results
```

### Inference

```python
import torch
from models.baseline_model import BaselineModel

# Load model with optimal configuration
model = BaselineModel(in_dim=114, hidden=256, classes=5, dropout=0.1)
model.load_state_dict(torch.load('checkpoints/best_baseline.pt', weights_only=False))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(input_sequence)  # [batch, seq_len, features]
    predictions = outputs.argmax(dim=1)
```

## 📊 Detailed Results

### Per-Class Performance

| Activity | Precision | Recall | F1-Score | Avg Length |
|----------|-----------|--------|----------|------------|
| Cooking (t1) | 0.882 | 0.918 | 0.896 | 24.5 steps |
| **Hand washing (t2)** | **0.982** | **0.932** | **0.956** ⭐ | **10.8 steps** |
| Sleeping (t3) | 0.988 | 0.982 | 0.985 | 100.0 steps |
| Taking medicine (t4) | 0.918 | 0.885 | 0.901 | 18.3 steps |
| Eating (t5) | 0.893 | 0.874 | 0.883 | 21.7 steps |

### Why Our Model Excels

**Problem**: Existing HAR models fail on short activities (<15 timesteps)
- SOTA models: t2 F1 = 0.84-0.90
- **Our model: t2 F1 = 0.956** (+5.6-11.6%p improvement)

**Solution**: Data-driven architectural design
- Small receptive field (13 steps) matches median activity duration
- Single attention focus preserves fine-grained patterns
- BiGRU efficiency for short sequences

## 🧪 Ablation Studies

We systematically tested 6 architectural variants:

| Variant | Accuracy | Macro F1 | t2 F1 | Result |
|---------|----------|----------|-------|--------|
| **Baseline** | **93.7%** | **0.924** | **0.956** | ✅ **Optimal** |
| TCN-5 (deeper) | 90.9% | 0.856 | 0.821 | ❌ Over-smoothing |
| Multi-Head 4h | 92.8% | 0.895 | 0.848 | ❌ Feature dilution |
| Multi-Head 8h | 93.0% | 0.889 | 0.857 | ❌ Worse dilution |
| SSL+Graph | 94.5% | 0.907 | 0.822 | ❌ Short seq damage |
| Graph-only | 89.7% | 0.845 | 0.814 | ❌ Implementation issues |
| Adaptive SSL | 93.4% | 0.920 | 0.937 | 🟡 Alternative |

**Key Finding**: *"Simplicity beats complexity for ADL recognition"*

## 📖 Hyperparameters

### 🎯 Optimal Configuration (95.40% Accuracy)

```python
# Model Architecture
in_dim = 114           # Sensor features
hidden = 256           # TCN/BiGRU hidden size (increased from 128)
tcn_blocks = 3         # Number of TCN blocks
dilations = [1, 2, 4]  # Exponential dilation
kernel_size = 3        # TCN kernel size
dropout = 0.1          # TCN dropout (reduced from 0.2)

# Training (Optimized)
batch_size = 32        # Reduced for more frequent updates
learning_rate = 3e-4   # Initial learning rate
weight_decay = 1e-4    # L2 regularization
epochs = 50            # Increased training duration
patience = 15          # Early stopping patience (increased)
loss = FocalLoss(gamma=1.5)  # Reduced focusing (from 2.0)
optimizer = AdamW      # With CosineAnnealingWarmRestarts
```

### 📚 Detailed Tuning Results
See **[docs/HYPERPARAMETER_TUNING.md](docs/HYPERPARAMETER_TUNING.md)** for comprehensive experimentation details, performance comparisons, and tuning insights.

## 📈 Visualization

### Attention Weights

Our model learns interpretable temporal patterns:

```bash
python evaluate/visualize.py \
    --model_path checkpoints/best_baseline.pt \
    --data_path processed_data/dataset.npz \
    --output_dir visualizations
```

Example: Hand washing activity shows high attention on water sensor activation/deactivation.
# Hybrid-Attention-Based_Temporal-Modeling
# Hybrid-Attention-Based_Temporal-Modeling
