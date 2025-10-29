# 🏠 Lightweight ADL Recognition with TCN-BiGRU-Attention

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight yet highly accurate deep learning model for Activities of Daily Living (ADL) recognition in smart home environments.

## 🏆 Key Results

- **93.7% Accuracy** (State-of-the-art)
- **0.924 Macro F1-Score**
- **0.58M Parameters** (Edge-deployable)
- **2.66ms Inference Time** (Real-time capable)

## 🎯 Highlights

Our model **outperforms** 5 state-of-the-art HAR models:
- ✅ **Best overall accuracy** among lightweight models (<1M params)
- ✅ **Exceptional short-activity recognition** (0.956 F1 on hand washing)
- ✅ **Balanced performance** across all activity classes
- ✅ **Interpretable** through attention weight visualization

## 📊 Performance Comparison

| Model | Accuracy | Macro F1 | Parameters | Inference Time |
|-------|----------|----------|------------|----------------|
| **Ours (Baseline)** | **93.7%** ⭐ | **0.924** ⭐ | 0.58M | 2.66ms |
| Transformer | 93.0% | 0.910 | 0.82M | 2.69ms |
| AttnSense | 92.0% | 0.900 | 0.31M | 1.90ms |
| LSTM-FCN | 91.0% | 0.890 | 0.51M | 1.74ms |
| DeepConvLSTM | 89.0% | 0.870 | 0.45M | 1.68ms |
| CNN-LSTM | 88.0% | 0.860 | 0.21M | 1.17ms |

## 🏗️ Architecture

```
Input (T=100, F=114)
       ↓
[Feature Projection] Linear(114 → 128)
       ↓
[Temporal Encoding] TCN (3 blocks, dilation=1,2,4)
       ↓
[Sequential Modeling] BiGRU (hidden=128, bidirectional)
       ↓
[Temporal Attention] Additive Attention (Bahdanau-style)
       ↓
[Classification] FC → ReLU → Dropout → FC(5 classes)
       ↓
Output: [cooking, hand_washing, sleeping, medicine, eating]
```

### Key Components

1. **Temporal Convolutional Network (TCN)**
   - 3 dilated convolution blocks (dilation: 1, 2, 4)
   - Receptive field: 13 timesteps
   - Causal padding for real-time inference

2. **Bidirectional GRU**
   - Hidden size: 128
   - 25% fewer parameters than BiLSTM
   - Captures past and future context

3. **Additive Attention**
   - Bahdanau-style attention mechanism
   - Single focus (vs. multi-head over-smoothing)
   - Interpretable attention weights

## 📂 Project Structure

```
baseline-adl-recognition/
├── data/                       # Data preprocessing
│   ├── preprocess.py          # Main preprocessing pipeline
│   ├── build_features.py      # Feature extraction from sensors
│   └── README.md              # Data format documentation
├── models/                     # Model architecture
│   ├── baseline_model.py      # Main TCN-BiGRU-Attention model
│   ├── components.py          # TCN, Attention components
│   └── README.md              # Architecture details
├── train/                      # Training scripts
│   ├── train.py               # Main training script
│   ├── config.py              # Hyperparameters and settings
│   └── utils.py               # Training utilities
├── evaluate/                   # Evaluation scripts
│   ├── evaluate.py            # Model evaluation
│   ├── visualize.py           # Attention visualization
│   └── compare_models.py      # Benchmark comparison
├── checkpoints/                # Pretrained models
│   └── best_baseline.pt       # Best model checkpoint
├── docs/                       # Documentation
│   ├── PAPER.md               # Full paper draft
│   └── EXPERIMENTS.md         # Ablation studies
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/baseline-adl-recognition.git
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
# Train with existing preprocessed dataset (fastest)
python train/train.py \
    --data_path data/processed/dataset_with_lengths_v3.npz \
    --epochs 30 --batch_size 64 --lr 3e-4

# Or train with your custom dataset
python train/train.py \
    --data_path data/processed/dataset.npz \
    --epochs 30 --batch_size 64 --lr 3e-4
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

# Load model
model = BaselineModel(in_dim=114, hidden=128, classes=5)
model.load_state_dict(torch.load('checkpoints/best_baseline.pt'))
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

```python
# Model Architecture
in_dim = 114           # Sensor features
hidden = 128           # TCN/BiGRU hidden size
tcn_blocks = 3         # Number of TCN blocks
dilations = [1, 2, 4]  # Exponential dilation
kernel_size = 3        # TCN kernel size
dropout = 0.1          # TCN dropout (0.2 for classifier)

# Training
batch_size = 64
learning_rate = 3e-4
weight_decay = 1e-5
epochs = 30
patience = 10          # Early stopping
loss = FocalLoss(gamma=2.0)  # Handles class imbalance
optimizer = AdamW
```

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
