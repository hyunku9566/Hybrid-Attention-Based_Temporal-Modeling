# Baseline ADL Recognition - Project Structure

## 📁 Complete Directory Structure

```
baseline-adl-recognition/
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data preprocessing
│   ├── README.md               # Data format documentation
│   ├── build_features.py       # Feature extraction script
│   ├── raw/                    # Raw sensor CSV files (not included)
│   ├── embeddings/             # Sensor embeddings (optional)
│   └── processed/              # Processed datasets (not included)
│
├── models/                      # Model architecture
│   ├── __init__.py             # Package initialization
│   ├── README.md               # Architecture documentation
│   ├── components.py           # TCNBlock, AdditiveAttention, FocalLoss
│   └── baseline_model.py       # Main BaselineModel class
│
├── train/                       # Training scripts
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Training configuration
│   ├── utils.py                # Training utilities
│   └── train.py                # Main training script
│
├── evaluate/                    # Evaluation scripts
│   ├── __init__.py             # Package initialization
│   ├── evaluate.py             # Model evaluation
│   └── visualize.py            # Attention visualization
│
├── scripts/                     # Helper scripts
│   ├── quick_start.sh          # Complete workflow script
│   └── evaluate_model.sh       # Quick evaluation script
│
├── checkpoints/                 # Saved model weights
│   └── .gitkeep                # (Checkpoint files not tracked in git)
│
├── results/                     # Evaluation results
│   └── .gitkeep                # (Result files not tracked in git)
│
└── docs/                        # Additional documentation
    └── .gitkeep                # (Paper drafts, experiments notes)
```

## 📄 File Descriptions

### Root Directory

- **README.md** (8.7 KB): Comprehensive project documentation with:
  - Architecture overview and diagram
  - Performance comparison table
  - Installation and usage instructions
  - Quick start guide
  - Citation information

- **LICENSE** (1.1 KB): MIT License

- **CONTRIBUTING.md** (5.3 KB): Guidelines for contributors including:
  - How to report issues
  - Pull request process
  - Coding standards
  - Testing requirements

- **requirements.txt** (351 B): Python dependencies
  - Core: torch>=2.0.0, numpy, pandas, scikit-learn
  - Visualization: matplotlib, seaborn, plotly
  - Utils: tqdm, tabulate

- **.gitignore** (734 B): Excludes:
  - Python cache files (\*.pyc, __pycache__)
  - Model weights (\*.pt, \*.pth)
  - Data files (\*.csv, \*.npz)
  - Results (plots, logs)

### data/ Directory (3 files, ~11 KB)

- **README.md** (3.2 KB): Data format and preprocessing documentation
- **build_features.py** (8.2 KB): Feature extraction script
  - Sliding window extraction (T=100, stride=5)
  - Time feature encoding
  - Sensor embedding integration
  - Output: dataset.npz with X, y, filenames, class_names

### models/ Directory (4 files, ~17 KB)

- **__init__.py** (611 B): Package exports
- **README.md** (4.0 KB): Architecture documentation
- **components.py** (6.7 KB): Building blocks
  - `TCNBlock`: Dilated causal convolutions
  - `AdditiveAttention`: Bahdanau-style attention
  - `FocalLoss`: Class imbalance handling
- **baseline_model.py** (6.5 KB): Main model class
  - Architecture: Linear → TCN(3) → BiGRU → Attention → Classifier
  - Methods: forward(), count_parameters()

### train/ Directory (4 files, ~24 KB)

- **__init__.py** (628 B): Package exports
- **config.py** (4.2 KB): Training configuration
  - `TrainingConfig`: Default hyperparameters
  - Alternative configs: FastTrainConfig, LargeModelConfig, SmallModelConfig
- **utils.py** (10.7 KB): Training utilities
  - `ADLDataset`: PyTorch Dataset
  - `load_data()`: Data loading and splitting
  - `create_dataloaders()`: DataLoader creation with WeightedRandomSampler
  - `train_epoch()`, `evaluate()`: Training loop functions
  - `plot_training_history()`, `plot_confusion_matrix()`: Visualization
- **train.py** (9.2 KB): Main training script
  - CLI interface with argparse
  - Training loop with early stopping
  - Checkpoint saving
  - Test set evaluation

### evaluate/ Directory (3 files, ~15 KB)

- **__init__.py** (74 B): Package initialization
- **evaluate.py** (6.5 KB): Model evaluation
  - Comprehensive metrics calculation
  - Per-class precision/recall/F1
  - Confusion matrix
  - Results saved to JSON
- **visualize.py** (9.1 KB): Visualization tools
  - Attention weight plots
  - Prediction confidence plots
  - Sensor activation plots

### scripts/ Directory (2 files, ~3.7 KB)

- **quick_start.sh** (2.7 KB): Complete workflow
  - Data preprocessing → Training → Evaluation → Visualization
  - Executable: `chmod +x scripts/quick_start.sh`
- **evaluate_model.sh** (1.0 KB): Quick evaluation
  - Usage: `./evaluate_model.sh <checkpoint_path>`

### Other Directories

- **checkpoints/**: Saved model weights (.pt files)
  - `.gitkeep` placeholder (actual weights not tracked)
  - Expected: `best_baseline.pt`, `last_epoch.pt`

- **results/**: Evaluation outputs
  - `.gitkeep` placeholder
  - Expected: confusion matrices, classification reports, visualizations

- **docs/**: Additional documentation
  - `.gitkeep` placeholder
  - For paper drafts, experiment notes, etc.

## 📊 Key Statistics

- **Total Python files**: 13
- **Total lines of code**: ~2,500
- **Documentation files**: 5 (README, CONTRIBUTING, 3× module READMEs)
- **Shell scripts**: 2
- **Packages**: 3 (models, train, evaluate)

## 🔗 File Dependencies

```
train.py
  ├── models/baseline_model.py
  │   └── models/components.py
  ├── train/config.py
  └── train/utils.py

evaluate.py
  ├── models/baseline_model.py
  │   └── models/components.py
  └── train/utils.py (for data loading)

visualize.py
  └── models/baseline_model.py
      └── models/components.py
```

## 🚀 Usage Workflows

### Workflow 1: From Scratch

```bash
# 1. Prepare data
python data/build_features.py --data_dir data/raw --output data/processed/dataset.npz

# 2. Train model
python train/train.py --data_path data/processed/dataset.npz --epochs 30

# 3. Evaluate
python evaluate/evaluate.py --checkpoint checkpoints/best_baseline.pt

# 4. Visualize
python evaluate/visualize.py --checkpoint checkpoints/best_baseline.pt
```

### Workflow 2: Quick Start

```bash
cd scripts
./quick_start.sh
```

### Workflow 3: Evaluation Only

```bash
cd scripts
./evaluate_model.sh ../checkpoints/best_baseline.pt
```

## 📦 Package Organization

### models Package
- **Purpose**: Model architecture components
- **Exports**: BaselineModel, TCNBlock, AdditiveAttention, FocalLoss
- **Usage**: `from models import BaselineModel`

### train Package
- **Purpose**: Training utilities and configuration
- **Exports**: TrainingConfig, data loaders, training functions
- **Usage**: `from train import TrainingConfig, load_data`

### evaluate Package
- **Purpose**: Evaluation and visualization
- **Exports**: None (use scripts directly)
- **Usage**: Run scripts via CLI

## 🎯 Next Steps

1. **Add raw data** to `data/raw/` directory
2. **Run preprocessing**: `python data/build_features.py`
3. **Train model**: `python train/train.py`
4. **Evaluate**: `python evaluate/evaluate.py --checkpoint checkpoints/best_baseline.pt`
5. **Visualize**: `python evaluate/visualize.py --checkpoint checkpoints/best_baseline.pt`

## 📝 Notes

- All Python files have comprehensive docstrings
- Scripts include CLI interfaces with `--help` option
- Code follows PEP 8 style guide
- Ready for GitHub upload and public release
- MIT License allows free use and modification

---

**Total Project Size**: ~70 KB (excluding data and checkpoints)  
**Last Updated**: 2024-10-29
