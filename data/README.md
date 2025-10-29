# Data Directory

This directory contains data preprocessing scripts and processed datasets.

## Directory Structure

```
data/
├── build_features.py          # Feature extraction script
├── raw/                        # Raw sensor data (CSV files)
│   ├── t1/
│   ├── t2/
│   ├── t3/
│   ├── t4/
│   └── t5/
├── embeddings/                 # Sensor embeddings (optional)
│   └── sensor_embeddings.npz
└── processed/                  # Processed datasets
    ├── dataset.npz
    └── metadata.json
```

## Data Format

### Raw Data (CSV)

Each CSV file should contain sensor readings with the following structure:
- **Columns**: Sensor IDs (e.g., `M01`, `M02`, ..., `M51`)
- **Rows**: Time steps
- **Values**: Binary activations (0 or 1) or continuous values

Example:
```csv
M01,M02,M03,...,M51
0,1,0,...,1
1,1,0,...,0
...
```

### Processed Data (NPZ)

The `build_features.py` script generates a `.npz` file with:
- **X**: Input sequences, shape `(N, T, D)`
  - N: Number of samples
  - T: Sequence length (default: 100)
  - D: Feature dimension (114 = 51 sensors × 2 + 12 time features)
- **y**: Labels, shape `(N,)` - integers 0-4 representing classes t1-t5
- **filenames**: Original filenames for each sample
- **class_names**: List of class names `['t1', 't2', 't3', 't4', 't5']`

### Sensor Embeddings (Optional)

If using pre-trained sensor embeddings:
- **sensor_embeddings.npz**: Contains learned embeddings for each sensor
- Shape: `(n_sensors, embedding_dim)`

## Usage

### 1. Prepare Raw Data

Place your raw CSV files in the appropriate class directories:

```bash
data/raw/t1/*.csv
data/raw/t2/*.csv
...
```

### 2. Build Features

Run the feature extraction script:

```bash
python data/build_features.py \
    --data_dir data/raw \
    --output data/processed/dataset.npz \
    --T 100 \
    --stride 5
```

**Arguments:**
- `--data_dir`: Directory containing raw CSV files
- `--output`: Output path for processed dataset
- `--T`: Sequence length (default: 100)
- `--stride`: Sliding window stride (default: 5)
- `--emb_dir`: (Optional) Directory with sensor embeddings

### 3. Verify Data

Check the processed dataset:

```python
import numpy as np

data = np.load('data/processed/dataset.npz', allow_pickle=True)
print(f"X shape: {data['X'].shape}")  # (N, 100, 114)
print(f"y shape: {data['y'].shape}")  # (N,)
print(f"Classes: {data['class_names']}")  # ['t1', 't2', 't3', 't4', 't5']
```

## Notes

- The preprocessing script automatically handles:
  - Sliding window extraction
  - Time feature encoding (hour, day of week, etc.)
  - Sensor embedding integration (if provided)
  - Class label extraction from filenames
  
- Default sequence length (T=100) corresponds to ~16.7 minutes at 10s intervals
  
- The stride parameter (default=5) controls overlap between consecutive windows

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@article{baseline-adl-2024,
  title={Temporal Convolutional Networks with Additive Attention for Activities of Daily Living Recognition},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```
