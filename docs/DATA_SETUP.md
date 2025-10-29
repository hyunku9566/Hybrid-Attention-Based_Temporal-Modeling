# Using Your Original Data

This document explains how to use your original ADL sensor data with the baseline project.

## 📂 Original Data Structure

Your original data is located at:
```
/home/lee/research-hub/hyunku/iot/iot-data/
├── adl_noerror/          # Normal activities (120 CSV files)
│   ├── p01.t1.csv        # Person 01, Task 1 (cooking)
│   ├── p01.t2.csv        # Person 01, Task 2 (hand washing)
│   ├── ...
│   └── p51.t5.csv
│
├── adl_error/            # Error activities (90 CSV files)
│   ├── p17.t1.csv
│   ├── ...
│   └── p59.t5.csv
│
└── processed_all/        # Preprocessed binary matrices (440 CSV files)
    ├── processed_p01.t1.csv
    ├── ...
    └── processed_p59.t5.csv
```

## 📋 Data Format

### Raw Data Format (adl_noerror, adl_error)

Event log format with timestamps:
```csv
date,time,sensor,message
2008-02-27,12:43:27.416392,M08,ON
2008-02-27,12:43:27.8481,M07,ON
2008-02-27,12:43:28.487061,M09,ON
...
```

- **date**: Date of event
- **time**: Timestamp
- **sensor**: Sensor ID (M01-M51)
- **message**: ON/OFF state

### Preprocessed Format (processed_all)

Binary matrix format (already converted):
```csv
M01,M02,M03,...,M51
0,1,0,...,1
1,1,0,...,0
...
```

- Each row is a time step (10-second intervals)
- Each column is a sensor (51 sensors total)
- Values are binary (0=OFF, 1=ON)

### Activity Labels

Files are named: `p{person_id}.t{task_id}.csv`

| Task | Label | Activity |
|------|-------|----------|
| t1   | 0     | Cooking |
| t2   | 1     | Hand washing |
| t3   | 2     | Sleeping |
| t4   | 3     | Taking medicine |
| t5   | 4     | Eating |

## 🚀 Quick Setup

### Step 1: Run Setup Script

```bash
cd /home/lee/research-hub/hyunku/iot/baseline-adl-recognition
chmod +x setup_data.sh
./setup_data.sh
```

This will create symbolic links to your original data (no copying needed!):
```
baseline-adl-recognition/data/
├── raw/
│   ├── adl_noerror -> /path/to/original/adl_noerror
│   ├── adl_error -> /path/to/original/adl_error
│   └── processed_all -> /path/to/original/processed_all
├── embeddings/
│   └── embeddings_all_augmented -> /path/to/original/embeddings
└── processed/
    └── dataset_with_lengths_v3.npz -> /path/to/original/dataset.npz
```

### Step 2: Choose Your Workflow

#### Option A: Use Existing Preprocessed Dataset (RECOMMENDED ⭐)

The fastest way - use the already prepared dataset:

```bash
python train/train.py \
  --data_path data/processed/dataset_with_lengths_v3.npz \
  --epochs 30 \
  --batch_size 64 \
  --lr 3e-4
```

**Why this is best:**
- ✅ Already preprocessed and tested
- ✅ Includes sensor embeddings
- ✅ Sequence lengths preserved
- ✅ Ready to train immediately

#### Option B: Build New Dataset from Processed CSVs

If you want to customize preprocessing:

```bash
python data/build_features.py \
  --data_dir data/raw/processed_all \
  --emb_dir data/embeddings/embeddings_all_augmented \
  --output data/processed/dataset_custom.npz \
  --T 100 \
  --stride 5
```

Then train:
```bash
python train/train.py \
  --data_path data/processed/dataset_custom.npz \
  --epochs 30
```

#### Option C: Preprocess Raw Event Logs (Advanced)

If you need to start from raw event logs:

1. Copy preprocessing pipeline:
```bash
cp ../iot-data/preprocess_pipeline.py data/
```

2. Preprocess:
```bash
python data/preprocess_pipeline.py \
  --input_dir data/raw/adl_noerror \
  --output_dir data/raw/processed_custom
```

3. Build features:
```bash
python data/build_features.py \
  --data_dir data/raw/processed_custom \
  --output data/processed/dataset.npz
```

## 📊 Dataset Statistics

After running `./setup_data.sh`, you'll have access to:

- **Original files**: 210 event logs (120 normal + 90 error)
- **Preprocessed files**: 440 binary matrices
- **Pre-built dataset**: `dataset_with_lengths_v3.npz`
  - Total sequences: 7,066
  - Sequence length: 100 timesteps
  - Features: 114 dimensions (51 sensors × 2 + 12 time features)
  - Classes: 5 (t1-t5)

### Class Distribution

| Class | Activity | Count | Percentage |
|-------|----------|-------|------------|
| t1 | Cooking | 445 | 6.3% |
| t2 | Hand washing | 697 | 9.9% |
| t3 | Sleeping | 5,924 | 83.8% |
| t4 | Taking medicine | ~500 | ~7% |
| t5 | Eating | ~500 | ~7% |

## 🔧 Customization

### Modify Preprocessing Parameters

Edit `data/build_features.py` arguments:

```bash
python data/build_features.py \
  --data_dir data/raw/processed_all \
  --T 150 \                    # Longer sequences
  --stride 10 \                # Less overlap
  --output data/processed/dataset_long.npz
```

### Use Only Normal or Error Data

```bash
# Normal activities only
python data/build_features.py \
  --data_dir data/raw/adl_noerror \
  --output data/processed/dataset_normal.npz

# Error activities only
python data/build_features.py \
  --data_dir data/raw/adl_error \
  --output data/processed/dataset_error.npz
```

## ✅ Verification

Check that data is properly linked:

```bash
# Check raw data
ls -lh data/raw/

# Check preprocessed data
ls -lh data/processed/

# Verify dataset
python -c "
import numpy as np
data = np.load('data/processed/dataset_with_lengths_v3.npz', allow_pickle=True)
print(f'X shape: {data[\"X\"].shape}')
print(f'y shape: {data[\"y\"].shape}')
print(f'Classes: {data[\"class_names\"]}')
"
```

Expected output:
```
X shape: (7066, 100, 114)
y shape: (7066,)
Classes: ['t1' 't2' 't3' 't4' 't5']
```

## 🎯 Recommended Workflow

For best results with your data:

1. **First time**: Use existing dataset
   ```bash
   ./setup_data.sh
   python train/train.py --data_path data/processed/dataset_with_lengths_v3.npz
   ```

2. **Experimenting**: Rebuild with custom parameters
   ```bash
   python data/build_features.py --data_dir data/raw/processed_all --T 120
   python train/train.py --data_path data/processed/dataset.npz
   ```

3. **Production**: Train on final configuration
   ```bash
   python train/train.py \
     --data_path data/processed/dataset_with_lengths_v3.npz \
     --epochs 50 \
     --batch_size 64 \
     --patience 10
   ```

## 📝 Notes

- **Symbolic links**: Setup script creates links, not copies (saves space!)
- **Data integrity**: Original data remains unchanged
- **Flexibility**: Can switch between different preprocessing versions
- **Git**: Data directories are in `.gitignore` (not tracked)

## 🆘 Troubleshooting

**Problem**: "File not found" error
- **Solution**: Run `./setup_data.sh` first to create links

**Problem**: "Permission denied"
- **Solution**: `chmod +x setup_data.sh`

**Problem**: Want to use different data split
- **Solution**: Modify train/config.py `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`

**Problem**: Need to preprocess raw event logs
- **Solution**: Copy `preprocess_pipeline.py` from iot-data directory
