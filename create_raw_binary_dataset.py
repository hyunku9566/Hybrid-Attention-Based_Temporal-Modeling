#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create dataset WITHOUT sensor embeddings (raw binary data only)

This creates a baseline comparison dataset using only binary sensor readings
without any Word2Vec-style embeddings to demonstrate the value of embeddings.

Comparison:
- Current dataset: F=114 (51 binary + 63 embedding features)
- This dataset: F=51 (51 binary sensors only)

Usage:
    python create_raw_binary_dataset.py
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_label_from_filename(fname: str) -> str:
    """Extract activity label from filename"""
    fname = Path(fname).stem
    # Remove 'processed_' prefix if exists
    fname = fname.replace('processed_', '')
    
    # Match patterns like 'p01.t1' → 't1'
    match = re.search(r'\.t(\d+)', fname)
    if match:
        task_num = match.group(1)
        return f't{task_num}'
    
    # Match patterns like 't1_' or '_t1'
    match = re.search(r'[._]t(\d+)', fname)
    if match:
        task_num = match.group(1)
        return f't{task_num}'
    
    # If no task found, return None (will be filtered out)
    return None


def load_binary_matrix_from_csv(
    csv_path: str, 
    all_sensor_cols: List[str]
) -> np.ndarray:
    """
    Load binary sensor matrix from CSV with unified sensor columns
    
    Args:
        csv_path: Path to CSV file
        all_sensor_cols: List of all possible sensor column names
    
    Returns:
        data: Binary matrix [L, num_sensors] aligned to all_sensor_cols
    """
    df = pd.read_csv(csv_path)
    
    # Find sensor columns in this file
    file_sensor_cols = [col for col in df.columns if col in all_sensor_cols]
    
    if len(file_sensor_cols) == 0:
        raise ValueError(f"No sensor columns found in {csv_path}")
    
    # Extract binary matrix and align to all_sensor_cols
    L = len(df)
    num_sensors = len(all_sensor_cols)
    data = np.zeros((L, num_sensors), dtype=np.float32)
    
    # Fill in available sensors
    for i, sensor in enumerate(all_sensor_cols):
        if sensor in file_sensor_cols:
            data[:, i] = df[sensor].values.astype(np.float32)
    
    return data


def build_sequences(
    sensor_data: np.ndarray,
    T: int = 100,
    stride: int = 5
) -> List[np.ndarray]:
    """
    Build sliding window sequences
    
    Args:
        sensor_data: Binary sensor matrix [L, num_sensors]
        T: Sequence length
        stride: Sliding window stride
    
    Returns:
        List of sequences, each [T, num_sensors]
    """
    L, num_sensors = sensor_data.shape
    sequences = []
    
    if L <= T:
        # Short sequence: zero-pad
        padded = np.zeros((T, num_sensors), dtype=np.float32)
        padded[:L] = sensor_data
        sequences.append(padded)
    else:
        # Long sequence: sliding window
        for start in range(0, L - T + 1, stride):
            window = sensor_data[start:start + T]
            sequences.append(window.astype(np.float32))
    
    return sequences


def create_raw_binary_dataset(
    data_dir: str,
    T: int = 100,
    stride: int = 5,
    output_path: str = 'data/processed/dataset_raw_binary.npz'
) -> None:
    """
    Create dataset from raw binary sensor data (no embeddings)
    
    Args:
        data_dir: Directory with CSV files containing binary matrices
        T: Sequence length
        stride: Sliding window stride
        output_path: Output path for dataset
    """
    print("="*80)
    print("Creating Raw Binary Dataset (No Embeddings)")
    print("="*80)
    
    # Find all binary matrix CSV files (*_wide_1s.csv)
    csv_files = list(Path(data_dir).glob('*_wide_1s.csv'))
    csv_files = sorted(csv_files)
    print(f"\n📁 Found {len(csv_files)} CSV files in {data_dir}")
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    # Discover all unique sensor columns across all files
    print(f"\n🔍 Discovering all unique sensors...")
    all_sensor_cols = set()
    for csv_path in tqdm(csv_files, desc="Scanning files"):
        df = pd.read_csv(str(csv_path), nrows=1)  # Just read header
        sensor_cols = [col for col in df.columns if re.match(r'M\d+|AD\d+-[AC]', col)]
        all_sensor_cols.update(sensor_cols)
    
    # Sort sensor columns: M01-M99, then AD sensors
    all_sensor_cols = sorted(
        all_sensor_cols,
        key=lambda x: (
            0 if x.startswith('M') else 1,  # M sensors first
            int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
            x
        )
    )
    
    print(f"✅ Found {len(all_sensor_cols)} unique sensors:")
    print(f"   {all_sensor_cols}")
    print()
    
    # Build sequences
    all_sequences = []
    all_labels = []
    all_filenames = []
    all_lengths = []
    label_to_idx = {}
    
    print(f"🔄 Processing files and building sequences...")
    for csv_path in tqdm(csv_files, desc="Building sequences"):
        try:
            # Load binary matrix with unified sensor columns
            sensor_data = load_binary_matrix_from_csv(str(csv_path), all_sensor_cols)
            
            # Extract label
            label = parse_label_from_filename(csv_path.name)
            
            # Skip files without valid task labels (e.g., 'all' or no task)
            if label is None or not label.startswith('t'):
                continue
            
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
            
            label_idx = label_to_idx[label]
            
            # Build sequences
            sequences = build_sequences(sensor_data, T=T, stride=stride)
            
            for seq in sequences:
                all_sequences.append(seq)
                all_labels.append(label_idx)
                all_filenames.append(csv_path.name)
                # Record actual length (before padding)
                actual_len = min(len(sensor_data), T)
                all_lengths.append(actual_len)
                
        except Exception as e:
            print(f"\n⚠️  Error processing {csv_path.name}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.stack(all_sequences, axis=0)  # [N, T, F]
    y = np.array(all_labels, dtype=np.int64)
    seq_lengths = np.array(all_lengths, dtype=np.int32)
    filenames = np.array(all_filenames)
    
    # Get class names
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = np.array([idx_to_label[i] for i in range(len(label_to_idx))])
    
    print(f"\n✅ Dataset created:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features: {X.shape[2]} (RAW BINARY SENSORS ONLY)")
    print(f"   Classes: {class_names}")
    print(f"   Samples per class:")
    for i, name in enumerate(class_names):
        count = (y == i).sum()
        print(f"      {name}: {count}")
    
    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    np.savez(
        output_path,
        X=X,
        y=y,
        seq_lengths=seq_lengths,
        filenames=filenames,
        class_names=class_names,
        sensor_cols=np.array(all_sensor_cols),  # Save unified sensor columns
        T=T,
        has_embeddings=False  # Mark as raw binary only
    )
    
    file_size = output_path.stat().st_size / (1024**2)
    print(f"✅ Saved! File size: {file_size:.2f} MB")
    
    print("\n" + "="*80)
    print("✅ Raw Binary Dataset Created Successfully!")
    print("="*80)
    print(f"\n📊 Comparison:")
    print(f"   Original dataset: F=114 (51 binary + 63 embedding)")
    print(f"   This dataset:     F={X.shape[2]} (binary sensors only)")
    print(f"   Reduction:        {114 - X.shape[2]} features removed")
    print(f"\n💡 This dataset can be used to demonstrate the value of embeddings!")


def main():
    parser = argparse.ArgumentParser(description='Create raw binary dataset without embeddings')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/lee/research-hub/hyunku/iot/iot-data/processed',
                       help='Directory with binary matrix CSV files')
    parser.add_argument('--T', type=int, default=100,
                       help='Sequence length (default: 100)')
    parser.add_argument('--stride', type=int, default=5,
                       help='Sliding window stride (default: 5)')
    parser.add_argument('--output', type=str,
                       default='data/processed/dataset_raw_binary.npz',
                       help='Output path for dataset')
    
    args = parser.parse_args()
    
    create_raw_binary_dataset(
        data_dir=args.data_dir,
        T=args.T,
        stride=args.stride,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
