#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Extraction for ADL Recognition

This script builds sequences from preprocessed sensor data for model training.

Input:
    - Processed CSV files with binary sensor readings
    - Sensor embeddings (optional, from Word2Vec-style training)

Output:
    - dataset.npz containing:
        - X: Feature sequences [N, T, F]
        - y: Activity labels [N]
        - filenames: Source file names
        - class_names: Activity class names

Usage:
    python build_features.py \
        --data_dir processed_all \
        --emb_dir embeddings_all_augmented \
        --output dataset.npz \
        --T 100 \
        --stride 5
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_label_from_filename(fname: str) -> str:
    """
    Extract activity label from filename
    
    Examples:
        'cooking_p001_001.csv' → 'cooking'
        't1_p002_005.csv' → 't1'
    """
    fname = Path(fname).stem
    match = re.match(r'([a-z0-9_]+)_p\d+', fname, re.IGNORECASE)
    if match:
        return match.group(1)
    return fname.split('_')[0]


def load_sensor_embeddings(emb_dir: str, vocab_size: int = 27, embed_dim: int = 32) -> np.ndarray:
    """
    Load pre-trained sensor embeddings
    
    Args:
        emb_dir: Directory containing embeddings
        vocab_size: Number of sensors (default: 27)
        embed_dim: Embedding dimension (default: 32)
    
    Returns:
        Embedding matrix [vocab_size, embed_dim]
    """
    emb_path = Path(emb_dir) / 'sensor_embeddings.npy'
    
    if emb_path.exists():
        print(f"📦 Loading embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        print(f"   Shape: {embeddings.shape}")
        return embeddings
    else:
        print(f"⚠️  No embeddings found, using random initialization")
        return np.random.randn(vocab_size, embed_dim) * 0.01


def build_sequence_features(
    csv_path: str,
    T: int = 100,
    stride: int = 5,
    sensor_cols: List[str] = None
) -> List[np.ndarray]:
    """
    Build sliding window sequences from a single CSV file
    
    Args:
        csv_path: Path to CSV file
        T: Target sequence length (default: 100)
        stride: Sliding window stride (default: 5)
        sensor_cols: List of sensor column names (if None, auto-detect M* columns)
    
    Returns:
        List of sequences, each [T, num_sensors]
    """
    df = pd.read_csv(csv_path)
    
    # Auto-detect sensor columns
    if sensor_cols is None:
        sensor_cols = [col for col in df.columns if col.startswith('M')]
        sensor_cols = sorted(sensor_cols, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Extract sensor matrix
    sensor_data = df[sensor_cols].values  # [L, num_sensors]
    L, num_sensors = sensor_data.shape
    
    sequences = []
    
    if L <= T:
        # Short sequence: zero-pad to length T
        padded = np.zeros((T, num_sensors), dtype=np.float32)
        padded[:L] = sensor_data
        sequences.append(padded)
    else:
        # Long sequence: sliding window
        for start in range(0, L - T + 1, stride):
            window = sensor_data[start:start + T]
            sequences.append(window.astype(np.float32))
    
    return sequences


def build_dataset(
    data_dir: str,
    T: int = 100,
    stride: int = 5,
    embeddings: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build complete dataset from all CSV files
    
    Args:
        data_dir: Directory containing processed CSV files
        T: Sequence length
        stride: Sliding window stride
        embeddings: Sensor embeddings (optional)
    
    Returns:
        X: Feature sequences [N, T, F]
        y: Activity labels [N]
        filenames: Source file for each sequence
        class_names: Unique class labels
    """
    csv_files = sorted(Path(data_dir).glob('*.csv'))
    print(f"📁 Found {len(csv_files)} CSV files in {data_dir}")
    
    all_sequences = []
    all_labels = []
    all_filenames = []
    
    # Build label mapping
    label_to_idx = {}
    
    for csv_path in tqdm(csv_files, desc="Processing files"):
        # Extract label
        label = parse_label_from_filename(csv_path.name)
        
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
        
        label_idx = label_to_idx[label]
        
        # Build sequences
        sequences = build_sequence_features(str(csv_path), T, stride)
        
        # Store
        all_sequences.extend(sequences)
        all_labels.extend([label_idx] * len(sequences))
        all_filenames.extend([csv_path.name] * len(sequences))
    
    # Convert to numpy
    X = np.stack(all_sequences)  # [N, T, num_sensors]
    y = np.array(all_labels, dtype=np.int64)
    
    # Feature engineering: add embeddings if provided
    if embeddings is not None:
        print(f"🔧 Adding sensor embeddings ({embeddings.shape})")
        N, T, num_sensors = X.shape
        embed_dim = embeddings.shape[1]
        
        # For each timestep, concatenate raw sensor + embedding
        X_with_embed = np.zeros((N, T, num_sensors + embed_dim), dtype=np.float32)
        X_with_embed[:, :, :num_sensors] = X
        
        # Add average embedding across active sensors
        for i in range(N):
            for t in range(T):
                active_sensors = np.where(X[i, t] > 0)[0]
                if len(active_sensors) > 0:
                    avg_embed = embeddings[active_sensors].mean(axis=0)
                    X_with_embed[i, t, num_sensors:] = avg_embed
        
        X = X_with_embed
    
    # Create class name list
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(label_to_idx))]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total sequences: {X.shape[0]}")
    print(f"   Sequence length: {X.shape[1]}")
    print(f"   Features per timestep: {X.shape[2]}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Classes: {class_names}")
    
    # Class distribution
    print(f"\n📈 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        print(f"   {class_names[cls_idx]:15s}: {count:5d} ({count/len(y)*100:.1f}%)")
    
    return X, y, all_filenames, class_names


def main():
    parser = argparse.ArgumentParser(description='Build feature dataset for ADL recognition')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed CSV files')
    parser.add_argument('--emb_dir', type=str, default=None,
                       help='Directory containing sensor embeddings (optional)')
    parser.add_argument('--output', type=str, default='dataset.npz',
                       help='Output dataset file (.npz)')
    parser.add_argument('--T', type=int, default=100,
                       help='Target sequence length (default: 100)')
    parser.add_argument('--stride', type=int, default=5,
                       help='Sliding window stride (default: 5)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🏠 ADL Feature Extraction")
    print("="*80)
    
    # Load embeddings if provided
    embeddings = None
    if args.emb_dir:
        embeddings = load_sensor_embeddings(args.emb_dir)
    
    # Build dataset
    X, y, filenames, class_names = build_dataset(
        args.data_dir,
        T=args.T,
        stride=args.stride,
        embeddings=embeddings
    )
    
    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving dataset to {output_path}")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        filenames=filenames,
        class_names=class_names
    )
    
    print(f"✅ Dataset saved successfully!")
    print(f"   Size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print("="*80)


if __name__ == '__main__':
    main()
