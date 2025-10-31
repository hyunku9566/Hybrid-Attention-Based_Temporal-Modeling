#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create datasets with different sequence lengths (T values)

This script generates three versions of the dataset:
- T=50: Short sequences (fast training, less context)
- T=100: Medium sequences (current default)
- T=150: Long sequences (more context, slower training)

Usage:
    python create_T_variants.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


def resample_sequence(seq, target_length):
    """
    Resample sequence to target length
    
    Args:
        seq: Original sequence [L, F]
        target_length: Target length T
    
    Returns:
        Resampled sequence [T, F]
    """
    L, F = seq.shape
    
    if L == target_length:
        return seq
    elif L < target_length:
        # Pad with zeros
        padded = np.zeros((target_length, F), dtype=seq.dtype)
        padded[:L] = seq
        return padded
    else:
        # Downsample or extract central window
        # Use central window to preserve important parts
        start = (L - target_length) // 2
        return seq[start:start + target_length]


def create_dataset_with_T(original_data_path, target_T, output_path):
    """
    Create dataset with specified sequence length
    
    Args:
        original_data_path: Path to original dataset (T=100)
        target_T: Target sequence length
        output_path: Output path for new dataset
    """
    print(f"\n{'='*80}")
    print(f"Creating dataset with T={target_T}")
    print(f"{'='*80}")
    
    # Load original data
    print(f"📂 Loading original data from {original_data_path}...")
    data = np.load(original_data_path, allow_pickle=True)
    
    X_orig = data['X']  # [N, T_orig, F]
    y = data['y']
    class_names = data['class_names']
    seq_lengths = data['seq_lengths']
    filenames = data.get('filenames', np.array([]))
    
    N, T_orig, F = X_orig.shape
    print(f"   Original shape: {X_orig.shape}")
    print(f"   Classes: {class_names}")
    print(f"   Samples: {N}")
    
    # Resample each sequence
    print(f"\n🔄 Resampling sequences to T={target_T}...")
    X_new = np.zeros((N, target_T, F), dtype=X_orig.dtype)
    new_lengths = np.zeros(N, dtype=np.int32)
    
    for i in tqdm(range(N), desc="Processing"):
        orig_len = seq_lengths[i]
        X_new[i] = resample_sequence(X_orig[i], target_T)
        new_lengths[i] = min(orig_len, target_T)
    
    print(f"\n   New shape: {X_new.shape}")
    print(f"   New length stats:")
    print(f"      Min: {new_lengths.min()}")
    print(f"      Max: {new_lengths.max()}")
    print(f"      Mean: {new_lengths.mean():.2f}")
    print(f"      Median: {np.median(new_lengths):.2f}")
    
    # Count length distribution
    print(f"\n   Length distribution:")
    unique, counts = np.unique(new_lengths, return_counts=True)
    for length, count in zip(unique[:10], counts[:10]):
        print(f"      Length {length}: {count} ({count/N*100:.1f}%)")
    
    # Save new dataset
    print(f"\n💾 Saving to {output_path}...")
    np.savez(
        output_path,
        X=X_new,
        y=y,
        seq_lengths=new_lengths,
        class_names=class_names,
        filenames=filenames,
        T=target_T
    )
    
    print(f"✅ Done! Dataset saved with T={target_T}")
    print(f"   File size: {Path(output_path).stat().st_size / (1024**2):.2f} MB")


def main():
    """Create datasets with T=50, 100, 150"""
    
    # Original dataset (T=100)
    original_path = 'data/processed/dataset_with_lengths_v3.npz'
    
    # Output directory
    output_dir = Path('data/processed/T_variants')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Creating Dataset Variants with Different T Values")
    print("="*80)
    
    # T=50 (short sequences)
    create_dataset_with_T(
        original_path,
        target_T=50,
        output_path=output_dir / 'dataset_T50.npz'
    )
    
    # T=100 (current default - just copy with consistent format)
    create_dataset_with_T(
        original_path,
        target_T=100,
        output_path=output_dir / 'dataset_T100.npz'
    )
    
    # T=150 (long sequences)
    create_dataset_with_T(
        original_path,
        target_T=150,
        output_path=output_dir / 'dataset_T150.npz'
    )
    
    print("\n" + "="*80)
    print("✅ All datasets created successfully!")
    print("="*80)
    print("\nDataset files:")
    for T in [50, 100, 150]:
        path = output_dir / f'dataset_T{T}.npz'
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"   T={T:3d}: {path} ({size_mb:.2f} MB)")
    
    print("\n💡 Usage:")
    print(f"   python run_train_all.py --model baseline --data_path {output_dir}/dataset_T50.npz")
    print(f"   python run_train_all.py --model local_hybrid --data_path {output_dir}/dataset_T150.npz")


if __name__ == '__main__':
    main()
