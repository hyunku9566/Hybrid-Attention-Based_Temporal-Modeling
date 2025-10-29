import numpy as np

# 현재 사용 중인 데이터
current = np.load('data/processed/dataset_with_lengths_v3.npz', allow_pickle=True)

print("=" * 80)
print("📊 Current Dataset Analysis")
print("=" * 80)
print(f"Shape: X={current['X'].shape}, y={current['y'].shape}")
print(f"Feature dimension: {current['X'].shape[2]}")
print(f"Classes: {current['class_names']}")
print(f"\nClass distribution:")
unique, counts = np.unique(current['y'], return_counts=True)
for idx, count in zip(unique, counts):
    print(f"  {current['class_names'][idx]}: {count:4d} ({count/len(current['y'])*100:5.1f}%)")

# 데이터 통계
print(f"\nData statistics:")
print(f"  Mean: {current['X'].mean():.4f}")
print(f"  Std: {current['X'].std():.4f}")
print(f"  Min: {current['X'].min():.4f}")
print(f"  Max: {current['X'].max():.4f}")

# 시퀀스 길이 확인
if 'seq_lengths' in current:
    seq_lengths = current['seq_lengths']
    print(f"\nSequence lengths:")
    print(f"  Mean: {seq_lengths.mean():.2f}")
    print(f"  Median: {np.median(seq_lengths):.2f}")
    print(f"  Min: {seq_lengths.min()}")
    print(f"  Max: {seq_lengths.max()}")
