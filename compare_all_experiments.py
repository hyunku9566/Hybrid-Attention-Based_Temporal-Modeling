#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare all experiments: T-variants and Embedding ablation
"""

import json
import numpy as np
from pathlib import Path


def load_test_results(checkpoint_dir):
    """Load test results from checkpoint directory"""
    results_file = Path(checkpoint_dir) / 'test_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def main():
    print("="*80)
    print("Complete Experiment Status")
    print("="*80)
    
    # Check datasets
    print("\n📊 Datasets:")
    print("-" * 80)
    
    datasets = {
        'T=100 (Embedded)': 'data/processed/T_variants/dataset_T100.npz',
        'T=50 (Embedded)': 'data/processed/T_variants/dataset_T50.npz',
        'T=150 (Embedded)': 'data/processed/T_variants/dataset_T150.npz',
        'T=100 (Raw Binary)': 'data/processed/dataset_raw_binary.npz',
    }
    
    for name, path in datasets.items():
        if Path(path).exists():
            data = np.load(path, allow_pickle=True)
            X = data['X']
            has_emb = data.get('has_embeddings', True)
            emb_str = "WITH embeddings" if has_emb else "WITHOUT embeddings"
            print(f"✅ {name:25s}: {X.shape} - {emb_str}")
        else:
            print(f"❌ {name:25s}: Not found")
    
    # Check training status
    print("\n\n🎯 Training Status:")
    print("-" * 80)
    
    models = ['baseline', 'deep_tcn', 'local_hybrid', 'conformer', 'transformer']
    
    # T=100 with embeddings (completed)
    print("\n📈 T=100 WITH Embeddings (Completed):")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Status'}")
    print("-" * 80)
    
    for model in models:
        checkpoint_dir = f'checkpoints_{model}'
        results = load_test_results(checkpoint_dir)
        if results:
            acc = results.get('test_accuracy', results.get('test_acc', 0)) * 100
            f1 = results.get('test_f1', 0)
            print(f"{model:<20} {acc:>10.2f}% {f1:>11.4f}    ✅ Complete")
        else:
            print(f"{model:<20} {'---':>10}  {'---':>11}    ⏳ Pending")
    
    # T=50 with embeddings
    print("\n📉 T=50 WITH Embeddings:")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Status'}")
    print("-" * 80)
    
    for model in models:
        checkpoint_dir = f'checkpoints_{model}_T50'
        results = load_test_results(checkpoint_dir)
        if results:
            acc = results.get('test_accuracy', results.get('test_acc', 0)) * 100
            f1 = results.get('test_f1', 0)
            print(f"{model:<20} {acc:>10.2f}% {f1:>11.4f}    ✅ Complete")
        else:
            print(f"{model:<20} {'---':>10}  {'---':>11}    🔄 Training")
    
    # T=150 with embeddings
    print("\n📈 T=150 WITH Embeddings:")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Status'}")
    print("-" * 80)
    
    for model in models:
        checkpoint_dir = f'checkpoints_{model}_T150'
        results = load_test_results(checkpoint_dir)
        if results:
            acc = results.get('test_accuracy', results.get('test_acc', 0)) * 100
            f1 = results.get('test_f1', 0)
            print(f"{model:<20} {acc:>10.2f}% {f1:>11.4f}    ✅ Complete")
        else:
            print(f"{model:<20} {'---':>10}  {'---':>11}    🔄 Training")
    
    # T=100 without embeddings (raw binary)
    print("\n🔬 T=100 WITHOUT Embeddings (Raw Binary):")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Status'}")
    print("-" * 80)
    
    for model in models:
        checkpoint_dir = f'checkpoints_{model}_raw'
        results = load_test_results(checkpoint_dir)
        if results:
            acc = results.get('test_accuracy', results.get('test_acc', 0)) * 100
            f1 = results.get('test_f1', 0)
            print(f"{model:<20} {acc:>10.2f}% {f1:>11.4f}    ✅ Complete")
        else:
            print(f"{model:<20} {'---':>10}  {'---':>11}    🔄 Training")
    
    print("\n" + "="*80)
    print("💡 Summary:")
    print("="*80)
    print("• T-Variant Experiments: Testing sequence length impact (T=50, 100, 150)")
    print("• Embedding Ablation: Comparing WITH (F=114) vs WITHOUT (F=28) embeddings")
    print("• Total Experiments: 20 models (15 T-variants + 5 raw binary)")
    print("• Purpose: Understand both sequence length and embedding contribution")
    print("="*80)


if __name__ == '__main__':
    main()
