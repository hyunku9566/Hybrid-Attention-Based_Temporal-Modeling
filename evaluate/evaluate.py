"""
Evaluate Baseline ADL Recognition Model

Comprehensive evaluation script with detailed metrics and analysis.
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_recall_fscore_support
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BaselineModel, FocalLoss
from train.utils import load_data, create_dataloaders, evaluate
from train.config import TrainingConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Baseline Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='../data/processed/dataset.npz',
                       help='Path to dataset')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for results')
    
    return parser.parse_args()


def evaluate_per_class(y_true, y_pred, class_names):
    """
    Detailed per-class evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        
    Returns:
        results: Dict with per-class metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return results


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("📊 Evaluating Baseline ADL Recognition Model")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print()
    
    # Load data
    print("📂 Loading data...")
    train_data, val_data, test_data, class_names = load_data(args.data_path)
    
    if args.split == 'val':
        eval_data = val_data
    else:
        eval_data = test_data
    
    print(f"Evaluation samples: {len(eval_data[0])}")
    print(f"Classes: {class_names}")
    print()
    
    # Create dataloaders
    _, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        use_weighted_sampler=False,
        num_workers=4
    )
    
    eval_loader = val_loader if args.split == 'val' else test_loader
    
    # Load model
    print("🔧 Loading model...")
    model = BaselineModel(
        in_dim=114,
        hidden=TrainingConfig.HIDDEN_DIM,
        classes=len(class_names),
        dropout=TrainingConfig.DROPOUT
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print()
    
    # Evaluate
    print("🔍 Evaluating...")
    criterion = FocalLoss(gamma=2.0)
    
    loss, accuracy, preds, labels = evaluate(model, eval_loader, criterion, device)
    
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    
    # Classification report
    print("📈 Classification Report:")
    print("-" * 80)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)
    
    # Macro F1
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"\n📊 Macro F1: {macro_f1:.4f}")
    print()
    
    # Per-class metrics
    per_class_results = evaluate_per_class(labels, preds, class_names)
    
    print("📋 Per-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for class_name, metrics in per_class_results.items():
        print(f"{class_name:<10} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['support']:<10}")
    print("-" * 80)
    print()
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("🔢 Confusion Matrix:")
    print("-" * 80)
    print("True \\ Pred", end="")
    for class_name in class_names:
        print(f"  {class_name:>6}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12}", end="")
        for j in range(len(class_names)):
            print(f"  {cm[i,j]:>6}", end="")
        print()
    print("-" * 80)
    print()
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        'split': args.split,
        'accuracy': float(accuracy),
        'loss': float(loss),
        'macro_f1': float(macro_f1),
        'per_class': per_class_results,
        'confusion_matrix': cm.tolist(),
        'n_params': n_params
    }
    
    import json
    results_path = os.path.join(args.output_dir, f'{args.split}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {results_path}")
    
    # Save predictions
    pred_path = os.path.join(args.output_dir, f'{args.split}_predictions.npz')
    np.savez(pred_path, predictions=preds, labels=labels)
    print(f"✅ Predictions saved: {pred_path}")
    
    print()
    print("=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
