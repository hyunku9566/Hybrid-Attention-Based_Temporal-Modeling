"""
Train Transformer ADL Recognition Model

Training script for the Transformer-based model.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TransformerModel, FocalLoss
from config import TrainingConfig
from utils import (
    load_data, create_dataloaders, compute_class_weights,
    train_epoch, evaluate, save_checkpoint, 
    plot_training_history, plot_confusion_matrix, save_results
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Transformer ADL Recognition Model')
    
    # Data
    parser.add_argument('--data_path', type=str, default='../data/processed/dataset.npz',
                       help='Path to preprocessed dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_transformer',
                       help='Directory to save checkpoints')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--focal_gamma', type=float, default=1.5,
                       help='Focal loss gamma')
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True,
                       help='Use weighted random sampler')
    parser.add_argument('--no_weighted_sampler', action='store_false', dest='use_weighted_sampler',
                       help='Do not use weighted sampler')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("🚀 Training Transformer ADL Recognition Model")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print()
    
    # Load data
    print("📊 Loading data...")
    train_data, val_data, test_data, class_names = load_data(
        args.data_path,
        seed=args.seed
    )
    print(f"Classes: {class_names}")
    print()
    
    # Create dataloaders
    print("🔄 Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        use_weighted_sampler=args.use_weighted_sampler,
        num_workers=args.num_workers
    )
    print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    print()
    
    # Create model
    print("🏗️  Building Transformer model...")
    model = TransformerModel(
        in_dim=114,
        hidden=args.hidden_dim,
        classes=len(class_names),
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)
    
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"Architecture: {args.n_layers} layers, {args.n_heads} heads")
    print()
    
    # Loss function (Focal Loss with class weights)
    print("📐 Setting up loss function...")
    class_weights = compute_class_weights(train_data[1], device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    criterion = FocalLoss(
        gamma=args.focal_gamma,
        weight=class_weights
    )
    print(f"Loss: Focal Loss (gamma={args.focal_gamma})")
    print()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay=1e-4)")
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6
    )
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0=10)")
    print()
    
    # Training loop
    print("🏋️  Training...")
    print("-" * 80)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, 'best_transformer.pt')
            save_checkpoint(model, optimizer, epoch, val_acc, save_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print()
    print("=" * 80)
    print(f"✅ Training complete! Best Val Acc: {best_val_acc:.4f}")
    print("=" * 80)
    print()
    
    # Load best model for evaluation
    print("📊 Evaluating best model on test set...")
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_transformer.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print()
    
    # Classification report
    from sklearn.metrics import classification_report, f1_score
    
    print("📈 Classification Report:")
    print("-" * 80)
    report = classification_report(
        test_labels, test_preds,
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Macro F1
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'macro_f1': float(macro_f1),
        'best_val_acc': float(best_val_acc),
        'n_params': n_params,
        'config': {
            'hidden_dim': args.hidden_dim,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'focal_gamma': args.focal_gamma
        }
    }
    
    results_path = os.path.join(args.checkpoint_dir, 'test_results.json')
    save_results(results, results_path)
    
    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.checkpoint_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    print("=" * 80)
    print("🎉 All done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
