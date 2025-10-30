#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified training script for all ADL recognition models

Supports:
    - baseline: TCN + BiGRU + Attention (95.40% baseline)
    - transformer: Transformer encoder (83.66%)
    - deep_tcn: Deep TCN with SE attention
    - local_hybrid: TCN + Local Self-Attention + BiGRU
    - conformer: Conformer (Conv-augmented Transformer)

Usage:
    python run_train_all.py --model baseline
    python run_train_all.py --model deep_tcn --epochs 50
    python run_train_all.py --model conformer --hidden 256
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import all models
from models import BaselineModel, TransformerModel
from models.deep_tcn_model import DeepTCNModel
from models.local_attention_hybrid import LocalAttentionHybrid
from models.conformer_model import ConformerModel
from models.components import FocalLoss


class ADLDataset(Dataset):
    """Dataset for ADL recognition"""
    def __init__(self, X, y, lengths):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.lengths = torch.LongTensor(lengths)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


def get_model(model_name, hidden_dim, dropout, num_classes=5, input_dim=114):
    """Factory function to create model by name"""
    models = {
        'baseline': lambda: BaselineModel(
            in_dim=input_dim, hidden=hidden_dim, classes=num_classes, dropout=dropout
        ),
        'transformer': lambda: TransformerModel(
            in_dim=input_dim, hidden=hidden_dim, classes=num_classes, dropout=dropout
        ),
        'deep_tcn': lambda: DeepTCNModel(
            in_dim=input_dim, hidden=hidden_dim, classes=num_classes, dropout=dropout
        ),
        'local_hybrid': lambda: LocalAttentionHybrid(
            in_dim=input_dim, hidden=hidden_dim, classes=num_classes, dropout=dropout
        ),
        'conformer': lambda: ConformerModel(
            in_dim=input_dim, hidden=hidden_dim, classes=num_classes, dropout=dropout
        )
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name]()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X, y, lengths in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y = X.to(device), y.to(device)
            
            logits = model(X)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, f1, all_preds, all_labels


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(history['train_f1'], label='Train', linewidth=2)
    axes[2].plot(history['val_f1'], label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train ADL Recognition Models')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['baseline', 'transformer', 'deep_tcn', 'local_hybrid', 'conformer'],
                        help='Model architecture to train')
    parser.add_argument('--data_path', type=str, default='data/processed/dataset_with_lengths_v3.npz',
                        help='Path to dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (default: checkpoints_{model})')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1.5, help='Focal loss gamma')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f'checkpoints_{args.model}'
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print(f"\n📂 Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    X = data['X']
    y = data['y']
    class_names = data['class_names']
    
    print(f"   Dataset: X={X.shape}, y={y.shape}")
    print(f"   Classes: {class_names}")
    
    # Shuffle and split (64% train, 16% val, 20% test)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    
    n_train = int(len(X) * 0.64)
    n_val = int(len(X) * 0.16)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Dummy lengths (not used by most models)
    lengths_train = np.full(len(X_train), X_train.shape[1])
    lengths_val = np.full(len(X_val), X_val.shape[1])
    lengths_test = np.full(len(X_test), X_test.shape[1])
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ADLDataset(X_train, y_train, lengths_train)
    val_dataset = ADLDataset(X_val, y_val, lengths_val)
    test_dataset = ADLDataset(X_test, y_test, lengths_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\n🏗️  Building {args.model.upper()} model...")
    model = get_model(args.model, args.hidden_dim, args.dropout)
    model = model.to(device)
    
    num_params = model.count_parameters()
    print(f"   Parameters: {num_params:,}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Dropout: {args.dropout}")
    
    # Loss and optimizer
    criterion = FocalLoss(gamma=args.gamma, weight=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Focal loss gamma: {args.gamma}")
    print(f"   Early stopping patience: {args.patience}")
    print("="*80)
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_{args.model}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'args': vars(args)
            }, checkpoint_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
    
    print("="*80)
    print(f"✅ Training completed!")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model for testing
    print(f"\n📊 Evaluating on test set...")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'best_{args.model}.pt'), 
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion, device
    )
    
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")
    
    # Per-class metrics
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    class_names_list = ['t1_cook', 't2_handwash', 't3_sleep', 't4_medicine', 't5_eat']
    
    print("\n📈 Per-class F1 scores:")
    for i, (name, score) in enumerate(zip(class_names_list, per_class_f1)):
        print(f"   {name}: {score:.4f}")
    
    # Save results
    results = {
        'model': args.model,
        'hyperparameters': vars(args),
        'num_parameters': num_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'per_class_f1': {name: float(score) for name, score in zip(class_names_list, per_class_f1)}
    }
    
    results_path = os.path.join(args.checkpoint_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")
    
    # Plot training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history, history_path)
    print(f"📊 Training history saved to {history_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.checkpoint_dir, 'confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names_list, cm_path)
    print(f"📊 Confusion matrix saved to {cm_path}")
    
    print("\n" + "="*80)
    print(f"🎉 All done! Model: {args.model.upper()}")
    print(f"   Final Test Accuracy: {test_acc:.4f}")
    print(f"   Final Test F1 Score: {test_f1:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
