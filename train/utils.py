"""
Training Utilities

Helper functions for data loading, metrics, checkpointing, and visualization.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ADLDataset(Dataset):
    """PyTorch Dataset for ADL recognition"""
    
    def __init__(self, X, y):
        """
        Args:
            X: numpy array of shape (N, T, D)
            y: numpy array of shape (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(data_path, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, seed=42):
    """
    Load and split dataset
    
    Args:
        data_path: Path to dataset.npz file
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Load data
    data = np.load(data_path, allow_pickle=True)
    X = data['X']  # (N, T, D)
    y = data['y']  # (N,)
    class_names = data['class_names']
    
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print(f"Classes: {class_names}")
    
    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


def create_dataloaders(train_data, val_data, test_data, 
                       batch_size=64, use_weighted_sampler=True,
                       num_workers=4, pin_memory=True):
    """
    Create PyTorch DataLoaders
    
    Args:
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        test_data: (X_test, y_test)
        batch_size: Batch size
        use_weighted_sampler: Use weighted random sampler to balance classes
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # Create datasets
    train_dataset = ADLDataset(X_train, y_train)
    val_dataset = ADLDataset(X_val, y_val)
    test_dataset = ADLDataset(X_test, y_test)
    
    # Weighted sampler for training (balance classes)
    if use_weighted_sampler:
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train, device='cpu'):
    """
    Compute class weights for Focal Loss
    
    Args:
        y_train: Training labels
        device: Device to put weights on
        
    Returns:
        class_weights: Tensor of shape (n_classes,)
    """
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    return torch.FloatTensor(class_weights).to(device)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved: {save_path}")


def load_checkpoint(model, checkpoint_path, device='cpu', load_optimizer=False, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f}")
    
    return checkpoint['epoch'], checkpoint['val_acc']


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(results, save_path):
    """Save evaluation results to JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved: {save_path}")


if __name__ == '__main__':
    # Test utilities
    print("Testing ADLDataset...")
    X = np.random.randn(100, 100, 114)
    y = np.random.randint(0, 5, 100)
    dataset = ADLDataset(X, y)
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample: X={dataset[0][0].shape}, y={dataset[0][1]}")
    
    print("\nTesting class weights...")
    y_train = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 4, 4])
    weights = compute_class_weights(y_train)
    print(f"Class weights: {weights}")
