"""
Visualize Model Predictions and Attention Weights

Generate visualizations for model interpretability and analysis.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BaselineModel
from train.utils import load_data
from train.config import TrainingConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize Model Predictions')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='../data/processed/dataset.npz',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='../results/visualizations',
                       help='Output directory')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    return parser.parse_args()


def plot_attention_weights(attention_weights, class_name, save_path):
    """
    Plot attention weights over time
    
    Args:
        attention_weights: Attention weights (T,)
        class_name: True class name
        save_path: Path to save plot
    """
    T = len(attention_weights)
    
    plt.figure(figsize=(12, 4))
    plt.plot(attention_weights, linewidth=2, color='#2E86C1')
    plt.fill_between(range(T), attention_weights, alpha=0.3, color='#AED6F1')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title(f'Attention Weights over Time (True Class: {class_name})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_confidence(logits, class_names, true_label, save_path):
    """
    Plot prediction confidence (softmax probabilities)
    
    Args:
        logits: Model logits (n_classes,)
        class_names: Class names
        true_label: True label index
        save_path: Path to save plot
    """
    # Softmax
    probs = torch.softmax(torch.FloatTensor(logits), dim=0).numpy()
    
    # Create bar plot
    colors = ['#28B463' if i == true_label else '#E74C3C' 
              if i == np.argmax(probs) else '#BDC3C7' 
              for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, probs, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Prediction Confidence', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#28B463', edgecolor='black', label='True Class'),
        Patch(facecolor='#E74C3C', edgecolor='black', label='Predicted Class'),
        Patch(facecolor='#BDC3C7', edgecolor='black', label='Other Classes')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sensor_activations(X, attention_weights, sensor_names=None, save_path=None):
    """
    Plot sensor activations with attention overlay
    
    Args:
        X: Input sequence (T, D)
        attention_weights: Attention weights (T,)
        sensor_names: List of sensor names (optional)
        save_path: Path to save plot
    """
    T, D = X.shape
    
    # Select subset of sensors to visualize (too many to show all)
    n_sensors_to_show = min(10, D)
    sensor_indices = np.linspace(0, D-1, n_sensors_to_show, dtype=int)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot sensor activations
    for i, sensor_idx in enumerate(sensor_indices):
        sensor_data = X[:, sensor_idx]
        label = sensor_names[sensor_idx] if sensor_names else f'Sensor {sensor_idx}'
        ax1.plot(sensor_data, label=label, alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Activation', fontsize=12)
    ax1.set_title('Sensor Activations over Time', fontsize=14)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot attention weights
    ax2.plot(attention_weights, linewidth=2, color='#E74C3C')
    ax2.fill_between(range(T), attention_weights, alpha=0.3, color='#F1948A')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('Attention', fontsize=12)
    ax2.set_title('Attention Weights', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main visualization function"""
    args = parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("🎨 Visualizing Model Predictions")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    train_data, val_data, test_data, class_names = load_data(args.data_path)
    X_test, y_test = test_data
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {class_names}")
    print()
    
    # Load model
    print("🔧 Loading model...")
    model = BaselineModel(
        input_dim=114,
        hidden_dim=TrainingConfig.HIDDEN_DIM,
        n_classes=len(class_names),
        n_tcn_blocks=TrainingConfig.N_TCN_BLOCKS,
        kernel_size=3,
        dilations=TrainingConfig.DILATIONS,
        dropout=TrainingConfig.DROPOUT
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")
    print()
    
    # Randomly sample examples from each class
    print("🎲 Sampling examples...")
    samples_per_class = max(1, args.n_samples // len(class_names))
    
    for class_idx, class_name in enumerate(class_names):
        # Find samples of this class
        class_mask = (y_test == class_idx)
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            print(f"⚠️  No samples found for class {class_name}")
            continue
        
        # Sample randomly
        sample_indices = np.random.choice(
            class_indices, 
            size=min(samples_per_class, len(class_indices)),
            replace=False
        )
        
        for sample_idx in sample_indices:
            X_sample = X_test[sample_idx]
            y_sample = y_test[sample_idx]
            
            # Forward pass
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).to(device)
                logits, attention_weights = model(X_tensor, return_attention=True)
                
                logits = logits.cpu().numpy()[0]
                attention_weights = attention_weights.cpu().numpy()[0]
            
            # Prediction
            pred_idx = np.argmax(logits)
            pred_class = class_names[pred_idx]
            true_class = class_names[y_sample]
            
            print(f"Sample {sample_idx}: True={true_class}, Pred={pred_class}")
            
            # Plot attention weights
            attn_path = os.path.join(
                args.output_dir, 
                f'attention_{class_name}_sample{sample_idx}.png'
            )
            plot_attention_weights(attention_weights, true_class, attn_path)
            
            # Plot prediction confidence
            conf_path = os.path.join(
                args.output_dir,
                f'confidence_{class_name}_sample{sample_idx}.png'
            )
            plot_prediction_confidence(logits, class_names, y_sample, conf_path)
            
            # Plot sensor activations
            sens_path = os.path.join(
                args.output_dir,
                f'sensors_{class_name}_sample{sample_idx}.png'
            )
            plot_sensor_activations(X_sample, attention_weights, save_path=sens_path)
    
    print()
    print("=" * 80)
    print(f"✅ Visualizations saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
