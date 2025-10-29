#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline ADL Recognition Model: TCN + BiGRU + Additive Attention

This is the main baseline model that achieves 93.7% accuracy on ADL classification.

Architecture:
    Input (T=100, F=114)
         ↓
    [Feature Projection] Linear(114 → 128)
         ↓
    [Temporal Encoding] TCN (3 blocks, dilation=1,2,4)
         ↓
    [Sequential Modeling] BiGRU (hidden=128, bidirectional)
         ↓
    [Temporal Attention] Additive Attention
         ↓
    [Classification] FC(256→128) → ReLU → Dropout → FC(128→5)
         ↓
    Output Logits (5 classes)

Performance:
    - Accuracy: 93.7%
    - Macro F1: 0.924
    - Parameters: 0.58M
    - Inference Time: 2.66ms (GPU)
"""

import torch
import torch.nn as nn
from .components import TCNBlock, AdditiveAttention


class BaselineModel(nn.Module):
    """
    TCN + BiGRU + Additive Attention for ADL Recognition
    
    Args:
        in_dim (int): Input feature dimension (default: 114)
            - 27 binary sensors × 4 channels (current, duration, count, cumulative)
            - + 6 temporal features (hour, day_of_week, etc.)
        hidden (int): Hidden dimension for TCN and GRU (default: 128)
        classes (int): Number of activity classes (default: 5)
            - t1: cooking
            - t2: hand washing
            - t3: sleeping
            - t4: taking medicine
            - t5: eating
    
    Input shape: [Batch, Time=100, Features=114]
    Output shape: [Batch, Classes=5]
    
    Example:
        >>> model = BaselineModel(in_dim=114, hidden=128, classes=5)
        >>> x = torch.randn(32, 100, 114)  # batch_size=32
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([32, 5])
    """
    
    def __init__(self, in_dim=114, hidden=128, classes=5):
        super().__init__()
        
        # Feature projection: map input features to hidden dimension
        self.proj = nn.Linear(in_dim, hidden)
        
        # Temporal Convolutional Network (TCN)
        # 3 blocks with exponentially increasing dilation
        # Receptive field: 13 timesteps
        self.tcn = nn.Sequential(
            TCNBlock(hidden, hidden, kernel=3, dilation=1, dropout=0.1),
            TCNBlock(hidden, hidden, kernel=3, dilation=2, dropout=0.1),
            TCNBlock(hidden, hidden, kernel=3, dilation=4, dropout=0.1),
        )
        
        # Bidirectional GRU
        # Input: [Batch, Time, 128]
        # Output: [Batch, Time, 256] (forward + backward)
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Additive Attention
        # Learns importance weights for each timestep
        self.attention = AdditiveAttention(hidden)
        
        # Classification head
        # Input: context vector [Batch, 256]
        # Output: logits [Batch, 5]
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [Batch, Time, Features]
            return_attention: If True, also return attention weights for visualization
        
        Returns:
            logits: Class logits [Batch, Classes]
            attn_weights (optional): Attention weights [Batch, Time]
        """
        # Feature projection: [B, T, F] → [B, T, H]
        h = self.proj(x)
        
        # TCN encoding: [B, T, H] → [B, H, T] → [B, H, T] → [B, T, H]
        h_tcn = self.tcn(h.transpose(1, 2))  # TCN expects [B, C, T]
        h = h_tcn.transpose(1, 2)  # Back to [B, T, H]
        
        # BiGRU encoding: [B, T, H] → [B, T, 2H]
        gru_out, _ = self.gru(h)
        
        # Attention: [B, T, 2H] → [B, 2H], [B, T]
        context, attn_weights = self.attention(gru_out)
        
        # Classification: [B, 2H] → [B, C]
        logits = self.head(context)
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test and example usage
if __name__ == '__main__':
    print("="*80)
    print("Baseline ADL Recognition Model")
    print("="*80)
    
    # Create model
    model = BaselineModel(in_dim=114, hidden=128, classes=5)
    
    # Model summary
    print(f"\n📊 Model Configuration:")
    print(f"   Input dimension: 114")
    print(f"   Hidden dimension: 128")
    print(f"   Output classes: 5")
    print(f"   Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    print(f"\n🔄 Testing forward pass:")
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 114)
    print(f"   Input shape: {x.shape}")
    
    # Forward without attention
    logits = model(x)
    print(f"   Output logits shape: {logits.shape}")
    
    # Forward with attention
    logits, attn = model(x, return_attention=True)
    print(f"   Attention weights shape: {attn.shape}")
    print(f"   Attention sum (should be ~1.0): {attn[0].sum():.4f}")
    
    # Component-wise parameter count
    print(f"\n🔍 Parameter breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name:12s}: {params:>8,} params")
    
    # Model size estimation
    param_size_mb = model.count_parameters() * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"\n💾 Model size: {param_size_mb:.2f} MB (float32)")
    
    # Inference example
    print(f"\n🎯 Inference example:")
    model.eval()
    with torch.no_grad():
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
    
    class_names = ['cooking', 'hand_washing', 'sleeping', 'medicine', 'eating']
    print(f"   Sample predictions:")
    for i in range(min(3, batch_size)):
        pred_class = predictions[i].item()
        pred_prob = probabilities[i, pred_class].item()
        print(f"      Sample {i+1}: {class_names[pred_class]} (confidence: {pred_prob:.2%})")
    
    print(f"\n✅ Model test completed successfully!")
    print("="*80)
