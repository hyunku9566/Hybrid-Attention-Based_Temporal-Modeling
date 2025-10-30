#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based ADL Recognition Model

A Transformer encoder model for Activities of Daily Living recognition,
as an alternative to the TCN-BiGRU-Attention baseline.

Architecture:
    Input (T=100, F=114)
         ↓
    [Feature Projection] Linear(114 → 256)
         ↓
    [Positional Encoding] Sinusoidal/Learned
         ↓
    [Transformer Encoder] 4 layers, 8 heads
         ↓
    [Global Pooling] Mean over time dimension
         ↓
    [Classification] FC(256→128) → ReLU → Dropout → FC(128→5)
         ↓
    Output Logits (5 classes)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer Encoder for ADL Recognition
    
    Args:
        in_dim (int): Input feature dimension (default: 114)
        hidden (int): Hidden dimension for Transformer (default: 256)
        classes (int): Number of activity classes (default: 5)
        n_layers (int): Number of Transformer encoder layers (default: 4)
        n_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate (default: 0.1)
        max_len (int): Maximum sequence length (default: 100)
    
    Input shape: [Batch, Time=100, Features=114]
    Output shape: [Batch, Classes=5]
    """
    
    def __init__(self, in_dim=114, hidden=256, classes=5, n_layers=4, n_heads=8, dropout=0.1, max_len=100):
        super().__init__()
        
        assert hidden % n_heads == 0, f"hidden ({hidden}) must be divisible by n_heads ({n_heads})"
        
        self.hidden = hidden
        self.n_heads = n_heads
        
        # Feature projection: map input features to hidden dimension
        self.proj = nn.Linear(in_dim, hidden)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden, max_len=max_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,  # Standard is 4x
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [Batch, Time, Features]
            return_attention: If True, return attention weights (not implemented for compatibility)
        
        Returns:
            logits: Class logits [Batch, Classes]
            attention_weights (optional): None (for API compatibility)
        """
        # Feature projection: [B, T, F] → [B, T, H]
        h = self.proj(x)
        
        # Add positional encoding: [B, T, H]
        h = self.pos_encoder(h)
        
        # Transformer encoding: [B, T, H] → [B, T, H]
        h = self.transformer_encoder(h)
        
        # Global average pooling over time: [B, T, H] → [B, H]
        h = h.mean(dim=1)
        
        # Classification: [B, H] → [B, C]
        logits = self.head(h)
        
        if return_attention:
            # Return None for attention weights (Transformer has multi-head attention)
            return logits, None
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test and example usage
if __name__ == '__main__':
    print("="*80)
    print("Transformer ADL Recognition Model")
    print("="*80)
    
    # Create model
    model = TransformerModel(in_dim=114, hidden=256, classes=5, n_layers=4, n_heads=8)
    
    # Model summary
    print(f"\n📊 Model Configuration:")
    print(f"   Input dimension: 114")
    print(f"   Hidden dimension: 256")
    print(f"   Transformer layers: 4")
    print(f"   Attention heads: 8")
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
    
    # Forward with attention (returns None for compatibility)
    logits, attn = model(x, return_attention=True)
    print(f"   Attention weights: {attn} (Transformer uses multi-head attention)")
    
    # Component-wise parameter count
    print(f"\n🔍 Parameter breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name:25s}: {params:>10,} params")
    
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
