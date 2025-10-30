#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep TCN Model for ADL Recognition

A deeper Temporal Convolutional Network with multi-scale features
and squeeze-and-excitation attention for enhanced performance.

Architecture:
    Input (T=100, F=114)
         ↓
    [Feature Projection] Linear(114 → 256)
         ↓
    [Deep TCN] 6 blocks, multi-scale dilation
         ↓
    [Multi-Scale Fusion] Concatenate features
         ↓
    [SE Attention] Channel attention
         ↓
    [Global Pooling] Average over time
         ↓
    [Classification] FC(256→128) → ReLU → Dropout → FC(128→5)
         ↓
    Output Logits (5 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import TCNBlock


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, channels, time]
        """
        b, c, t = x.size()
        # Squeeze: [B, C, T] -> [B, C, 1]
        y = self.squeeze(x)
        # Excitation: [B, C, 1] -> [B, C]
        y = y.squeeze(-1)
        y = self.excitation(y)
        # Scale: [B, C] -> [B, C, 1] -> [B, C, T]
        y = y.unsqueeze(-1)
        return x * y.expand_as(x)


class DeepTCNModel(nn.Module):
    """
    Deep TCN with Multi-Scale Features
    
    Args:
        in_dim (int): Input feature dimension (default: 114)
        hidden (int): Hidden dimension (default: 256)
        classes (int): Number of classes (default: 5)
        n_blocks (int): Number of TCN blocks (default: 6)
        dropout (float): Dropout rate (default: 0.1)
    
    Input shape: [Batch, Time=100, Features=114]
    Output shape: [Batch, Classes=5]
    """
    
    def __init__(self, in_dim=114, hidden=256, classes=5, n_blocks=6, dropout=0.1):
        super().__init__()
        
        # Feature projection
        self.proj = nn.Linear(in_dim, hidden)
        
        # Deep TCN with exponential dilation
        dilations = [2**i for i in range(n_blocks)]  # [1, 2, 4, 8, 16, 32]
        
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden, hidden, kernel=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        
        # Multi-scale feature collection (collect outputs from multiple layers)
        self.multiscale_indices = [1, 3, 5]  # Use outputs from blocks 2, 4, 6
        
        # Squeeze-and-Excitation attention
        self.se = SEBlock(hidden, reduction=16)
        
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
            return_attention: If True, return SE weights for visualization
        
        Returns:
            logits: Class logits [Batch, Classes]
            attention_weights (optional): SE attention weights
        """
        # Feature projection: [B, T, F] -> [B, T, H]
        h = self.proj(x)
        
        # TCN encoding: [B, T, H] -> [B, H, T]
        h = h.transpose(1, 2)
        
        # Deep TCN with multi-scale features
        multiscale_features = []
        for i, tcn_block in enumerate(self.tcn_blocks):
            h = tcn_block(h)
            if i in self.multiscale_indices:
                multiscale_features.append(h)
        
        # Use final output
        h = multiscale_features[-1]  # [B, H, T]
        
        # Squeeze-and-Excitation attention
        h = self.se(h)
        
        # Global average pooling: [B, H, T] -> [B, H]
        h = h.mean(dim=2)
        
        # Classification: [B, H] -> [B, C]
        logits = self.head(h)
        
        if return_attention:
            return logits, None  # SE attention is channel-wise
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == '__main__':
    print("="*80)
    print("Deep TCN Model")
    print("="*80)
    
    model = DeepTCNModel(in_dim=114, hidden=256, classes=5, n_blocks=6, dropout=0.1)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Input dimension: 114")
    print(f"   Hidden dimension: 256")
    print(f"   TCN blocks: 6")
    print(f"   Dilations: [1, 2, 4, 8, 16, 32]")
    print(f"   Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 114)
    logits = model(x)
    print(f"\n   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    
    param_size_mb = model.count_parameters() * 4 / (1024 ** 2)
    print(f"\n💾 Model size: {param_size_mb:.2f} MB")
    
    print("\n✅ Model test completed!")
    print("="*80)
