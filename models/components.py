#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Components: TCN Block and Additive Attention

This module contains reusable components for the baseline ADL recognition model:
- TCNBlock: Temporal Convolutional Network block with dilated convolutions
- AdditiveAttention: Bahdanau-style attention mechanism for temporal focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network (TCN) Block
    
    Features:
    - Dilated causal convolutions for large receptive fields
    - Residual connections for gradient flow
    - Dropout for regularization
    
    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels
        kernel (int): Kernel size (default: 3)
        dilation (int): Dilation factor (default: 1)
        dropout (float): Dropout rate (default: 0.2)
    
    Input shape: [Batch, Channels, Time]
    Output shape: [Batch, Channels, Time]
    """
    
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual connection (1x1 conv if channel dimensions don't match)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Channels, Time]
        
        Returns:
            Output tensor [Batch, Channels, Time] with residual connection
        """
        y = self.net(x)
        
        # Causal trimming: remove future time steps
        trim = y.shape[-1] - x.shape[-1]
        if trim > 0:
            y = y[..., :-trim]
        
        # Residual connection
        return y + self.downsample(x)


class AdditiveAttention(nn.Module):
    """
    Additive Attention (Bahdanau-style)
    
    Learns to focus on important timesteps in the sequence.
    Unlike multi-head attention, this uses a single attention focus,
    which is more suitable for short sequences and provides better
    interpretability.
    
    Args:
        hidden_dim (int): Hidden dimension of GRU output (before bidirectional concat)
    
    Input shape: [Batch, Time, 2*hidden_dim] (BiGRU output)
    Output shape: 
        - context: [Batch, 2*hidden_dim] (weighted sum)
        - weights: [Batch, Time] (attention weights for visualization)
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Score function: learns importance of each timestep
        # Input: 2*hidden_dim (BiGRU concatenates forward and backward)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, gru_out):
        """
        Compute attention-weighted context vector
        
        Args:
            gru_out: BiGRU output [Batch, Time, 2*hidden_dim]
        
        Returns:
            context: Weighted sum of gru_out [Batch, 2*hidden_dim]
            attn_weights: Attention weights [Batch, Time] (for visualization)
        """
        # Compute attention scores for each timestep
        scores = self.attention(gru_out)  # [Batch, Time, 1]
        scores = scores.squeeze(-1)  # [Batch, Time]
        
        # Softmax to get attention weights (sum to 1 across time dimension)
        attn_weights = torch.softmax(scores, dim=1)  # [Batch, Time]
        
        # Weighted sum: context vector
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # [Batch, 2*hidden_dim]
        
        return context, attn_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Focal Loss down-weights easy examples and focuses on hard examples.
    Useful for imbalanced datasets where some classes have fewer samples.
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        gamma (float): Focusing parameter (default: 2.0)
        weight (Tensor, optional): Class weights for further balancing
    """
    
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, logits, target):
        """
        Args:
            logits: Model predictions [Batch, Classes]
            target: Ground truth labels [Batch]
        
        Returns:
            Focal loss value (scalar)
        """
        # Standard cross-entropy loss (per sample)
        ce_loss = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        
        # Probability of true class
        pt = torch.exp(-ce_loss)
        
        # Focal loss: down-weight easy examples
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


# Test function
if __name__ == '__main__':
    print("Testing model components...")
    
    # Test TCNBlock
    print("\n1. Testing TCNBlock:")
    tcn = TCNBlock(in_ch=128, out_ch=128, kernel=3, dilation=2, dropout=0.1)
    x_tcn = torch.randn(4, 128, 100)  # [batch=4, channels=128, time=100]
    out_tcn = tcn(x_tcn)
    print(f"   Input shape: {x_tcn.shape}")
    print(f"   Output shape: {out_tcn.shape}")
    print(f"   Parameters: {sum(p.numel() for p in tcn.parameters()):,}")
    
    # Test AdditiveAttention
    print("\n2. Testing AdditiveAttention:")
    attn = AdditiveAttention(hidden_dim=128)
    x_attn = torch.randn(4, 100, 256)  # [batch=4, time=100, features=256] (BiGRU output)
    context, weights = attn(x_attn)
    print(f"   Input shape: {x_attn.shape}")
    print(f"   Context shape: {context.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights sum: {weights.sum(dim=1)}")  # Should be ~1.0
    print(f"   Parameters: {sum(p.numel() for p in attn.parameters()):,}")
    
    # Test FocalLoss
    print("\n3. Testing FocalLoss:")
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 5)  # [batch=4, classes=5]
    targets = torch.tensor([0, 1, 2, 3])
    loss = focal(logits, targets)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Loss value: {loss.item():.4f}")
    
    print("\n✅ All components tested successfully!")
