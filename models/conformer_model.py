#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conformer Model for ADL Recognition

Conformer combines the strengths of CNNs and Transformers,
originally designed for speech recognition but adapted for ADL recognition.

Architecture inspired by: "Conformer: Convolution-augmented Transformer for Speech Recognition"

Architecture:
    Input (T=100, F=114)
         ↓
    [Feature Projection] Linear(114 → 256)
         ↓
    [Conformer Block] × 4
         ├─ Feed Forward Module (1/2)
         ├─ Multi-Head Self-Attention Module
         ├─ Convolution Module
         └─ Feed Forward Module (1/2)
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
import math


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class DepthwiseConv1d(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, groups=channels
        )
    
    def forward(self, x):
        return self.depthwise(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    """Feed Forward Module in Conformer"""
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-Head Self-Attention Module in Conformer"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer_norm(x)
        attn_out, _ = self.attention(x, x, x)
        return self.dropout(attn_out)


class ConvolutionModule(nn.Module):
    """Convolution Module in Conformer"""
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise_conv = DepthwiseConv1d(
            d_model, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        return x


class ConformerBlock(nn.Module):
    """
    Conformer Block = FFN(1/2) + MHSA + Conv + FFN(1/2)
    """
    def __init__(self, d_model, n_heads, conv_kernel_size=31, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.ffn1 = FeedForwardModule(d_model, expansion_factor, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, n_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, expansion_factor, dropout)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        # FFN 1 (1/2 scaling)
        x = x + 0.5 * self.ffn1(x)
        
        # MHSA
        x = x + self.mhsa(x)
        
        # Conv
        x = x + self.conv(x)
        
        # FFN 2 (1/2 scaling)
        x = x + 0.5 * self.ffn2(x)
        
        x = self.norm(x)
        return x


class ConformerModel(nn.Module):
    """
    Conformer Model for ADL Recognition
    
    Args:
        in_dim (int): Input feature dimension (default: 114)
        hidden (int): Hidden dimension (default: 256)
        classes (int): Number of classes (default: 5)
        n_layers (int): Number of Conformer blocks (default: 4)
        n_heads (int): Number of attention heads (default: 4)
        conv_kernel_size (int): Convolution kernel size (default: 31)
        dropout (float): Dropout rate (default: 0.1)
    
    Input shape: [Batch, Time=100, Features=114]
    Output shape: [Batch, Classes=5]
    """
    
    def __init__(self, in_dim=114, hidden=256, classes=5, n_layers=4, 
                 n_heads=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        
        # Feature projection
        self.proj = nn.Linear(in_dim, hidden)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden, dropout=dropout)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden, n_heads, conv_kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        
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
            return_attention: If True, return None for API compatibility
        
        Returns:
            logits: Class logits [Batch, Classes]
            attention_weights (optional): None
        """
        # Feature projection: [B, T, F] -> [B, T, H]
        h = self.proj(x)
        
        # Add positional encoding
        h = self.pos_encoder(h)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            h = block(h)
        
        # Global average pooling: [B, T, H] -> [B, H]
        h = h.mean(dim=1)
        
        # Classification: [B, H] -> [B, C]
        logits = self.head(h)
        
        if return_attention:
            return logits, None
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == '__main__':
    print("="*80)
    print("Conformer Model")
    print("="*80)
    
    model = ConformerModel(in_dim=114, hidden=256, classes=5, n_layers=4, dropout=0.1)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Input dimension: 114")
    print(f"   Hidden dimension: 256")
    print(f"   Conformer blocks: 4")
    print(f"   Attention heads: 4")
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
