#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Attention Hybrid Model for ADL Recognition

Combines TCN, Local Self-Attention, and BiGRU for enhanced performance
on activity recognition tasks with better long-range modeling.

Architecture:
    Input (T=100, F=114)
         ↓
    [Feature Projection] Linear(114 → 256)
         ↓
    [Deep TCN] 5 blocks, multi-scale
         ↓
    [Local Self-Attention] Window size=25
         ↓
    [BiGRU] Bidirectional, hidden=256
         ↓
    [Global Attention] Additive attention
         ↓
    [Classification] FC(512→256) → ReLU → Dropout → FC(256→5)
         ↓
    Output Logits (5 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .components import TCNBlock, AdditiveAttention


class LocalSelfAttention(nn.Module):
    """
    Local Self-Attention with sliding window
    """
    def __init__(self, hidden_dim, n_heads=2, window_size=25, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, heads, T, T]
        
        # Apply local attention mask (sliding window)
        mask = torch.ones(T, T, device=x.device)
        for i in range(T):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0
        
        mask = mask.bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        
        # Output projection
        out = self.out_proj(out)
        return out


class LocalAttentionHybrid(nn.Module):
    """
    Hybrid Model: TCN + Local Self-Attention + BiGRU + Attention
    
    Args:
        in_dim (int): Input feature dimension (default: 114)
        hidden (int): Hidden dimension (default: 256)
        classes (int): Number of classes (default: 5)
        n_tcn_blocks (int): Number of TCN blocks (default: 5)
        n_heads (int): Number of attention heads (default: 2)
        window_size (int): Local attention window size (default: 25)
        dropout (float): Dropout rate (default: 0.1)
    
    Input shape: [Batch, Time=100, Features=114]
    Output shape: [Batch, Classes=5]
    """
    
    def __init__(self, in_dim=114, hidden=256, classes=5, n_tcn_blocks=5, 
                 n_heads=2, window_size=25, dropout=0.1):
        super().__init__()
        
        # Feature projection
        self.proj = nn.Linear(in_dim, hidden)
        
        # Deep TCN
        dilations = [2**i for i in range(n_tcn_blocks)]  # [1, 2, 4, 8, 16]
        self.tcn = nn.Sequential(*[
            TCNBlock(hidden, hidden, kernel=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        
        # Local Self-Attention
        self.local_attention = LocalSelfAttention(
            hidden, n_heads=n_heads, window_size=window_size, dropout=dropout
        )
        
        # Layer normalization after attention
        self.norm = nn.LayerNorm(hidden)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Global Attention
        self.attention = AdditiveAttention(hidden)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor [Batch, Time, Features]
            return_attention: If True, return attention weights
        
        Returns:
            logits: Class logits [Batch, Classes]
            attention_weights (optional): Attention weights [Batch, Time]
        """
        # Feature projection: [B, T, F] -> [B, T, H]
        h = self.proj(x)
        
        # TCN encoding: [B, T, H] -> [B, H, T] -> [B, H, T] -> [B, T, H]
        h_tcn = self.tcn(h.transpose(1, 2))
        h = h_tcn.transpose(1, 2)
        
        # Local Self-Attention: [B, T, H] -> [B, T, H]
        h_attn = self.local_attention(h)
        h = self.norm(h + h_attn)  # Residual connection + Layer norm
        
        # BiGRU encoding: [B, T, H] -> [B, T, 2H]
        gru_out, _ = self.gru(h)
        
        # Global Attention: [B, T, 2H] -> [B, 2H], [B, T]
        context, attn_weights = self.attention(gru_out)
        
        # Classification: [B, 2H] -> [B, C]
        logits = self.head(context)
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test
if __name__ == '__main__':
    print("="*80)
    print("Local Attention Hybrid Model")
    print("="*80)
    
    model = LocalAttentionHybrid(in_dim=114, hidden=256, classes=5, dropout=0.1)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Input dimension: 114")
    print(f"   Hidden dimension: 256")
    print(f"   TCN blocks: 5")
    print(f"   Local attention window: 25")
    print(f"   Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 114)
    logits, attn = model(x, return_attention=True)
    print(f"\n   Input: {x.shape}")
    print(f"   Output: {logits.shape}")
    print(f"   Attention: {attn.shape}")
    
    param_size_mb = model.count_parameters() * 4 / (1024 ** 2)
    print(f"\n💾 Model size: {param_size_mb:.2f} MB")
    
    print("\n✅ Model test completed!")
    print("="*80)
