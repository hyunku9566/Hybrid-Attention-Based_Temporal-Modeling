"""
Models Package

This package contains the baseline ADL recognition model and its components.

Modules:
    - baseline_model: Main TCN-BiGRU-Attention model
    - transformer_model: Transformer encoder model
    - components: Reusable components (TCN, Attention, FocalLoss)

Example usage:
    >>> from models import BaselineModel, TransformerModel
    >>> model = BaselineModel(in_dim=114, hidden=128, classes=5)
    >>> logits = model(input_tensor)
"""

from .baseline_model import BaselineModel
from .transformer_model import TransformerModel
from .components import TCNBlock, AdditiveAttention, FocalLoss

__all__ = [
    'BaselineModel',
    'TransformerModel',
    'TCNBlock',
    'AdditiveAttention',
    'FocalLoss'
]

__version__ = '1.0.0'
