"""
Models Package

This package contains the baseline ADL recognition model and its components.

Modules:
    - baseline_model: Main TCN-BiGRU-Attention model
    - components: Reusable components (TCN, Attention, FocalLoss)

Example usage:
    >>> from models import BaselineModel
    >>> model = BaselineModel(in_dim=114, hidden=128, classes=5)
    >>> logits = model(input_tensor)
"""

from .baseline_model import BaselineModel
from .components import TCNBlock, AdditiveAttention, FocalLoss

__all__ = [
    'BaselineModel',
    'TCNBlock',
    'AdditiveAttention',
    'FocalLoss'
]

__version__ = '1.0.0'
