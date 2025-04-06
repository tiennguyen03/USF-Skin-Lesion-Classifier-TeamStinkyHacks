"""
ML package for skin cancer classification
Contains:
- data/ - Data loading utilities
- model/ - Model architecture and training
"""

# Expose key components at package level
from .data.loader import get_dataloaders
from .model.model import SkinCancerModel

__all__ = ['get_dataloaders', 'SkinCancerModel']