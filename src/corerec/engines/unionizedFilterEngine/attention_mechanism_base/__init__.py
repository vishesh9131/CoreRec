"""
Attention Mechanism Base Module

This module provides foundational classes and utilities for implementing attention-based 
unionized filtering mechanisms in recommendation systems. It includes implementations 
of various attention architectures like self-attention, multi-head attention, and 
transformer-based approaches.

Available Classes:
    - AttentionBasedUFBase: Base class for attention-based filtering
    - TransformerBasedUFBase: Base class for transformer architectures
    - SASRecBase: Base implementation of Self-Attentive Sequential Recommendation
"""

from .Transformer_based_uf_base import TransformerBasedUFBase as AM_TRANSFORMER
from .Attention_based_uf_base import AttentionBasedUFBase as AM_ATTENTION

from .a2svd import A2SVD as AM_A2SD
from .sasrec import SASRec as AM_SAS

__all__ = [
    "AM_TRANSFORMER",
    "AM_ATTENTION",
    "AM_A2SD",
    "AM_SAS",
]
