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

from .SASRec_base import SASRecBase as AM_SASREC
from .Transformer_based_uf_base import TransformerBasedUFBase as AM_TRANSFORMER
from .Attention_based_uf_base import AttentionBasedUFBase as AM_ATTENTION
