"""
Transformer Module
=================

Top-level module for Transformer algorithm access.

Usage:
------
    import corerec.transformer as transformer
    
    model = transformer.TransformerModel(input_dim=500, embed_dim=256, ...)
"""

from corerec.engines.contentFilterEngine.nn_based_algorithms.transformer import TransformerModel

__all__ = ["TransformerModel"]

