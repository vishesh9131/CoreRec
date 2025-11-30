"""
DKN (Deep Knowledge-aware Network) Module
==========================================

Top-level module for DKN algorithm access.

Usage:
------
    import corerec.dkn as dkn
    
    model = dkn.DKN(vocab_size=1000, embedding_dim=128, ...)
"""

from corerec.engines.contentFilterEngine.nn_based_algorithms.dkn import DKN

__all__ = ["DKN"]

