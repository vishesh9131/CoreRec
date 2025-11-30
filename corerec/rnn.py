"""
RNN Module
==========

Top-level module for RNN algorithm access.

Usage:
------
    import corerec.rnn as rnn
    
    model = rnn.RNNModel(input_dim=500, embed_dim=256, ...)
"""

from corerec.engines.contentFilterEngine.nn_based_algorithms.rnn import RNNModel

__all__ = ["RNNModel"]

