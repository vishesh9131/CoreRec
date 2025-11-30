"""
AITM (Adaptive Information Transfer Multi-task) Module
======================================================

Top-level module for AITM algorithm access.

Usage:
------
    import corerec.aitm as aitm
    
    model = aitm.AITM(input_dim=18, feature_dim=50, output_dim=18)
    model.train(dataloader, criterion, optimizer, num_epochs)
"""

from corerec.engines.contentFilterEngine.nn_based_algorithms.AITM import AITM

__all__ = ["AITM"]

