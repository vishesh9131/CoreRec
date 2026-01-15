"""
CNN (Convolutional Neural Network) Module
==========================================

Top-level module for CNN algorithm access.

Usage:
------
    import corerec.cnn as cnn
    
    model = cnn.CNN(input_dim=100, num_classes=10, ...)
"""

from corerec.engines.content_based.nn_based_algorithms.cnn import CNN

__all__ = ["CNN"]

