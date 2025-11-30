"""
Autoencoder Module for Content-Based Recommendation
===================================================

Top-level module for Autoencoder algorithm access.

Usage:
------
    import corerec.autoencoder as autoencoder
    
    model = autoencoder.Autoencoder(input_dim=18, hidden_dim=12, latent_dim=6)
"""

from corerec.engines.contentFilterEngine.nn_based_algorithms.autoencoder import Autoencoder

__all__ = ["Autoencoder"]

