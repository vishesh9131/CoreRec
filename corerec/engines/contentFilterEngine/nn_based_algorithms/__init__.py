"""
Neural Network Based Algorithms
===============================

Deep learning algorithms for content-based recommendations.

This module provides:
- DSSM (Deep Structured Semantic Model)
- MIND (Multi-Interest Network with Dynamic Routing)
- TDM (Tree-based Deep Model)
- YouTube DNN
- Transformers, CNN, RNN, Autoencoders, VAE

Usage:
------
    from corerec.engines.content import nn
    
    # Neural network models
    model = nn.DSSM()
    model = nn.ContentMIND()
    model = nn.YoutubeDNN()
    model = nn.TransformerRecommender()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Export Neural Network Based Algorithms
# ============================================================================

try:
    from .DSSM import DSSM
except ImportError:
    DSSM = None

try:
    from .MIND import MIND as ContentMIND
except ImportError:
    ContentMIND = None

try:
    from .TDM import TDM
except ImportError:
    TDM = None

try:
    from .Youtube_dnn import YoutubeDNN
except ImportError:
    YoutubeDNN = None

try:
    from .AITM import AITM
except ImportError:
    AITM = None

try:
    from .WidenDeep import WideAndDeep
except ImportError:
    WideAndDeep = None

try:
    from .transformer import TransformerRecommender
except ImportError:
    TransformerRecommender = None

try:
    from .cnn import CNNRecommender
except ImportError:
    CNNRecommender = None

try:
    from .rnn import RNNRecommender
except ImportError:
    RNNRecommender = None

try:
    from .autoencoder import AutoencoderRecommender
except ImportError:
    AutoencoderRecommender = None

try:
    from .vae import VAERecommender
except ImportError:
    VAERecommender = None

try:
    from .Word2Vec import Word2VecNN
except ImportError:
    Word2VecNN = None

try:
    from .dkn import DKN
except ImportError:
    DKN = None

try:
    from .lstur import LSTUR
except ImportError:
    LSTUR = None

try:
    from .naml import NAML
except ImportError:
    NAML = None

try:
    from .npa import NPA
except ImportError:
    NPA = None

try:
    from .nrms import NRMS
except ImportError:
    NRMS = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    'DSSM',
    'ContentMIND',
    'TDM',
    'YoutubeDNN',
    'AITM',
    'WideAndDeep',
    'TransformerRecommender',
    'CNNRecommender',
    'RNNRecommender',
    'AutoencoderRecommender',
    'VAERecommender',
    'Word2VecNN',
    'DKN',
    'LSTUR',
    'NAML',
    'NPA',
    'NRMS',
]
