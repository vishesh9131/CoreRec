"""
Unionized Filter Engine
=======================

Collaborative filtering and hybrid recommendation algorithms.

This engine provides 50+ collaborative filtering algorithms organized by category:
- Matrix Factorization (SVD, ALS, NMF, PMF, etc.)
- Neural Network Based (NCF, DeepFM, AutoInt, DCN, etc.)
- Graph-Based (LightGCN, DeepWalk, GNN, etc.)
- Attention Mechanisms (SASRec, Transformers, etc.)
- Bayesian Methods (BPR, Bayesian MF, etc.)
- Sequential Models (LSTM, GRU, Caser, etc.)
- Variational Encoders (VAE, CVAE, etc.)

Usage:
------
    from corerec.engines import unionized
    
    # Popular algorithms - direct access
    model = unionized.FastRecommender()
    model = unionized.SAR()
    model = unionized.RBM()
    
    # Matrix Factorization
    model = unionized.mf.SVD(n_factors=50)
    model = unionized.mf.ALS()
    
    # Neural Networks
    model = unionized.nn.NCF(embedding_dim=64)
    model = unionized.nn.DeepFM()
    
    # Graph-Based
    model = unionized.graph.LightGCN(embedding_dim=64)

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Most Popular Algorithms - Direct access
# ============================================================================

try:
    from .fast import FastRecommender
except ImportError:
    FastRecommender = None

try:
    from .sar import SAR
except ImportError:
    SAR = None

try:
    from .rbm import RBM
except ImportError:
    RBM = None

try:
    from .rlrmc import RLRMC
except ImportError:
    RLRMC = None

try:
    from .geomlc import GeoMLC
except ImportError:
    GeoMLC = None

try:
    from .fast_recommender import FastRecommender as FastRec
except ImportError:
    FastRec = None

try:
    from .sli import SLI
except ImportError:
    SLI = None

try:
    from .sum import SUM
except ImportError:
    SUM = None

# ============================================================================
# Organized Sub-modules by Algorithm Category
# ============================================================================

# Matrix Factorization algorithms
from . import mf_base as mf

# Neural Network based algorithms
from . import nn_base as nn

# Graph-based algorithms
from . import graph_based_base as graph

# Attention mechanism based algorithms
from . import attention_mechanism_base as attention

# Bayesian methods
from . import bayesian_method_base as bayesian

# Sequential models
from . import sequential_model_base as sequential

# Variational encoders
from . import variational_encoder_base as vae

# Regularization based methods
try:
    from . import regularization_based_base as regularization
except ImportError:
    regularization = None

# ============================================================================
# Legacy/Compatibility
# ============================================================================

try:
    from .base_recommender import BaseRecommender as UnionizedBaseRecommender
except ImportError:
    UnionizedBaseRecommender = None

try:
    from .cornac_bpr import BPR as CornacBPR
except ImportError:
    CornacBPR = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    # Popular algorithms (direct access)
    'FastRecommender',
    'FastRec',
    'SAR',
    'RBM',
    'RLRMC',
    'GeoMLC',
    'SLI',
    'SUM',
    
    # Organized sub-modules
    'mf',              # Matrix Factorization (SVD, ALS, NMF, etc.)
    'nn',              # Neural Networks (NCF, DeepFM, etc.)
    'graph',           # Graph-based (LightGCN, GNN, etc.)
    'attention',       # Attention mechanisms (SASRec, etc.)
    'bayesian',        # Bayesian methods (BPR, etc.)
    'sequential',      # Sequential models (LSTM, GRU, etc.)
    'vae',             # Variational encoders
    'regularization',  # Regularization methods
    
    # Base classes
    'UnionizedBaseRecommender',
    'CornacBPR',
]

# ============================================================================
# Helper Functions
# ============================================================================

def list_algorithms():
    """List all available algorithms in this engine."""
    algorithms = []
    
    # Direct access algorithms
    if FastRecommender is not None:
        algorithms.append('FastRecommender')
    if SAR is not None:
        algorithms.append('SAR')
    if RBM is not None:
        algorithms.append('RBM')
    if RLRMC is not None:
        algorithms.append('RLRMC')
    if GeoMLC is not None:
        algorithms.append('GeoMLC')
    
    return algorithms

def list_categories():
    """List all algorithm categories."""
    return [
        'mf (Matrix Factorization)',
        'nn (Neural Networks)',
        'graph (Graph-Based)',
        'attention (Attention Mechanisms)',
        'bayesian (Bayesian Methods)',
        'sequential (Sequential Models)',
        'vae (Variational Encoders)',
        'regularization (Regularization Methods)',
    ]
