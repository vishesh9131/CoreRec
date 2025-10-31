"""
CoreRec: Advanced Recommendation Systems Library
=================================================

CoreRec provides state-of-the-art recommendation algorithms with a unified API.

Quick Start:
-----------
    from corerec import engines
    
    # Deep learning models
    model = engines.DCN(embedding_dim=64)
    model.fit(user_ids, item_ids, ratings)
    recs = model.recommend(user_id=123, top_k=10)
    
    # Unionized Filter (Collaborative)
    model = engines.unionized.FastRecommender()
    
    # Content Filter
    model = engines.content.TFIDFRecommender()

Engine-Level Organization:
--------------------------
    from corerec import engines
    
    # Deep Learning Models
    engines.DCN, engines.DeepFM, engines.GNNRec
    engines.MIND, engines.NASRec, engines.SASRec
    
    # Unionized Filter Engine (Collaborative)
    engines.unionized.FastRecommender
    engines.unionized.SAR
    engines.unionized.LightGCN
    
    # Content Filter Engine
    engines.content.TFIDFRecommender
    engines.content.AttentionMechanisms
    engines.content.EnsembleRecommender

Author: Vishesh Yadav (sciencely98@gmail.com)
License: Research purposes only
"""

__version__ = "0.5.1"
__author__ = "Vishesh Yadav"
__email__ = "sciencely98@gmail.com"

# ============================================================================
# HIGH-LEVEL API - Sub-modules for organized access
# ============================================================================

# Import engines module (provides organized access to all algorithms)
from . import engines

# Import core components
from . import core

# Import utilities
from . import utils
from . import metrics
from . import evaluation
from . import vish_graphs
from . import visualization

# Import data and training
from . import data
from . import training
from . import trainer

# Import API base classes
from .api.base_recommender import BaseRecommender

# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Legacy base class (deprecated but maintained for backward compatibility)
from .base_recommender import BaseCorerec  # Will show deprecation warning when used

# Legacy imports that some code might depend on
try:
    from . import models  # If models module exists
except ImportError:
    pass

# ============================================================================
# __all__ - Controls what gets exported with "from corerec import *"
# ============================================================================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Main modules (Engine-Level Organization)
    'engines',      # Access all recommendation engines
    'core',         # Core components (towers, encoders, losses)
    'training',     # Training utilities
    'trainer',      # Model trainer
    'data',         # Data loading and processing
    'utils',        # Utility functions
    'metrics',      # Evaluation metrics
    'evaluation',   # Evaluation tools
    'vish_graphs',  # Graph visualization
    'visualization',# Visualization utilities
    
    # Base classes
    'BaseRecommender',
    'BaseCorerec',  # Deprecated, use BaseRecommender
]
