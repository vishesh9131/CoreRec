"""
Hybrid & Ensemble Methods
=========================

Methods for combining multiple recommendation approaches.

This module provides:
- Attention Mechanisms for hybrid recommendations
- Ensemble Methods for combining multiple models
- Hybrid Collaborative-Content filtering

Usage:
------
    from corerec.engines.content import hybrid
    
    # Direct access to classes
    attention_model = hybrid.AttentionMechanisms()
    ensemble_model = hybrid.EnsembleRecommender()
    hybrid_model = hybrid.HybridCollaborative()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Export Classes with User-Friendly Names
# ============================================================================

try:
    from .attention_mechanisms import ATTENTION_MECHANISMS as AttentionMechanisms
except ImportError:
    AttentionMechanisms = None

try:
    from .ensemble_methods import ENSEMBLE_METHODS as EnsembleRecommender
except ImportError:
    EnsembleRecommender = None

try:
    from .hybrid_collaborative import HYBRID_COLLABORATIVE as HybridCollaborative
except ImportError:
    HybridCollaborative = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    "AttentionMechanisms",
    "EnsembleRecommender",
    "HybridCollaborative",
]
