"""
Collaborative Filtering Engine (Refactored)
============================================

This engine provides the TOP 5 most useful collaborative filtering methods.
All other methods are available in the sandbox for development.

Top 5 Production-Ready Methods:
--------------------------------
1. TwoTower - Modern retrieval (new standard)
2. SAR - Simple Algorithm for Recommendation
3. LightGCN - Graph-based collaborative filtering  
4. NCF - Neural Collaborative Filtering
5. FastRecommender - FastAI-style quick prototyping

Other 45+ algorithms moved to sandbox for refinement.

Usage:
------
    from corerec.engines import unionized
    
    # Modern retrieval
    model = unionized.TwoTower(embedding_dim=256)
    
    # Fast and simple
    model = unionized.SAR()
    
    # Graph-based
    model = unionized.LightGCN(embedding_dim=128)
    
    # Neural collaborative
    model = unionized.NCF(embedding_dim=64)
    
    # Quick prototyping
    model = unionized.FastRecommender()

Sandbox Access:
---------------
    # For experimental methods, import from sandbox
    from corerec.sandbox.collaborative import DeepFM, DCN, etc.

Author: Vishesh Yadav
"""

# ============================================================================
# TOP 5 PRODUCTION-READY METHODS
# ============================================================================

# 1. TwoTower (Modern Standard - NEW)
try:
    from corerec.engines.two_tower import TwoTower
except ImportError:
    TwoTower = None

# 2. SAR (Simple, Fast, Proven)
try:
    from .sar import SAR
except ImportError:
    SAR = None

# 3. LightGCN (Graph-based, Modern)
try:
    from .graph_based_base.lightgcn_base import LightGCN
except ImportError:
    LightGCN = None

# 4. NCF (Neural Collaborative Filtering - Foundation)
try:
    from .nn_base.ncf import NCF
except ImportError:
    try:
        from .nn_base.ncf_base import NCF
    except ImportError:
        NCF = None

# 5. FastRecommender (Quick Prototyping)
try:
    from .fast_recommender import FastRecommender
except ImportError:
    try:
        from .fast import FastRecommender
    except ImportError:
        FastRecommender = None

# ============================================================================
# Backward Compatibility - Deprecated but Available
# ============================================================================

# Keep these for backward compat, but discourage new usage
try:
    from .rbm import RBM
except ImportError:
    RBM = None

try:
    from .geomlc import GeoMLC
except ImportError:
    GeoMLC = None


# ============================================================================
# Sandbox Access (Development/Experimental)
# ============================================================================

class SandboxAccess:
    """
    Gateway to experimental methods under development.
    
    These methods are functional but not yet production-ready.
    Use at your own risk for research/experimentation.
    """
    
    @staticmethod
    def list_available():
        """List all sandbox methods."""
        return [
            "Matrix Factorization: SVD, ALS, NMF, PMF, BPR",
            "Neural Networks: DeepFM, DCN, AutoInt, PNN, xDeepFM",
            "Sequential: GRU, LSTM, Caser, SASRec, NextItNet",
            "Attention: Transformer-based, DIEN, DIN, BST",
            "Graph: DeepWalk, GNN variants",
            "Variational: VAE, BiVAE, CVAE",
            "Bayesian: Bayesian MF, MCMC methods",
            "Others: RLRMC, SLI, SUM, etc.",
            "",
            "Total: 45+ methods in sandbox",
            "Import from: corerec.sandbox.collaborative"
        ]
    
    @staticmethod
    def get_info(method_name):
        """Get info about a sandbox method."""
        info_map = {
            "DeepFM": "Deep Factorization Machine - combines FM with deep learning",
            "DCN": "Deep & Cross Network - explicit feature crossing",
            "SASRec": "Self-Attentive Sequential Rec - transformer for sequences",
            "GRU": "GRU-based sequential recommendation",
            "DeepWalk": "Random walk based graph embeddings",
        }
        return info_map.get(method_name, "No info available. Check sandbox docs.")


sandbox = SandboxAccess()


# ============================================================================
# __all__ - Export List (ONLY Top 5)
# ============================================================================

__all__ = [
    # Top 5 Production Methods
    "TwoTower",      # Modern standard
    "SAR",           # Simple & fast
    "LightGCN",      # Graph-based
    "NCF",           # Neural collab
    "FastRecommender",  # Quick proto
    
    # Backward compat (deprecated)
    "RBM",
    "GeoMLC",
    
    # Sandbox gateway
    "sandbox",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_methods():
    """List the top 5 production-ready methods."""
    methods = []
    
    if TwoTower is not None:
        methods.append("TwoTower - Modern retrieval standard")
    if SAR is not None:
        methods.append("SAR - Simple Algorithm for Recommendation")
    if LightGCN is not None:
        methods.append("LightGCN - Graph-based collaborative filtering")
    if NCF is not None:
        methods.append("NCF - Neural Collaborative Filtering")
    if FastRecommender is not None:
        methods.append("FastRecommender - Fast prototyping")
    
    return methods


def get_recommendation():
    """Get recommendation on which method to use."""
    return """
    Recommendation Guide:
    
    Use TwoTower if:
    - Large item catalog (>100K items)
    - Need real-time serving
    - First stage of pipeline
    
    Use SAR if:
    - Quick baseline needed
    - Simple item-to-item similarity
    - No deep learning infrastructure
    
    Use LightGCN if:
    - Have user-item graph structure
    - Want to leverage network effects
    - Social recommendation scenario
    
    Use NCF if:
    - Learning collaborative patterns
    - Mid-size dataset
    - Need interpretable embeddings
    
    Use FastRecommender if:
    - Rapid prototyping
    - Simple embedding-based model
    - Educational/demo purposes
    
    For other methods, check sandbox.list_available()
    """


def migrate_to_sandbox_notice():
    """Information about the refactoring."""
    return """
    ⚠️  REFACTORING NOTICE
    
    45+ methods have been moved to sandbox for quality refinement:
    - All methods are still accessible
    - Import from: corerec.sandbox.collaborative
    - Top 5 methods remain in main engine
    - Sandbox methods will graduate when production-ready
    
    This ensures main engine stays lean and battle-tested.
    """
