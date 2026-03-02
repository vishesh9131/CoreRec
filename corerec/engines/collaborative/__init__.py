"""
Collaborative Filtering Engine
==============================

Production-ready collaborative filtering methods:

1. SAR - Simple Algorithm for Recommendation (fast, no DL needed)
2. TwoTower - Modern retrieval architecture  
3. LightGCN - Graph-based collaborative filtering
4. NCF - Neural Collaborative Filtering
5. FastRecommender - Quick prototyping

Usage:
------
    from corerec.engines.collaborative import SAR
    
    model = SAR(similarity_type='jaccard')
    model.fit(train_df)
    recs = model.recommend_k_items(test_df, top_k=10)

Author: Vishesh Yadav
"""

# ============================================================================
# LAZY IMPORTS - only load what's requested
# ============================================================================

_model_imports = {
    "SAR": (".sar", "SAR"),
    "TwoTower": ("corerec.engines.two_tower", "TwoTower"),
    "LightGCN": (".graph_based_base.lightgcn_base", "LightGCN"),
    "NCF": (".nn_base.ncf", "NCF"),
    "FastRecommender": (".fast_recommender", "FastRecommender"),
    # legacy/deprecated
    "RBM": (".rbm", "RBM"),
    "GeoMLC": (".geomlc", "GeoMLC"),
}

# alternate import paths for some models
_fallback_imports = {
    "NCF": (".nn_base.ncf_base", "NCF"),
    "FastRecommender": (".fast", "FastRecommender"),
}


def __getattr__(name):
    """Lazy import handler."""
    import importlib
    
    if name in _model_imports:
        mod_path, cls_name = _model_imports[name]
        try:
            # handle absolute vs relative imports
            if mod_path.startswith("corerec"):
                mod = importlib.import_module(mod_path)
            else:
                mod = importlib.import_module(mod_path, __name__)
            cls = getattr(mod, cls_name)
            globals()[name] = cls
            return cls
        except (ImportError, AttributeError):
            # try fallback
            if name in _fallback_imports:
                fb_path, fb_cls = _fallback_imports[name]
                try:
                    mod = importlib.import_module(fb_path, __name__)
                    cls = getattr(mod, fb_cls)
                    globals()[name] = cls
                    return cls
                except (ImportError, AttributeError):
                    pass
            globals()[name] = None
            return None
    
    if name == "sandbox":
        globals()["sandbox"] = SandboxAccess()
        return globals()["sandbox"]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


# ============================================================================
# Sandbox Access
# ============================================================================

class SandboxAccess:
    """Gateway to experimental methods under development."""
    
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


# ============================================================================
# __all__ - Export List
# ============================================================================

__all__ = [
    # Production methods
    "SAR",
    "TwoTower",
    "LightGCN",
    "NCF",
    "FastRecommender",
    # Legacy
    "RBM",
    "GeoMLC",
    # Sandbox
    "sandbox",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_methods():
    """List the production-ready methods."""
    available = []
    for name in ["SAR", "TwoTower", "LightGCN", "NCF", "FastRecommender"]:
        try:
            if __getattr__(name) is not None:
                available.append(name)
        except (ImportError, AttributeError):
            pass
    return available


def get_recommendation():
    """Get recommendation on which method to use."""
    return """
    Recommendation Guide:
    
    Use SAR if:
    - Quick baseline needed
    - Simple item-to-item similarity
    - No deep learning infrastructure
    
    Use TwoTower if:
    - Large item catalog (>100K items)
    - Need real-time serving
    - First stage of pipeline
    
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
    """
