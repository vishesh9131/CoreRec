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
    "LightGCN": (".graph_based_base.lightgcn", "LightGCN"),
    "NCF": (".nn_base.ncf", "NCF"),
    "FAST": (".fast", "FAST"),
    "FastRecommender": (".fast_recommender", "FASTRecommender"),
    "FASTRecommender": (".fast_recommender", "FASTRecommender"),
    # legacy/deprecated
    "RBM": (".rbm", "RBM"),
    "GeoMLC": (".geomlc", "GeoMLC"),
}

# alternate import paths for some models
_fallback_imports = {
    "NCF": (".nn_base.ncf_base", "NCF"),
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
            # Public alias: FastRecommender -> FASTRecommender class
            if name == "FastRecommender":
                globals()["FASTRecommender"] = cls
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
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)




# ============================================================================
# __all__ - Export List
# ============================================================================

__all__ = [
    # Production methods
    "SAR",
    "TwoTower",
    "LightGCN",
    "NCF",
    "FAST",
    "FastRecommender",
    "FASTRecommender",
    # Legacy
    "RBM",
    "GeoMLC",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_methods():
    """List the production-ready methods."""
    available = []
    for name in ["SAR", "TwoTower", "LightGCN", "NCF", "FAST", "FastRecommender"]:
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
