"""
CoreRec Engines
===============

Three main recommendation engines with organized access:

1. Deep Learning Models (DCN, DeepFM, GNNRec, MIND, NASRec, SASRec)
2. Unionized Filter Engine (Collaborative filtering algorithms)
3. Content Filter Engine (Content-based filtering algorithms)

Usage:
------
    from corerec.engines.collaborative import SAR
    model = SAR(similarity_type='jaccard')
    
    # or for deep learning
    from corerec import engines
    model = engines.DCN(embedding_dim=64)

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# LAZY IMPORTS - heavy modules only loaded when accessed
# ============================================================================

_deep_learning_models = {
    "DCN": ".dcn",
    "DeepFM": ".deepfm", 
    "GNNRec": ".gnnrec",
    "MIND": ".mind",
    "NASRec": ".nasrec",
    "SASRec": ".sasrec",
    "TwoTower": ".two_tower",
    "BERT4Rec": ".bert4rec",
}

_submodules = {
    "collaborative",
    "content_based",
    "unionized",  # alias for collaborative
    "content",    # alias for content_based
}


def __getattr__(name):
    """Lazy import handler."""
    import importlib
    
    # deep learning models
    if name in _deep_learning_models:
        mod_name = _deep_learning_models[name]
        try:
            mod = importlib.import_module(mod_name, __name__)
            cls = getattr(mod, name)
            globals()[name] = cls
            return cls
        except (ImportError, AttributeError):
            globals()[name] = None
            return None
    
    # submodules
    if name == "unionized" or name == "collaborative":
        mod = importlib.import_module(".collaborative", __name__)
        globals()["collaborative"] = mod
        globals()["unionized"] = mod
        globals()["UF_Engine"] = mod  # legacy alias
        return mod
    
    if name == "content" or name == "content_based":
        mod = importlib.import_module(".content_based", __name__)
        globals()["content_based"] = mod
        globals()["content"] = mod
        globals()["CF_Engine"] = mod  # legacy alias
        return mod
    
    # legacy aliases
    if name == "UF_Engine":
        return __getattr__("unionized")
    if name == "CF_Engine":
        return __getattr__("content")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    # Deep Learning Models
    "DCN",
    "DeepFM",
    "GNNRec",
    "MIND",
    "NASRec",
    "SASRec",
    "TwoTower",
    "BERT4Rec",
    # Engine Namespaces
    "unionized",
    "content",
    "collaborative",
    "content_based",
    # Legacy aliases
    "UF_Engine",
    "CF_Engine",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_deep_learning_models():
    """List all available deep learning models."""
    available = []
    for name in _deep_learning_models:
        try:
            if __getattr__(name) is not None:
                available.append(name)
        except (ImportError, AttributeError):
            pass
    return available


def get_engine_info():
    """Get information about available engines."""
    return {
        "deep_learning": list(_deep_learning_models.keys()),
        "unionized_filter": "engines.unionized or engines.collaborative",
        "content_filter": "engines.content or engines.content_based",
    }
