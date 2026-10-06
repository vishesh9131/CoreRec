"""
CoreRec: Advanced Recommendation Systems Library
=================================================

CoreRec provides state-of-the-art recommendation algorithms with a unified API.

Quick Start:
-----------
    import corerec.collaborative as clb
    
    model = clb.SAR(similarity_type='jaccard')
    model.fit(train_df)
    recs = model.recommend_k_items(test_df, top_k=10)

Alternative imports:
    from corerec.collaborative import SAR
    from corerec.content_based import TFIDFRecommender

Author: Vishesh Yadav (sciencely98@gmail.com)
License: MIT (see pyproject.toml)
"""

__version__ = "0.6.0"
__author__ = "Vishesh Yadav"
__email__ = "sciencely98@gmail.com"


# ============================================================================
# LAZY IMPORTS - modules loaded only when accessed
# ============================================================================
# This prevents loading heavy deps (sklearn, torch, matplotlib) on every import

def __getattr__(name):
    """Lazy import handler - loads modules on first access."""
    import importlib
    
    # shortcut aliases - corerec.collaborative -> corerec.engines.collaborative
    _shortcuts = {
        "collaborative": ".engines.collaborative",
        "content_based": ".engines.content_based",
    }
    
    if name in _shortcuts:
        mod = importlib.import_module(_shortcuts[name], __name__)
        globals()[name] = mod
        return mod
    
    # submodules that should be importable
    _submodules = {
        "engines",
        "core",
        "utils",
        "metrics",
        "evaluation",
        "data",
        "training",
        "trainer",
        "pipelines",
        "retrieval",
        "ranking",
        "reranking",
        "multimodal",
        "embeddings",
        "explanation",
        "api",
        "hybrid",
        "serving",
    }
    
    # Submodules whose dependencies live behind an optional extra, so the
    # failure can be explained instead of looking like the module is missing.
    _extras = {
        "serving": "serving",
        "multimodal": "transformers",
    }

    if name in _submodules:
        try:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
            return module
        except ImportError as exc:
            extra = _extras.get(name)
            if extra:
                raise AttributeError(
                    f"corerec.{name} needs its optional dependencies: "
                    f"pip install corerec[{extra}]  ({exc})"
                ) from exc
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} ({exc})"
            ) from exc
    
    # constants - these are lightweight, load directly
    _constants = {
        "DEFAULT_USER_COL",
        "DEFAULT_ITEM_COL", 
        "DEFAULT_RATING_COL",
        "DEFAULT_TIMESTAMP_COL",
        "DEFAULT_PREDICTION_COL",
        "SIM_COOCCURRENCE",
        "SIM_COSINE",
        "SIM_JACCARD",
        "SIM_LIFT",
        "SIM_INCLUSION_INDEX",
        "SIM_MUTUAL_INFORMATION",
        "SIM_LEXICOGRAPHERS_MI",
        "SUPPORTED_SIMILARITY_TYPES",
    }
    
    if name in _constants:
        from . import constants
        val = getattr(constants, name)
        globals()[name] = val
        return val
    
    # base classes
    if name == "BaseRecommender":
        from .api.base_recommender import BaseRecommender
        globals()["BaseRecommender"] = BaseRecommender
        return BaseRecommender
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List available attributes."""
    return list(__all__)


# ============================================================================
# __all__ - Controls what gets exported with "from corerec import *"
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Shortcut aliases (recommended)
    "collaborative",    # -> corerec.engines.collaborative
    "content_based",    # -> corerec.engines.content_based
    # Main modules (lazy loaded)
    "engines",
    "core",
    "training",
    "trainer",
    "data",
    "utils",
    "metrics",
    "evaluation",
    "pipelines",
    "retrieval",
    "ranking",
    "reranking",
    "multimodal",
    "embeddings",
    "explanation",
    "hybrid",
    "serving",
    # Base classes
    "BaseRecommender",
    # NOTE: 35 names were removed here (TransferLearning, ZeroShot, MetaLearning,
    # MultiModal, CrossDomain, ColdStart, FairnessAware, the LEA_/MUL_/OTH_/MIS_
    # aliases, ...). Every one of them resolved to None: the modules behind them
    # do not exist. They are gone from __all__ rather than re-listed as broken.
    # Constants
    "DEFAULT_USER_COL",
    "DEFAULT_ITEM_COL",
    "DEFAULT_RATING_COL",
    "DEFAULT_TIMESTAMP_COL",
    "DEFAULT_PREDICTION_COL",
    "SIM_COOCCURRENCE",
    "SIM_COSINE",
    "SIM_JACCARD",
    "SIM_LIFT",
    "SIM_INCLUSION_INDEX",
    "SIM_MUTUAL_INFORMATION",
    "SIM_LEXICOGRAPHERS_MI",
    "SUPPORTED_SIMILARITY_TYPES",
]
