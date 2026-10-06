"""
CoreRec Engines
===============

Every model CoreRec ships, behind one registry (``MODELS``). The CLI, the
docs counts and the contract tests all read from it, so there is exactly one
place that says what the zoo contains.

Usage:
------
    from corerec.engines import ALS, SASRec, TwoTower
    model = ALS(factors=64)
    model.fit(user_ids, item_ids, ratings)
    model.recommend(user_id=1, top_k=10)

Models removed in 0.7.0 (GNNRec, MIND, NASRec, BERT4Rec, NCF, NGCF, the deep
CTR family and the GRU4Rec/Caser/BST/DIN/DIEN/NARM sequential family) live in
the git history at commit 33911a3.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# REGISTRY - the single source of truth for the model zoo
# ============================================================================

# name -> (module, family, one-line summary). Modules are imported lazily.
MODELS = {
    # classic collaborative filtering: fast, CPU-only, strong baselines
    "ALS": (".matrix_factorization", "classic", "Alternating least squares matrix factorization"),
    "SAR": (".collaborative.sar", "classic", "Simple Algorithm for Recommendation (item co-occurrence)"),
    "ItemKNN": (".classic_cf", "classic", "Item-based nearest neighbours"),
    "UserKNN": (".classic_cf", "classic", "User-based nearest neighbours"),
    "EASE": (".classic_cf", "classic", "Embarrassingly shallow autoencoder (closed form)"),
    "SLIM": (".classic_cf", "classic", "Sparse linear item-item model"),
    "Item2Vec": (".matrix_factorization", "classic", "Skip-gram item embeddings"),
    # retrieval and graph
    "TwoTower": (".two_tower", "retrieval", "Dual-encoder candidate retrieval"),
    "LightGCN": (".collaborative.graph_based_base.lightgcn", "graph", "Simplified graph convolution CF"),
    # ranking
    "DCN": (".dcn", "ranking", "Deep & Cross Network"),
    "DeepFM": (".deepfm", "ranking", "Factorization machine + deep network"),
    # sequential
    "SASRec": (".sasrec", "sequential", "Self-attentive next-item prediction"),
    # autoencoders
    "MultVAE": (".vae_cf", "autoencoder", "Variational autoencoder for implicit feedback"),
    "MultiDAE": (".vae_cf", "autoencoder", "Denoising autoencoder for implicit feedback"),
    # content-based
    "TFIDFRecommender": (".content_based.tfidf_recommender", "content", "TF-IDF text similarity"),
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

    if name in MODELS:
        mod_name = MODELS[name][0]
        try:
            mod = importlib.import_module(mod_name, __name__)
            cls = getattr(mod, name)
        except (ImportError, AttributeError) as exc:
            # Fail where the cause is visible instead of handing back None.
            raise AttributeError(
                f"{__name__}.{name} is unavailable: importing {mod_name} failed ({exc})"
            ) from exc
        globals()[name] = cls
        return cls

    if name in ("unionized", "collaborative"):
        mod = importlib.import_module(".collaborative", __name__)
        globals()["collaborative"] = mod
        globals()["unionized"] = mod
        return mod

    if name in ("content", "content_based"):
        mod = importlib.import_module(".content_based", __name__)
        globals()["content_based"] = mod
        globals()["content"] = mod
        return mod

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    *MODELS,
    # Engine namespaces
    "unionized",
    "content",
    "collaborative",
    "content_based",
    # Registry and helpers
    "MODELS",
    "list_models",
    "get_engine_info",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_models(family=None):
    """Names of the shipped models, optionally filtered by family."""
    return [n for n, (_, fam, _) in MODELS.items() if family is None or fam == family]


def get_engine_info():
    """Models grouped by family, with their one-line summaries."""
    info = {}
    for name, (_, family, summary) in MODELS.items():
        info.setdefault(family, {})[name] = summary
    return info
