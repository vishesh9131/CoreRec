"""
Collaborative Filtering Engine
==============================

Namespace for the collaborative models that live in this package, plus
TwoTower for backwards compatibility. Everything here is also importable from
``corerec.engines`` directly.

Usage:
------
    from corerec.engines.collaborative import SAR

    model = SAR(similarity_type='jaccard')
    model.fit(train_df)
    recs = model.recommend_k_items(test_df, top_k=10)

Author: Vishesh Yadav
"""

_model_imports = {
    "SAR": (".sar", "SAR"),
    "LightGCN": (".graph_based_base.lightgcn", "LightGCN"),
    "TwoTower": ("corerec.engines.two_tower", "TwoTower"),
}


def __getattr__(name):
    """Lazy import handler."""
    import importlib

    if name in _model_imports:
        mod_path, cls_name = _model_imports[name]
        try:
            mod = importlib.import_module(mod_path, __name__)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as exc:
            raise AttributeError(
                f"{__name__}.{name} is unavailable: importing {mod_path} failed ({exc})"
            ) from exc
        globals()[name] = cls
        return cls

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)


__all__ = ["SAR", "LightGCN", "TwoTower"]
