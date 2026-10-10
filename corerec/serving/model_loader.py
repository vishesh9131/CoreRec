"""
Model Loader for Production

Handles model loading with caching and versioning.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging


class ModelLoader:
    """
    Production model loader with caching.

    Loads models efficiently with caching to avoid repeated loading.

    Example:
        loader = ModelLoader()
        model = loader.load('models/ncf_v1.pkl')

        # Subsequent loads use cache
        model2 = loader.load('models/ncf_v1.pkl')  # From cache!

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for model cache

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.cache: Dict[tuple, Any] = {}
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("ModelLoader")

    def load(self, model_path: str, use_cache: bool = True, model_class: Any = None,
             *, allow_pickle: bool = False) -> Any:
        """
        Load a model saved with ``model.save(model_path)``.

        Models don't all write one file at ``model_path``: torch models write
        ``<stem>.meta.json`` (+ weights/arrays) next to it, classic CF models
        pickle a state dict. Each format carries the class name, so this reads
        it and hands off to that class's own ``load()``. Legacy files require
        ``allow_pickle=True``; safe metadata can select only registered classes.

        Args:
            model_path: the same path that was passed to ``save()``
            use_cache: Whether to use cached model
            model_class: skip detection and call ``model_class.load(model_path)``
            allow_pickle: explicitly trust legacy files that can execute Python

        Returns:
            The reconstructed model
        """
        from corerec.api.model_bundle import is_safe_bundle, require_legacy_pickle
        if not is_safe_bundle(model_path) and not allow_pickle:
            require_legacy_pickle(model_path)
        key = (str(model_path), allow_pickle, model_class)
        if use_cache and key in self.cache:
            self.logger.info(f"Loading model from cache: {key}")
            return self.cache[key]

        self.logger.info(f"Loading model from disk: {key}")
        cls = model_class or _detect_class(Path(model_path), allow_pickle=allow_pickle)
        load_kwargs = {"allow_pickle": True} if allow_pickle else {}
        model = cls.load(str(model_path), **load_kwargs)

        if use_cache:
            self.cache[key] = model
        return model

    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        self.logger.info("Model cache cleared")


def _engine_class(name: str):
    from corerec import engines

    cls = getattr(engines, name, None)
    if cls is None:
        raise ValueError(f"Saved model names class {name!r}, which corerec.engines doesn't export")
    return cls


def _detect_class(path: Path, *, allow_pickle: bool = False):
    """Find the model class recorded in whatever save() wrote for ``path``."""
    import importlib
    import json
    import pickle

    from corerec.api.model_bundle import is_safe_bundle, require_legacy_pickle
    if is_safe_bundle(path):
        from corerec.engines import MODELS
        from corerec.api.model_bundle import bundle_meta_path
        dotted = json.loads(bundle_meta_path(path).read_text(encoding="utf-8"))["model_class"]
        allowed = {f"corerec.engines{module}.{name}": name
                   for name, (module, _, _) in MODELS.items()}
        if dotted in allowed:
            return _engine_class(allowed[dotted])
        if dotted == "corerec.nn.recommender.Recommender":
            from corerec.nn import Recommender
            return Recommender
        raise ValueError("Bundle model_class is not a registered CoreRec model; "
                         "pass model_class= explicitly for a custom model")

    require_legacy_pickle(path, allow_pickle)

    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
    except Exception:
        # MultVAE/MultiDAE use torch.save
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        if state.get("corerec_class"):  # corerec.nn.Recommender and friends
            module, _, name = state["corerec_class"].rpartition(".")
            return getattr(importlib.import_module(module), name)
        name = state.get("cls") or state.get("params", {}).get("name") or state.get("cfg", {}).get("name")
        if name:
            return _engine_class(name)
    elif hasattr(state, "recommend"):
        # pickled model object (old saves): wrap so .load() just returns it
        class _Loaded:
            @staticmethod
            def load(_, **kwargs):
                return state

        return _Loaded
    raise ValueError(f"Can't tell which model class wrote {path}; pass model_class=")
