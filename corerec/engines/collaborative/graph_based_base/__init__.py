"""
Graph-based collaborative filtering algorithms.

Imports are lazy so a missing sandbox module does not block the package.
"""

import importlib as _il
import logging as _log

_logger = _log.getLogger(__name__)

_IMPORTS = {
    "GB_EDGE_AWARE_CF": (".edge_aware_ufilter_base", "EdgeAwareCFBase"),
    "GB_HETEROGENEOUS_NETWORK_UF": (".heterogeneous_network_ufilter_base", "HeterogeneousNetworkUFBase"),
    "GB_GEOIMC": (".geoimc_base", "GeoIMCBase"),
    "GB_DEEPWALK": (".DeepWalk_base", "DeepWalk_base"),
    "GB_EDGE_AWARE": (".edge_aware_ufilter_base", "EdgeAwareCFBase"),
    "GB_GNN": (".GNN_base", "GNN_base"),
    "GB_CF": (".gnn_ufilter_base", "GraphBasedCFBase"),
    "GB_UF": (".graph_based_ufilter_base", "GraphBasedUFBase"),
    "GB_LIGHTGCN": (".lightgcn", "LightGCN"),
    "GB_MULTI_REL_UF": (".multi_relational_ufilter_base", "MultiRelationalUFBase"),
    "GB_MULTI_VIEW_UF": (".multi_view_ufilter_base", "MultiViewUFBase"),
    "LightGCN": (".lightgcn", "LightGCN"),
}


def __getattr__(name):
    if name in _IMPORTS:
        mod_path, cls_name = _IMPORTS[name]
        try:
            mod = _il.import_module(mod_path, __name__)
            cls = getattr(mod, cls_name)
            globals()[name] = cls
            return cls
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            _logger.debug("Optional import %s failed: %s", name, exc)
            globals()[name] = None
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_IMPORTS.keys())
