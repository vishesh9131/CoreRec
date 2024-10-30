# engines/__init__.py
# from .contentFilterEngine import CollaborativeFilteringEngine
# from .unionizedFilterEngine import UnionizedFilterEngine
from .hybrid import HybridEngine
from .content_based import ContentBasedFilteringEngine

# Alias the package to UF_Engine
import corerec.engines.unionizedFilterEngine as UF_Engine
import corerec.engines.contentFilterEngine as CF_Engine

__all__ = [
    'CollaborativeFilteringEngine',
    'ContentBasedFilteringEngine',
    'HybridEngine',
    'UF_Engine',
    'CF_Engine',
]
