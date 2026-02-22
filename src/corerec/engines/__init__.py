# engines/__init__.py
# from .content_based import CollaborativeFilteringEngine
# from .collaborative import UnionizedFilterEngine
# from .hybrid import HybridEngine

# Alias the package to UF_Engine, CF_Engine
# import corerec.engines.collaborative as UF_Engine
# import corerec.engines.content_based as CF_Engine

from . import collaborative as UF_Engine
from . import content_based as CF_Engine

# __all__ = [
#     'HybridEngine',
#     'UF_Engine',
#     'CF_Engine',
# ]
