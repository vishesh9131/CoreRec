# engines/__init__.py
# from .contentFilterEngine import CollaborativeFilteringEngine
# from .unionizedFilterEngine import UnionizedFilterEngine
# from .hybrid import HybridEngine

# Alias the package to UF_Engine, CF_Engine
# import corerec.engines.unionizedFilterEngine as UF_Engine
# import corerec.engines.contentFilterEngine as CF_Engine

from . import unionizedFilterEngine as UF_Engine
from . import contentFilterEngine as CF_Engine

# __all__ = [
#     'HybridEngine',
#     'UF_Engine',
#     'CF_Engine',
# ]
