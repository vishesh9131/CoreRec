"""
CoreRec Engines
===============

Three main recommendation engines with organized access:

1. Deep Learning Models (DCN, DeepFM, GNNRec, MIND, NASRec, SASRec)
2. Unionized Filter Engine (Collaborative filtering algorithms)
3. Content Filter Engine (Content-based filtering algorithms)

Usage:
------
    from corerec import engines
    
    # Deep Learning Models - Direct access
    model = engines.DCN(embedding_dim=64)
    model = engines.DeepFM(embedding_dim=64)
    model = engines.SASRec(hidden_units=64)
    
    # Unionized Filter - Organized namespace
    model = engines.unionized.FastRecommender()
    model = engines.unionized.SAR()
    model = engines.unionized.LightGCN(embedding_dim=64)
    
    # Content Filter - Organized namespace
    model = engines.content.TFIDFRecommender()
    model = engines.content.AttentionMechanisms()
    model = engines.content.EnsembleRecommender()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Deep Learning Models - Direct access at engines level
# ============================================================================

try:
    from .dcn import DCN
except ImportError:
    DCN = None

try:
    from .deepfm import DeepFM
except ImportError:
    DeepFM = None

try:
    from .gnnrec import GNNRec
except ImportError:
    GNNRec = None

try:
    from .mind import MIND
except ImportError:
    MIND = None

try:
    from .nasrec import NASRec
except ImportError:
    NASRec = None

try:
    from .sasrec import SASRec
except ImportError:
    SASRec = None

try:
    from .two_tower import TwoTower
except ImportError:
    TwoTower = None

try:
    from .bert4rec import BERT4Rec
except ImportError:
    BERT4Rec = None

# ============================================================================
# Engine Namespaces - Organized access to algorithm families
# ============================================================================

# Unionized Filter Engine (Collaborative Filtering)
from . import collaborative as unionized

# Content Filter Engine (Content-Based Filtering)
from . import content_based as content

# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Legacy names that existing code might use
UF_Engine = unionized
CF_Engine = content

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    # Deep Learning Models (direct access)
    "DCN",
    "DeepFM",
    "GNNRec",
    "MIND",
    "NASRec",
    "SASRec",
    "TwoTower",
    "BERT4Rec",
    # Engine Namespaces (organized access)
    "unionized",  # Collaborative filtering algorithms
    "content",  # Content-based filtering algorithms
    # Legacy aliases (backward compatibility)
    "UF_Engine",
    "CF_Engine",
]

# ============================================================================
# Helper Functions
# ============================================================================


def list_deep_learning_models():
    """List all available deep learning models."""
    models = []
    if DCN is not None:
        models.append("DCN")
    if DeepFM is not None:
        models.append("DeepFM")
    if GNNRec is not None:
        models.append("GNNRec")
    if MIND is not None:
        models.append("MIND")
    if NASRec is not None:
        models.append("NASRec")
    if SASRec is not None:
        models.append("SASRec")
    if TwoTower is not None:
        models.append("TwoTower")
    if BERT4Rec is not None:
        models.append("BERT4Rec")
    return models


def get_engine_info():
    """Get information about available engines."""
    return {
        "deep_learning": list_deep_learning_models(),
        "unionized_filter": "engines.unionized",
        "content_filter": "engines.content",
    }
