"""
CoreRec: Advanced Recommendation Systems Library
=================================================

CoreRec provides state-of-the-art recommendation algorithms with a unified API.

Quick Start:
-----------
    from corerec import engines

    # Deep learning models
    model = engines.DCN(embedding_dim=64)
    model.fit(user_ids, item_ids, ratings)
    recs = model.recommend(user_id=123, top_k=10)

    # Unionized Filter (Collaborative)
    model = engines.unionized.FastRecommender()

    # Content Filter
    model = engines.content.TFIDFRecommender()

Engine-Level Organization:
--------------------------
    from corerec import engines

    # Deep Learning Models
    engines.DCN, engines.DeepFM, engines.GNNRec
    engines.MIND, engines.NASRec, engines.SASRec

    # Unionized Filter Engine (Collaborative)
    engines.unionized.FastRecommender
    engines.unionized.SAR
    engines.unionized.LightGCN

    # Content Filter Engine
    engines.content.TFIDFRecommender
    engines.content.AttentionMechanisms
    engines.content.EnsembleRecommender

Author: Vishesh Yadav (sciencely98@gmail.com)
License: Research purposes only
"""

__version__ = "0.5.1"
__author__ = "Vishesh Yadav"
__email__ = "sciencely98@gmail.com"

# ============================================================================
# HIGH-LEVEL API - Sub-modules for organized access
# ============================================================================
# Note: To check for Intel MKL warnings, run:
#   from corerec.utils.diagnostics import check_mkl_warning, fix_mkl_warning_instructions
#   fix_mkl_warning_instructions()
# Or run: python check_mkl_fix.py

# Import engines module (provides organized access to all algorithms)
from . import engines

# Import core components
from . import core

# Import utilities
from . import utils
from . import metrics

# Import constants for convenient access
from .constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
    SIM_COOCCURRENCE,
    SIM_COSINE,
    SIM_JACCARD,
    SIM_LIFT,
    SIM_INCLUSION_INDEX,
    SIM_MUTUAL_INFORMATION,
    SIM_LEXICOGRAPHERS_MI,
    SUPPORTED_SIMILARITY_TYPES,
)
from . import evaluation
from . import vish_graphs
from . import visualization

# Import data and training
from . import data
from . import training
from . import trainer

# Import modern components
from . import pipelines
from . import retrieval
from . import multimodal

# Import sandbox (experimental methods)
from . import sandbox

# Import API base classes
from .api.base_recommender import BaseRecommender

# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Legacy base class (deprecated but maintained for backward compatibility)
# Will show deprecation warning when used
from .base_recommender import BaseCorerec

# Legacy imports that some code might depend on
try:
    from . import models  # If models module exists
except ImportError:
    pass

# ============================================================================
# Direct Model Imports - Short access for common models
# ============================================================================

# Learning Paradigms
try:
    from .engines.content_based.learning_paradigms import LEA_TRANSFER_LEARNING, LEA_ZERO_SHOT, LEA_META_LEARNING
    # Short aliases
    TransferLearning = LEA_TRANSFER_LEARNING
    ZeroShot = LEA_ZERO_SHOT
    MetaLearning = LEA_META_LEARNING
except ImportError:
    LEA_TRANSFER_LEARNING = None
    LEA_ZERO_SHOT = None
    LEA_META_LEARNING = None
    TransferLearning = None
    ZeroShot = None
    MetaLearning = None

# Multi-modal and Cross-domain
try:
    from .engines.content_based.multi_modal_cross_domain_methods import (
        MUL_MULTI_MODAL,
        MUL_CROSS_DOMAIN,
        MUL_CROSS_LINGUAL,
    )
    # Short aliases
    MultiModal = MUL_MULTI_MODAL
    CrossDomain = MUL_CROSS_DOMAIN
    CrossLingual = MUL_CROSS_LINGUAL
except ImportError:
    MUL_MULTI_MODAL = None
    MUL_CROSS_DOMAIN = None
    MUL_CROSS_LINGUAL = None
    MultiModal = None
    CrossDomain = None
    CrossLingual = None

# Other Approaches
try:
    from .engines.content_based.other_approaches import (
        OTH_RULE_BASED,
        OTH_SENTIMENT_ANALYSIS,
        OTH_ONTOLOGY_BASED,
    )
    # Short aliases
    RuleBased = OTH_RULE_BASED
    SentimentAnalysis = OTH_SENTIMENT_ANALYSIS
    OntologyBased = OTH_ONTOLOGY_BASED
except ImportError:
    OTH_RULE_BASED = None
    OTH_SENTIMENT_ANALYSIS = None
    OTH_ONTOLOGY_BASED = None
    RuleBased = None
    SentimentAnalysis = None
    OntologyBased = None

# Miscellaneous Techniques
try:
    from .engines.content_based.miscellaneous_techniques import (
        MIS_FEATURE_SELECTION,
        MIS_NOISE_HANDLING,
        MIS_COLD_START,
    )
    # Short aliases
    FeatureSelection = MIS_FEATURE_SELECTION
    NoiseHandling = MIS_NOISE_HANDLING
    ColdStart = MIS_COLD_START
except ImportError:
    MIS_FEATURE_SELECTION = None
    MIS_NOISE_HANDLING = None
    MIS_COLD_START = None
    FeatureSelection = None
    NoiseHandling = None
    ColdStart = None

# CNN (already exists in corerec/cnn.py, but add direct import)
try:
    from .cnn import CNN
except ImportError:
    try:
        from .engines.content_based.nn_based_algorithms.cnn import CNN
    except ImportError:
        CNN = None

# Performance Scalability
try:
    from .engines.content_based.performance_scalability import (
        PER_SCALABLE_ALGORITHMS,
        PER_FEATURE_EXTRACTION,
        PER_LOAD_BALANCING,
    )
    # Short aliases
    ScalableAlgorithms = PER_SCALABLE_ALGORITHMS
    FeatureExtraction = PER_FEATURE_EXTRACTION
    LoadBalancing = PER_LOAD_BALANCING
except ImportError:
    PER_SCALABLE_ALGORITHMS = None
    PER_FEATURE_EXTRACTION = None
    PER_LOAD_BALANCING = None
    ScalableAlgorithms = None
    FeatureExtraction = None
    LoadBalancing = None

# Context Personalization
try:
    from .engines.content_based.context_personalization import (
        CON_CONTEXT_AWARE,
        CON_USER_PROFILING,
        CON_ITEM_PROFILING,
    )
    # Short aliases
    ContextAware = CON_CONTEXT_AWARE
    UserProfiling = CON_USER_PROFILING
    ItemProfiling = CON_ITEM_PROFILING
except ImportError:
    CON_CONTEXT_AWARE = None
    CON_USER_PROFILING = None
    CON_ITEM_PROFILING = None
    ContextAware = None
    UserProfiling = None
    ItemProfiling = None

# Probabilistic Statistical Methods
try:
    from .engines.content_based.probabilistic_statistical_methods import PRO_LSA
    # Short alias
    LSA = PRO_LSA
except ImportError:
    PRO_LSA = None
    LSA = None

# Special Techniques
try:
    from .engines.content_based.special_techniques import (
        SPE_INTERACTIVE_FILTERING,
        SPE_DYNAMIC_FILTERING,
    )
    # Short aliases
    InteractiveFiltering = SPE_INTERACTIVE_FILTERING
    DynamicFiltering = SPE_DYNAMIC_FILTERING
except ImportError:
    SPE_INTERACTIVE_FILTERING = None
    SPE_DYNAMIC_FILTERING = None
    InteractiveFiltering = None
    DynamicFiltering = None

# Fairness and Explainability
try:
    from .engines.content_based.fairness_explainability import (
        FAI_EXPLAINABLE,
        FAI_FAIRNESS_AWARE,
        FAI_PRIVACY_PRESERVING,
    )
    # Short aliases
    Explainable = FAI_EXPLAINABLE
    FairnessAware = FAI_FAIRNESS_AWARE
    PrivacyPreserving = FAI_PRIVACY_PRESERVING
except ImportError:
    FAI_EXPLAINABLE = None
    FAI_FAIRNESS_AWARE = None
    FAI_PRIVACY_PRESERVING = None
    Explainable = None
    FairnessAware = None
    PrivacyPreserving = None


# ============================================================================
# __all__ - Controls what gets exported with "from corerec import *"
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Main modules (Engine-Level Organization)
    "engines",  # Access all recommendation engines
    "core",  # Core components (towers, encoders, losses)
    "training",  # Training utilities
    "trainer",  # Model trainer
    "data",  # Data loading and processing
    "utils",  # Utility functions
    "metrics",  # Evaluation metrics
    "evaluation",  # Evaluation tools
    "vish_graphs",  # Graph visualization
    "visualization",  # Visualization utilities
    # Modern RecSys components
    "pipelines",  # Multi-stage recommendation pipelines
    "retrieval",  # Vector stores and retrieval models
    "multimodal",  # Multi-modal fusion strategies
    "sandbox",  # Experimental methods under development
    # Base classes
    "BaseRecommender",
    "BaseCorerec",  # Deprecated, use BaseRecommender
    # Learning Paradigms (short names)
    "TransferLearning",
    "ZeroShot",
    "MetaLearning",
    # Multi-modal and Cross-domain (short names)
    "MultiModal",
    "CrossDomain",
    "CrossLingual",
    # Other Approaches (short names)
    "RuleBased",
    "SentimentAnalysis",
    "OntologyBased",
    # Miscellaneous Techniques (short names)
    "FeatureSelection",
    "NoiseHandling",
    "ColdStart",
    # CNN
    "CNN",
    # Performance Scalability
    "ScalableAlgorithms",
    "FeatureExtraction",
    "LoadBalancing",
    # Context Personalization
    "ContextAware",
    "UserProfiling",
    "ItemProfiling",
    # Probabilistic Statistical Methods
    "LSA",
    # Special Techniques
    "InteractiveFiltering",
    "DynamicFiltering",
    # Fairness and Explainability
    "Explainable",
    "FairnessAware",
    "PrivacyPreserving",
    # UPPERCASE aliases (for backward compatibility)
    "LEA_META_LEARNING",
    "MUL_MULTI_MODAL",
    "MUL_CROSS_DOMAIN",
    "MUL_CROSS_LINGUAL",
    "OTH_RULE_BASED",
    "OTH_SENTIMENT_ANALYSIS",
    "OTH_ONTOLOGY_BASED",
    "MIS_FEATURE_SELECTION",
    "MIS_NOISE_HANDLING",
    "MIS_COLD_START",
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
