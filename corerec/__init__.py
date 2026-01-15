"""
CoreRec: Advanced Recommendation Systems Library
=================================================

CoreRec provides state-of-the-art recommendation algorithms with a unified API.

Quick Start:
-----------
    from corerec.engines.collaborative import SAR
    
    model = SAR(similarity_type='jaccard')
    model.fit(train_df)
    recs = model.recommend_k_items(test_df, top_k=10)

For deep learning models:
    from corerec import engines
    model = engines.DCN(embedding_dim=64)

Author: Vishesh Yadav (sciencely98@gmail.com)
License: Research purposes only
"""

__version__ = "0.5.1"
__author__ = "Vishesh Yadav"
__email__ = "sciencely98@gmail.com"


# ============================================================================
# LAZY IMPORTS - modules loaded only when accessed
# ============================================================================
# This prevents loading heavy deps (sklearn, torch, matplotlib) on every import

def __getattr__(name):
    """Lazy import handler - loads modules on first access."""
    
    # submodules that should be importable
    _submodules = {
        "engines",
        "core", 
        "utils",
        "metrics",
        "evaluation",
        "vish_graphs",
        "visualization",
        "data",
        "training",
        "trainer",
        "pipelines",
        "retrieval",
        "multimodal",
        "sandbox",
        "api",
        "models",
    }
    
    if name in _submodules:
        import importlib
        try:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
            return module
        except ImportError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    
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
    
    if name == "BaseCorerec":
        from .base_recommender import BaseCorerec
        globals()["BaseCorerec"] = BaseCorerec
        return BaseCorerec
    
    # model aliases - only load when explicitly requested
    _model_aliases = {
        # learning paradigms
        "LEA_TRANSFER_LEARNING": (".engines.content_based.learning_paradigms", "LEA_TRANSFER_LEARNING"),
        "LEA_ZERO_SHOT": (".engines.content_based.learning_paradigms", "LEA_ZERO_SHOT"),
        "LEA_META_LEARNING": (".engines.content_based.learning_paradigms", "LEA_META_LEARNING"),
        "TransferLearning": (".engines.content_based.learning_paradigms", "LEA_TRANSFER_LEARNING"),
        "ZeroShot": (".engines.content_based.learning_paradigms", "LEA_ZERO_SHOT"),
        "MetaLearning": (".engines.content_based.learning_paradigms", "LEA_META_LEARNING"),
        # multimodal
        "MUL_MULTI_MODAL": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_MULTI_MODAL"),
        "MUL_CROSS_DOMAIN": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_CROSS_DOMAIN"),
        "MUL_CROSS_LINGUAL": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_CROSS_LINGUAL"),
        "MultiModal": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_MULTI_MODAL"),
        "CrossDomain": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_CROSS_DOMAIN"),
        "CrossLingual": (".engines.content_based.multi_modal_cross_domain_methods", "MUL_CROSS_LINGUAL"),
        # other
        "OTH_RULE_BASED": (".engines.content_based.other_approaches", "OTH_RULE_BASED"),
        "OTH_SENTIMENT_ANALYSIS": (".engines.content_based.other_approaches", "OTH_SENTIMENT_ANALYSIS"),
        "OTH_ONTOLOGY_BASED": (".engines.content_based.other_approaches", "OTH_ONTOLOGY_BASED"),
        "RuleBased": (".engines.content_based.other_approaches", "OTH_RULE_BASED"),
        "SentimentAnalysis": (".engines.content_based.other_approaches", "OTH_SENTIMENT_ANALYSIS"),
        "OntologyBased": (".engines.content_based.other_approaches", "OTH_ONTOLOGY_BASED"),
        # misc
        "MIS_FEATURE_SELECTION": (".engines.content_based.miscellaneous_techniques", "MIS_FEATURE_SELECTION"),
        "MIS_NOISE_HANDLING": (".engines.content_based.miscellaneous_techniques", "MIS_NOISE_HANDLING"),
        "MIS_COLD_START": (".engines.content_based.miscellaneous_techniques", "MIS_COLD_START"),
        "FeatureSelection": (".engines.content_based.miscellaneous_techniques", "MIS_FEATURE_SELECTION"),
        "NoiseHandling": (".engines.content_based.miscellaneous_techniques", "MIS_NOISE_HANDLING"),
        "ColdStart": (".engines.content_based.miscellaneous_techniques", "MIS_COLD_START"),
        # cnn
        "CNN": (".cnn", "CNN"),
        # performance
        "PER_SCALABLE_ALGORITHMS": (".engines.content_based.performance_scalability", "PER_SCALABLE_ALGORITHMS"),
        "PER_FEATURE_EXTRACTION": (".engines.content_based.performance_scalability", "PER_FEATURE_EXTRACTION"),
        "PER_LOAD_BALANCING": (".engines.content_based.performance_scalability", "PER_LOAD_BALANCING"),
        "ScalableAlgorithms": (".engines.content_based.performance_scalability", "PER_SCALABLE_ALGORITHMS"),
        "FeatureExtraction": (".engines.content_based.performance_scalability", "PER_FEATURE_EXTRACTION"),
        "LoadBalancing": (".engines.content_based.performance_scalability", "PER_LOAD_BALANCING"),
        # context
        "CON_CONTEXT_AWARE": (".engines.content_based.context_personalization", "CON_CONTEXT_AWARE"),
        "CON_USER_PROFILING": (".engines.content_based.context_personalization", "CON_USER_PROFILING"),
        "CON_ITEM_PROFILING": (".engines.content_based.context_personalization", "CON_ITEM_PROFILING"),
        "ContextAware": (".engines.content_based.context_personalization", "CON_CONTEXT_AWARE"),
        "UserProfiling": (".engines.content_based.context_personalization", "CON_USER_PROFILING"),
        "ItemProfiling": (".engines.content_based.context_personalization", "CON_ITEM_PROFILING"),
        # probabilistic
        "PRO_LSA": (".engines.content_based.probabilistic_statistical_methods", "PRO_LSA"),
        "LSA": (".engines.content_based.probabilistic_statistical_methods", "PRO_LSA"),
        # special
        "SPE_INTERACTIVE_FILTERING": (".engines.content_based.special_techniques", "SPE_INTERACTIVE_FILTERING"),
        "SPE_DYNAMIC_FILTERING": (".engines.content_based.special_techniques", "SPE_DYNAMIC_FILTERING"),
        "InteractiveFiltering": (".engines.content_based.special_techniques", "SPE_INTERACTIVE_FILTERING"),
        "DynamicFiltering": (".engines.content_based.special_techniques", "SPE_DYNAMIC_FILTERING"),
        # fairness
        "FAI_EXPLAINABLE": (".engines.content_based.fairness_explainability", "FAI_EXPLAINABLE"),
        "FAI_FAIRNESS_AWARE": (".engines.content_based.fairness_explainability", "FAI_FAIRNESS_AWARE"),
        "FAI_PRIVACY_PRESERVING": (".engines.content_based.fairness_explainability", "FAI_PRIVACY_PRESERVING"),
        "Explainable": (".engines.content_based.fairness_explainability", "FAI_EXPLAINABLE"),
        "FairnessAware": (".engines.content_based.fairness_explainability", "FAI_FAIRNESS_AWARE"),
        "PrivacyPreserving": (".engines.content_based.fairness_explainability", "FAI_PRIVACY_PRESERVING"),
    }
    
    if name in _model_aliases:
        import importlib
        mod_path, attr_name = _model_aliases[name]
        try:
            mod = importlib.import_module(mod_path, __name__)
            val = getattr(mod, attr_name)
            globals()[name] = val
            return val
        except (ImportError, AttributeError):
            globals()[name] = None
            return None
    
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
    # Main modules (lazy loaded)
    "engines",
    "core",
    "training",
    "trainer",
    "data",
    "utils",
    "metrics",
    "evaluation",
    "vish_graphs",
    "visualization",
    "pipelines",
    "retrieval",
    "multimodal",
    "sandbox",
    # Base classes
    "BaseRecommender",
    "BaseCorerec",
    # Learning Paradigms
    "TransferLearning",
    "ZeroShot",
    "MetaLearning",
    # Multi-modal and Cross-domain
    "MultiModal",
    "CrossDomain",
    "CrossLingual",
    # Other Approaches
    "RuleBased",
    "SentimentAnalysis",
    "OntologyBased",
    # Miscellaneous
    "FeatureSelection",
    "NoiseHandling",
    "ColdStart",
    # CNN
    "CNN",
    # Performance
    "ScalableAlgorithms",
    "FeatureExtraction",
    "LoadBalancing",
    # Context
    "ContextAware",
    "UserProfiling",
    "ItemProfiling",
    # Probabilistic
    "LSA",
    # Special
    "InteractiveFiltering",
    "DynamicFiltering",
    # Fairness
    "Explainable",
    "FairnessAware",
    "PrivacyPreserving",
    # UPPERCASE aliases
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
