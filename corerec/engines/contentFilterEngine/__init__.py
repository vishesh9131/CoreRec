"""
Content Filter Engine
=====================

Content-based recommendation algorithms organized by category.

This engine provides 40+ content-based filtering algorithms:
- Traditional ML (TF-IDF, SVM, LightGBM, Decision Trees)
- Neural Networks (DSSM, MIND, TDM, YouTube DNN, Transformers)
- Graph-Based (GNN, Semantic Models, Graph Filtering)
- Embedding Learning (Word2Vec, Doc2Vec)
- Hybrid & Ensemble (Attention, Ensemble Methods)
- Fairness & Explainability (Fair Ranking, Explainable AI)
- Learning Paradigms (Transfer, Meta, Few-shot, Zero-shot)

Usage:
------
    from corerec.engines import content
    
    # Popular algorithms - direct access
    model = content.TFIDFRecommender()
    model = content.AttentionMechanisms()
    model = content.EnsembleRecommender()
    
    # By category
    model = content.nn.DSSM()
    model = content.nn.YoutubeDNN()
    model = content.embedding.Word2VecRecommender()
    model = content.traditional.SVMRecommender()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Most Popular Algorithms - Direct access
# ============================================================================

# TF-IDF (most commonly used)
try:
    from .tfidf_recommender import TFIDFRecommender
except ImportError:
    TFIDFRecommender = None

# Hybrid & Ensemble Methods (very popular)
try:
    from .hybrid_ensemble_methods import AttentionMechanisms
except ImportError:
    AttentionMechanisms = None

try:
    from .hybrid_ensemble_methods import EnsembleRecommender
except ImportError:
    EnsembleRecommender = None

try:
    from .hybrid_ensemble_methods import HybridCollaborative
except ImportError:
    HybridCollaborative = None

# Neural Network Based (popular)
try:
    from .nn_based_algorithms.DSSM import DSSM
except ImportError:
    DSSM = None

try:
    from .nn_based_algorithms.MIND import MIND as ContentMIND
except ImportError:
    ContentMIND = None

try:
    from .nn_based_algorithms.Youtube_dnn import YoutubeDNN
except ImportError:
    YoutubeDNN = None

# Embedding Learning (popular)
try:
    from .embedding_representation_learning.word2vec import Word2VecRecommender
except ImportError:
    Word2VecRecommender = None

try:
    from .embedding_representation_learning.doc2vec import Doc2VecRecommender
except ImportError:
    Doc2VecRecommender = None

# ============================================================================
# Organized Sub-modules by Algorithm Category
# ============================================================================

# Traditional ML algorithms
from . import traditional_ml_algorithms as traditional

# Neural Network based algorithms
from . import nn_based_algorithms as nn

# Context and Personalization
from . import context_personalization as context

# Embedding and Representation Learning
from . import embedding_representation_learning as embedding

# Fairness and Explainability
from . import fairness_explainability as fairness

# Graph-based algorithms
from . import graph_based_algorithms as graph

# Hybrid and Ensemble methods
from . import hybrid_ensemble_methods as hybrid

# Learning Paradigms
from . import learning_paradigms as learning

# Miscellaneous techniques
from . import miscellaneous_techniques as misc

# Multi-modal and Cross-domain methods
from . import multi_modal_cross_domain_methods as multimodal

# Other approaches
from . import other_approaches as other

# Performance and Scalability
from . import performance_scalability as performance

# Probabilistic and Statistical methods
from . import probabilistic_statistical_methods as probabilistic

# Special techniques
from . import special_techniques as special

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    # Popular algorithms (direct access)
    "TFIDFRecommender",
    "AttentionMechanisms",
    "EnsembleRecommender",
    "HybridCollaborative",
    "DSSM",
    "ContentMIND",
    "YoutubeDNN",
    "Word2VecRecommender",
    "Doc2VecRecommender",
    # Organized sub-modules
    "traditional",  # Traditional ML (TF-IDF, SVM, etc.)
    "nn",  # Neural Networks (DSSM, MIND, etc.)
    "context",  # Context & Personalization
    "embedding",  # Embedding Learning (Word2Vec, etc.)
    "fairness",  # Fairness & Explainability
    "graph",  # Graph-based methods
    "hybrid",  # Hybrid & Ensemble methods
    "learning",  # Learning Paradigms
    "misc",  # Miscellaneous techniques
    "multimodal",  # Multi-modal methods
    "other",  # Other approaches
    "performance",  # Performance & Scalability
    "probabilistic",  # Probabilistic methods
    "special",  # Special techniques
]

# ============================================================================
# Helper Functions
# ============================================================================


def list_algorithms():
    """List all available algorithms with direct access."""
    algorithms = []

    if TFIDFRecommender is not None:
        algorithms.append("TFIDFRecommender")
    if AttentionMechanisms is not None:
        algorithms.append("AttentionMechanisms")
    if EnsembleRecommender is not None:
        algorithms.append("EnsembleRecommender")
    if HybridCollaborative is not None:
        algorithms.append("HybridCollaborative")
    if DSSM is not None:
        algorithms.append("DSSM")
    if ContentMIND is not None:
        algorithms.append("ContentMIND")
    if YoutubeDNN is not None:
        algorithms.append("YoutubeDNN")
    if Word2VecRecommender is not None:
        algorithms.append("Word2VecRecommender")
    if Doc2VecRecommender is not None:
        algorithms.append("Doc2VecRecommender")

    return algorithms


def list_categories():
    """List all algorithm categories."""
    return [
        "traditional (Traditional ML)",
        "nn (Neural Networks)",
        "context (Context & Personalization)",
        "embedding (Embedding Learning)",
        "fairness (Fairness & Explainability)",
        "graph (Graph-Based)",
        "hybrid (Hybrid & Ensemble)",
        "learning (Learning Paradigms)",
        "multimodal (Multi-Modal)",
        "probabilistic (Probabilistic Methods)",
        "special (Special Techniques)",
    ]
