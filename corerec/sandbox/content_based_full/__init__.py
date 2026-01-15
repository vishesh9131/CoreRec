"""
Content-Based Filtering Engine (Refactored)
============================================

This engine provides the TOP 5 most useful content-based methods.
All other methods are available in the sandbox for development.

Top 5 Production-Ready Methods:
--------------------------------
1. TFIDFRecommender - Text-based, classic, reliable
2. YoutubeDNN - Deep neural network, industry-proven
3. DSSM - Deep semantic matching (Microsoft)
4. BERT4Rec - Sequential transformer-based
5. Word2VecRecommender - Embedding-based, versatile

Other 35+ algorithms moved to sandbox for refinement.

Usage:
------
    from corerec.engines import content
    
    # Text-based (classic)
    model = content.TFIDFRecommender()
    
    # Deep learning
    model = content.YoutubeDNN(embedding_dim=256)
    
    # Semantic matching
    model = content.DSSM(embedding_dim=128)
    
    # Sequential
    model = content.BERT4Rec(hidden_dim=256)
    
    # Embedding-based
    model = content.Word2VecRecommender(vector_size=100)

Sandbox Access:
---------------
    # For experimental methods
    from corerec.sandbox.content_based import Transformer, CNN, etc.

Author: Vishesh Yadav
"""

# ============================================================================
# TOP 5 PRODUCTION-READY METHODS
# ============================================================================

# 1. TFIDFRecommender (Classic, Reliable)
try:
    from .tfidf_recommender import TFIDFRecommender
except ImportError:
    TFIDFRecommender = None

# 2. YoutubeDNN (Industry Standard)
try:
    from .nn_based_algorithms.Youtube_dnn import YoutubeDNN
except ImportError:
    YoutubeDNN = None

# 3. DSSM (Deep Semantic Matching)
try:
    from .nn_based_algorithms.DSSM import DSSM
except ImportError:
    DSSM = None

# 4. BERT4Rec (Sequential Transformer)
try:
    from corerec.engines.bert4rec import BERT4Rec
except ImportError:
    BERT4Rec = None

# 5. Word2VecRecommender (Embedding-Based)
try:
    from .embedding_representation_learning.word2vec import Word2VecRecommender
except ImportError:
    Word2VecRecommender = None


# ============================================================================
# Backward Compatibility - Deprecated but Available
# ============================================================================

# Keep for backward compat
try:
    from .hybrid_ensemble_methods import AttentionMechanisms
except ImportError:
    AttentionMechanisms = None

try:
    from .hybrid_ensemble_methods import EnsembleRecommender
except ImportError:
    EnsembleRecommender = None


# ============================================================================
# Sandbox Access (Development/Experimental)
# ============================================================================

class SandboxAccess:
    """
    Gateway to experimental content-based methods.
    
    These are functional but under active development.
    Use for research/experimentation.
    """
    
    @staticmethod
    def list_available():
        """List all sandbox methods."""
        return [
            "Traditional ML: SVM, LightGBM, Decision Trees",
            "Neural Networks: CNN, RNN, Transformer, VAE, Autoencoder",
            "Graph-Based: GNN, Semantic Models",
            "Hybrid: Ensemble methods, Attention mechanisms",
            "Context-Aware: User profiling, Item profiling",
            "Fairness: Fair ranking, Explainable AI",
            "Learning: Transfer learning, Meta-learning, Few-shot",
            "Multi-Modal: Text+Image+Audio fusion",
            "Others: Doc2Vec, TDM, MIND, AITM",
            "",
            "Total: 35+ methods in sandbox",
            "Import from: corerec.sandbox.content_based"
        ]
    
    @staticmethod
    def get_info(method_name):
        """Get info about a sandbox method."""
        info_map = {
            "CNN": "Convolutional Neural Network for item features",
            "Transformer": "Transformer architecture for content understanding",
            "VAE": "Variational Autoencoder for content representation",
            "Doc2Vec": "Document embeddings for text-based items",
            "MIND": "Multi-Interest Network with Dynamic routing",
        }
        return info_map.get(method_name, "No info available. Check sandbox docs.")


sandbox = SandboxAccess()


# ============================================================================
# __all__ - Export List (ONLY Top 5)
# ============================================================================

__all__ = [
    # Top 5 Production Methods
    "TFIDFRecommender",     # Classic text-based
    "YoutubeDNN",           # Industry standard
    "DSSM",                 # Semantic matching
    "BERT4Rec",             # Sequential transformer
    "Word2VecRecommender",  # Embedding-based
    
    # Backward compat (deprecated)
    "AttentionMechanisms",
    "EnsembleRecommender",
    
    # Sandbox gateway
    "sandbox",
]


# ============================================================================
# Helper Functions
# ============================================================================

def list_methods():
    """List the top 5 production-ready methods."""
    methods = []
    
    if TFIDFRecommender is not None:
        methods.append("TFIDFRecommender - Text-based, classic")
    if YoutubeDNN is not None:
        methods.append("YoutubeDNN - Deep neural network")
    if DSSM is not None:
        methods.append("DSSM - Deep semantic matching")
    if BERT4Rec is not None:
        methods.append("BERT4Rec - Sequential transformer")
    if Word2VecRecommender is not None:
        methods.append("Word2VecRecommender - Embedding-based")
    
    return methods


def get_recommendation():
    """Get recommendation on which method to use."""
    return """
    Recommendation Guide:
    
    Use TFIDFRecommender if:
    - Text-based items (articles, products)
    - No deep learning needed
    - Fast baseline required
    
    Use YoutubeDNN if:
    - Large-scale deployment
    - Multi-stage pipeline
    - Rich item features
    
    Use DSSM if:
    - Semantic understanding needed
    - Query-document matching
    - Deep feature learning
    
    Use BERT4Rec if:
    - Sequential user behavior
    - Time-aware recommendations
    - Transformer architecture preferred
    
    Use Word2VecRecommender if:
    - Item-to-item similarity
    - Embedding-based approach
    - Medium-sized catalog
    
    For other methods, check sandbox.list_available()
    """


def migrate_to_sandbox_notice():
    """Information about the refactoring."""
    return """
    ⚠️  REFACTORING NOTICE
    
    35+ methods have been moved to sandbox for quality refinement:
    - All methods still accessible
    - Import from: corerec.sandbox.content_based
    - Top 5 methods remain in main engine
    - Sandbox methods will graduate when production-ready
    
    This ensures clean, battle-tested main engine.
    """
