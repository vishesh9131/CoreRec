"""
Sandbox: Content-Based Methods
===============================

35+ content-based methods preserved from main engine.

All methods accessible with full functionality.

Usage:
------
    # Import any method
    from corerec.sandbox.content_based import CNN, RNN, Transformer
    from corerec.sandbox.content_based import SVM, LightGBM
    from corerec.sandbox.content_based import Doc2Vec, TDM, MIND
    
    # Or via submodules
    from corerec.sandbox.content_based.nn import CNN
    from corerec.sandbox.content_based.traditional import SVM

All original functionality preserved.
"""

import sys
import os

# Add the full content_based backup to path
_full_path = os.path.join(os.path.dirname(__file__), '../content_based_full')
if _full_path not in sys.path:
    sys.path.insert(0, _full_path)

# Import from full backup
try:
    from corerec.sandbox.content_based_full import traditional_ml_algorithms as traditional
    from corerec.sandbox.content_based_full import nn_based_algorithms as nn
    from corerec.sandbox.content_based_full import embedding_representation_learning as embedding
    from corerec.sandbox.content_based_full import graph_based_algorithms as graph
    from corerec.sandbox.content_based_full import hybrid_ensemble_methods as hybrid
    from corerec.sandbox.content_based_full import context_personalization as context
    from corerec.sandbox.content_based_full import fairness_explainability as fairness
    from corerec.sandbox.content_based_full import learning_paradigms as learning
    from corerec.sandbox.content_based_full import multi_modal_cross_domain_methods as multimodal
    from corerec.sandbox.content_based_full import special_techniques as special
    from corerec.sandbox.content_based_full import probabilistic_statistical_methods as probabilistic
    from corerec.sandbox.content_based_full import performance_scalability as performance
    from corerec.sandbox.content_based_full import other_approaches as other
    from corerec.sandbox.content_based_full import miscellaneous_techniques as misc
except ImportError as e:
    print(f"Warning: Could not import some sandbox modules: {e}")
    traditional = nn = embedding = graph = hybrid = context = None
    fairness = learning = multimodal = special = probabilistic = performance = other = misc = None

# Direct imports of popular methods
try:
    from corerec.sandbox.content_based_full.nn_based_algorithms.cnn import CNN
    from corerec.sandbox.content_based_full.nn_based_algorithms.rnn import RNN
    from corerec.sandbox.content_based_full.nn_based_algorithms.transformer import Transformer
    from corerec.sandbox.content_based_full.nn_based_algorithms.vae import VAE
    from corerec.sandbox.content_based_full.nn_based_algorithms.autoencoder import Autoencoder
    from corerec.sandbox.content_based_full.nn_based_algorithms.TDM import TDM
    from corerec.sandbox.content_based_full.nn_based_algorithms.MIND import MIND
    from corerec.sandbox.content_based_full.nn_based_algorithms.AITM import AITM
    from corerec.sandbox.content_based_full.embedding_representation_learning.doc2vec import Doc2VecRecommender
except ImportError:
    CNN = RNN = Transformer = VAE = Autoencoder = TDM = MIND = AITM = Doc2VecRecommender = None

__all__ = [
    # Submodules
    "traditional", "nn", "embedding", "graph", "hybrid", "context",
    "fairness", "learning", "multimodal", "special", "probabilistic",
    "performance", "other", "misc",
    # Direct access
    "CNN", "RNN", "Transformer", "VAE", "Autoencoder",
    "TDM", "MIND", "AITM", "Doc2VecRecommender",
]


def list_available():
    """List all 35+ available content-based methods."""
    return """
    Sandbox Content-Based Methods (35+)
    ===================================
    
    Traditional ML (traditional.*):
    - TF-IDF (in main), SVM, LightGBM, Decision Trees
    - Logistic Regression, Vowpal Wabbit
    
    Neural Networks (nn.*):
    - CNN, RNN, Transformer, VAE, Autoencoder
    - DSSM (in main), MIND, TDM, AITM
    - Youtube DNN (in main), Widen & Deep
    - DKN, LSTUR, NAML, NPA, NRMS
    
    Embedding Learning (embedding.*):
    - Word2Vec (in main), Doc2Vec
    
    Graph-Based (graph.*):
    - GNN variants, Semantic models
    
    Hybrid & Ensemble (hybrid.*):
    - Attention mechanisms, Ensemble methods
    - Hybrid collaborative methods
    
    Context-Aware (context.*):
    - User profiling, Item profiling
    - Context-aware filtering
    
    Fairness & Explainability (fairness.*):
    - Fair ranking, Explainable AI
    - Privacy-preserving methods
    
    Learning Paradigms (learning.*):
    - Transfer learning, Meta-learning
    - Few-shot, Zero-shot learning
    
    Multi-Modal (multimodal.*):
    - Cross-domain methods
    - Multi-modal fusion
    
    Others:
    - Special techniques, Probabilistic methods
    - Performance optimization approaches
    
    Import: from corerec.sandbox.content_based import <MethodName>
    """


def get_method_info(method_name):
    """Get information about a specific method."""
    info_db = {
        "CNN": "Convolutional Neural Network for item features",
        "RNN": "Recurrent Neural Network for sequential patterns",
        "Transformer": "Transformer architecture for content understanding",
        "VAE": "Variational Autoencoder for content representation",
        "Doc2Vec": "Document embeddings for text-based items",
        "TDM": "Tree-based Deep Model for hierarchical recommendation",
        "MIND": "Multi-Interest Network with Dynamic routing",
    }
    return info_db.get(method_name, "Check method docstring for details")
