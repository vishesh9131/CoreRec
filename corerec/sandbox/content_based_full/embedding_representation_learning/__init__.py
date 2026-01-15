"""
Embedding & Representation Learning
===================================

Algorithms for learning feature embeddings and representations.

This module provides:
- Word2Vec for text embeddings
- Doc2Vec for document embeddings
- Personalized embeddings

Usage:
------
    from corerec.engines.content import embedding
    
    # Embedding models
    model = embedding.Word2VecRecommender()
    model = embedding.Doc2VecRecommender()
    model = embedding.PersonalizedEmbeddings()

Author: Vishesh Yadav (sciencely98@gmail.com)
"""

# ============================================================================
# Export Embedding Learning Algorithms
# ============================================================================

try:
    from .word2vec import Word2VecRecommender
except ImportError:
    Word2VecRecommender = None

try:
    from .doc2vec import Doc2VecRecommender
except ImportError:
    Doc2VecRecommender = None

try:
    from .personalized_embeddings import PersonalizedEmbeddings
except ImportError:
    PersonalizedEmbeddings = None

# ============================================================================
# __all__ - Export list
# ============================================================================

__all__ = [
    "Word2VecRecommender",
    "Doc2VecRecommender",
    "PersonalizedEmbeddings",
]
