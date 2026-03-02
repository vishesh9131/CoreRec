"""
Retrieval Module - Candidate Generation (Stage 1)

This module provides retrievers that generate candidate items from
large catalogs efficiently. Retrievers prioritize recall over precision.

Usage:
    from corerec.retrieval import SemanticRetriever, CollaborativeRetriever
    
    retriever = SemanticRetriever(encoder="all-MiniLM-L6-v2")
    retriever.index(items)
    candidates = retriever.retrieve(query, top_k=100)
"""

from .base import BaseRetriever, Candidate, RetrievalResult
from .collaborative import CollaborativeRetriever
from .semantic import SemanticRetriever
from .popularity import PopularityRetriever
from .index import VectorIndex
from .ensemble import EnsembleRetriever

__all__ = [
    "BaseRetriever",
    "Candidate",
    "RetrievalResult",
    "CollaborativeRetriever",
    "SemanticRetriever",
    "PopularityRetriever",
    "VectorIndex",
    "EnsembleRetriever",
]
