"""
Retrieval module for CoreRec framework.

This module contains retrievers for efficient candidate generation.
"""

from corerec.retrieval.base_retriever import BaseRetriever
from corerec.retrieval.dssm import DSSM

# Optional imports (may not exist yet)
try:
    from corerec.retrieval.dense_encoder import DenseEncoderRetriever
except ImportError:
    DenseEncoderRetriever = None

try:
    from corerec.retrieval.contrastive_retriever import ContrastiveRetriever
except ImportError:
    ContrastiveRetriever = None

try:
    from corerec.retrieval.faiss_index import FAISSIndexRetriever
except ImportError:
    FAISSIndexRetriever = None

# New vector store
try:
    from corerec.retrieval.vector_store import create_index, VectorIndex, NumpyIndex, FAISSIndex, AnnoyIndex
except ImportError:
    create_index = None
    VectorIndex = None
    NumpyIndex = None
    FAISSIndex = None
    AnnoyIndex = None

__all__ = [
    "BaseRetriever",
    "DSSM",
    "DenseEncoderRetriever",
    "ContrastiveRetriever",
    "FAISSIndexRetriever",
    "create_index",
    "VectorIndex",
    "NumpyIndex",
    "FAISSIndex",
    "AnnoyIndex",
]
