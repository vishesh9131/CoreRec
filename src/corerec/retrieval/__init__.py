"""
Retrieval module for CoreRec framework.

This module contains retrievers for efficient candidate generation.
"""

from corerec.retrieval.base_retriever import BaseRetriever
from corerec.retrieval.dssm import DSSM
from corerec.retrieval.dense_encoder import DenseEncoderRetriever
from corerec.retrieval.contrastive_retriever import ContrastiveRetriever
from corerec.retrieval.faiss_index import FAISSIndexRetriever

__all__ = [
    'BaseRetriever',
    'DSSM',
    'DenseEncoderRetriever',
    'ContrastiveRetriever',
    'FAISSIndexRetriever'
] 