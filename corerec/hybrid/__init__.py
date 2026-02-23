"""
Hybrid Recommendation Module

Provides hybrid approaches combining retrieval and reranking stages.
"""

try:
    from .retrieval_then_rerank import RetrievalThenRerank
except ImportError:
    RetrievalThenRerank = None

try:
    from .prompt_reranker import PromptReranker
except ImportError:
    PromptReranker = None

__all__ = ["RetrievalThenRerank", "PromptReranker"]
