"""
Hybrid Recommendation Module

Provides hybrid approaches combining retrieval and reranking stages.
"""

# Imported directly: a broken import here used to be swallowed into
# RetrievalThenRerank = None, which surfaced later as "NoneType is not callable".
from .retrieval_then_rerank import RetrievalThenRerank

try:
    from .prompt_reranker import PromptReranker
except ImportError as _err:
    # requests/aiohttp are not core deps; fail loudly only when it's actually used
    _prompt_err = _err

    class PromptReranker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(f"PromptReranker needs requests and aiohttp: {_prompt_err}")

__all__ = ["RetrievalThenRerank", "PromptReranker"]
