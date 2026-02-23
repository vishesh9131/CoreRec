"""
Semantic Retrieval using Text Embeddings

Uses sentence transformers or similar encoders to embed items
and queries into a shared vector space, then retrieves via ANN search.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from .base import BaseRetriever, Candidate, RetrievalResult


class SemanticRetriever(BaseRetriever):
    """
    Retriever based on semantic similarity of text embeddings.
    
    Encodes item descriptions (or any text) into dense vectors,
    builds an index, and retrieves via approximate nearest neighbor search.
    
    Example::

        retriever = SemanticRetriever(
            encoder="sentence-transformers/all-MiniLM-L6-v2"
        )
        retriever.fit(
            item_ids=[1, 2, 3],
            item_texts=["red shoes", "blue jacket", "green hat"]
        )
        
        # text query
        results = retriever.retrieve("comfortable footwear", top_k=10)
        
        # or embed a user profile as query
        user_embedding = retriever.encode("likes outdoor hiking gear")
        results = retriever.retrieve(user_embedding, top_k=10)
    """
    
    def __init__(
        self,
        encoder: Optional[Union[str, Any]] = None,
        embedding_dim: int = 384,
        similarity: str = "cosine",
        name: str = "semantic",
    ):
        """
        Args:
            encoder: sentence transformer model name or encoder instance.
                     If string, will try to load from sentence-transformers.
                     If None, you must provide embeddings directly.
            embedding_dim: dimension of embeddings (auto-detected if encoder provided)
            similarity: 'cosine' or 'dot' for similarity computation
            name: identifier for this retriever
        """
        super().__init__(name=name)
        
        self.encoder_name = encoder if isinstance(encoder, str) else None
        self._encoder = None if isinstance(encoder, str) else encoder
        self.embedding_dim = embedding_dim
        self.similarity = similarity
        
        # populated by fit()
        self.item_ids: List[Any] = []
        self.item_embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[Any, int] = {}
    
    @property
    def encoder(self):
        """Lazy load encoder on first access."""
        if self._encoder is None and self.encoder_name is not None:
            self._encoder = self._load_encoder(self.encoder_name)
        return self._encoder
    
    def _load_encoder(self, model_name: str):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            self.embedding_dim = model.get_sentence_embedding_dimension()
            return model
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: single string or list of strings
        
        Returns:
            embeddings array of shape (n, dim) or (dim,) for single text
        """
        if self.encoder is None:
            raise ValueError("No encoder available. Provide encoder in __init__.")
        
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        
        if single:
            return embeddings[0]
        return embeddings
    
    def fit(
        self,
        item_ids: List[Any],
        item_texts: Optional[List[str]] = None,
        item_embeddings: Optional[np.ndarray] = None,
        **kwargs
    ) -> "SemanticRetriever":
        """
        Index items for retrieval.
        
        Args:
            item_ids: unique identifiers for each item
            item_texts: text descriptions to encode (if not providing embeddings)
            item_embeddings: pre-computed embeddings (if not providing texts)
        
        You must provide either item_texts or item_embeddings.
        """
        if item_texts is None and item_embeddings is None:
            raise ValueError("Provide either item_texts or item_embeddings")
        
        self.item_ids = list(item_ids)
        self._id_to_idx = {id_: i for i, id_ in enumerate(self.item_ids)}
        
        if item_embeddings is not None:
            self.item_embeddings = np.asarray(item_embeddings)
            self.embedding_dim = self.item_embeddings.shape[1]
        else:
            # encode texts
            self.item_embeddings = self.encode(item_texts)
        
        # normalize for cosine similarity
        if self.similarity == "cosine":
            norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid div by 0
            self.item_embeddings = self.item_embeddings / norms
        
        self._is_fitted = True
        return self
    
    def retrieve(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 100,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve items similar to query.
        
        Args:
            query: text string or embedding vector
            top_k: number of candidates to return
        
        Returns:
            RetrievalResult with candidates sorted by similarity
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        # encode query if text
        if isinstance(query, str):
            query_emb = self.encode(query)
        else:
            query_emb = np.asarray(query)
        
        # normalize query for cosine
        if self.similarity == "cosine":
            norm = np.linalg.norm(query_emb)
            if norm > 0:
                query_emb = query_emb / norm
        
        # compute similarities (dot product after normalization = cosine)
        scores = self.item_embeddings @ query_emb
        
        # get top k
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # partial sort is faster for large catalogs
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        candidates = []
        for idx in top_indices[:top_k]:
            candidates.append(Candidate(
                item_id=self.item_ids[idx],
                score=float(scores[idx]),
                source=self.name,
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RetrievalResult(
            candidates=candidates,
            query_id=query if isinstance(query, str) else "embedding",
            retriever_name=self.name,
            timing_ms=elapsed,
        )
    
    def retrieve_batch(
        self,
        queries: List[Union[str, np.ndarray]],
        top_k: int = 100,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Batch retrieval - more efficient for multiple queries.
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        # encode all text queries at once
        text_queries = [q for q in queries if isinstance(q, str)]
        if text_queries:
            text_embeddings = self.encode(text_queries)
            text_emb_map = dict(zip(text_queries, text_embeddings))
        else:
            text_emb_map = {}
        
        # build query matrix
        query_embs = []
        for q in queries:
            if isinstance(q, str):
                query_embs.append(text_emb_map[q])
            else:
                query_embs.append(np.asarray(q))
        
        query_matrix = np.stack(query_embs)
        
        # normalize
        if self.similarity == "cosine":
            norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            query_matrix = query_matrix / norms
        
        # batch dot product
        all_scores = query_matrix @ self.item_embeddings.T
        
        elapsed_per = (time.perf_counter() - start) * 1000 / len(queries)
        
        results = []
        for i, q in enumerate(queries):
            scores = all_scores[i]
            
            if top_k >= len(scores):
                top_indices = np.argsort(scores)[::-1][:top_k]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            
            candidates = []
            for idx in top_indices:
                candidates.append(Candidate(
                    item_id=self.item_ids[idx],
                    score=float(scores[idx]),
                    source=self.name,
                ))
            
            results.append(RetrievalResult(
                candidates=candidates,
                query_id=q if isinstance(q, str) else f"query_{i}",
                retriever_name=self.name,
                timing_ms=elapsed_per,
            ))
        
        return results
