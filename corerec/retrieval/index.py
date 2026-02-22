"""
Vector Index for Approximate Nearest Neighbor Search

Wraps FAISS or other ANN libraries for efficient similarity search
on large embedding collections.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import BaseRetriever, Candidate, RetrievalResult


class VectorIndex(BaseRetriever):
    """
    High-performance vector similarity search using ANN algorithms.
    
    Wraps FAISS (or falls back to brute-force numpy) for scalable
    nearest neighbor search on embeddings.
    
    For catalogs with >100K items, this is essential for sub-millisecond
    retrieval. Smaller catalogs can use brute-force without issues.
    
    Example:
        index = VectorIndex(index_type="flat", metric="cosine")
        index.fit(item_ids, embeddings)
        
        # single query
        results = index.retrieve(query_embedding, top_k=100)
        
        # batch
        results = index.retrieve_batch(query_embeddings, top_k=100)
    """
    
    def __init__(
        self,
        index_type: str = "flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        name: str = "vector_index",
    ):
        """
        Args:
            index_type: 'flat' (exact), 'ivf' (approximate), or 'hnsw'
            metric: 'cosine', 'l2', or 'ip' (inner product)
            nlist: number of clusters for IVF index
            nprobe: number of clusters to search for IVF
            name: identifier for this retriever
        """
        super().__init__(name=name)
        
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        
        self._faiss_available = self._check_faiss()
        self._index = None
        self.item_ids: List[Any] = []
        self.embedding_dim: int = 0
    
    def _check_faiss(self) -> bool:
        """Check if FAISS is available."""
        try:
            import faiss
            return True
        except ImportError:
            return False
    
    def fit(
        self,
        item_ids: List[Any],
        embeddings: np.ndarray,
        **kwargs
    ) -> "VectorIndex":
        """
        Build the vector index.
        
        Args:
            item_ids: unique identifiers for each embedding
            embeddings: array of shape (n_items, dim)
        """
        self.item_ids = list(item_ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embedding_dim = embeddings.shape[1]
        
        # normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms
        
        if self._faiss_available:
            self._build_faiss_index(embeddings)
        else:
            # fallback: store embeddings for brute-force search
            self._embeddings = embeddings
        
        self._is_fitted = True
        return self
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index."""
        import faiss
        
        d = embeddings.shape[1]
        n = embeddings.shape[0]
        
        if self.index_type == "flat":
            if self.metric in ["cosine", "ip"]:
                self._index = faiss.IndexFlatIP(d)
            else:
                self._index = faiss.IndexFlatL2(d)
        
        elif self.index_type == "ivf":
            # IVF requires training
            nlist = min(self.nlist, n // 10)  # ensure enough points per cluster
            
            if self.metric in ["cosine", "ip"]:
                quantizer = faiss.IndexFlatIP(d)
                self._index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(d)
                self._index = faiss.IndexIVFFlat(quantizer, d, nlist)
            
            self._index.train(embeddings)
            self._index.nprobe = self.nprobe
        
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(d, 32)
        
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        
        self._index.add(embeddings)
    
    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 100,
        **kwargs
    ) -> RetrievalResult:
        """
        Search for nearest neighbors.
        
        Args:
            query: query embedding vector
            top_k: number of neighbors to return
        """
        self._check_fitted()
        
        start = time.perf_counter()
        
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        
        # normalize for cosine
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
        
        if self._faiss_available and self._index is not None:
            distances, indices = self._index.search(query, top_k)
            distances = distances[0]
            indices = indices[0]
        else:
            # brute force
            if self.metric in ["cosine", "ip"]:
                scores = self._embeddings @ query.T
                scores = scores.flatten()
                indices = np.argsort(scores)[::-1][:top_k]
                distances = scores[indices]
            else:
                # L2
                diff = self._embeddings - query
                dists = np.sum(diff ** 2, axis=1)
                indices = np.argsort(dists)[:top_k]
                distances = -dists[indices]  # negate so higher = better
        
        candidates = []
        for idx, dist in zip(indices, distances):
            if idx < 0:  # FAISS returns -1 for not enough results
                continue
            candidates.append(Candidate(
                item_id=self.item_ids[idx],
                score=float(dist),
                source=self.name,
            ))
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return RetrievalResult(
            candidates=candidates,
            query_id="embedding",
            retriever_name=self.name,
            timing_ms=elapsed,
        )
    
    def retrieve_batch(
        self,
        queries: List[np.ndarray],
        top_k: int = 100,
        **kwargs
    ) -> List[RetrievalResult]:
        """Batch search - more efficient than individual queries."""
        self._check_fitted()
        
        start = time.perf_counter()
        
        query_matrix = np.stack([np.asarray(q) for q in queries]).astype(np.float32)
        
        if self.metric == "cosine":
            norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            query_matrix = query_matrix / norms
        
        if self._faiss_available and self._index is not None:
            all_distances, all_indices = self._index.search(query_matrix, top_k)
        else:
            # brute force batch
            if self.metric in ["cosine", "ip"]:
                scores = query_matrix @ self._embeddings.T
                all_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
                all_distances = np.take_along_axis(scores, all_indices, axis=1)
            else:
                # L2 batch
                all_indices = []
                all_distances = []
                for q in query_matrix:
                    diff = self._embeddings - q
                    dists = np.sum(diff ** 2, axis=1)
                    idx = np.argsort(dists)[:top_k]
                    all_indices.append(idx)
                    all_distances.append(-dists[idx])
                all_indices = np.array(all_indices)
                all_distances = np.array(all_distances)
        
        elapsed_per = (time.perf_counter() - start) * 1000 / len(queries)
        
        results = []
        for i in range(len(queries)):
            candidates = []
            for idx, dist in zip(all_indices[i], all_distances[i]):
                if idx < 0:
                    continue
                candidates.append(Candidate(
                    item_id=self.item_ids[idx],
                    score=float(dist),
                    source=self.name,
                ))
            results.append(RetrievalResult(
                candidates=candidates,
                query_id=f"query_{i}",
                retriever_name=self.name,
                timing_ms=elapsed_per,
            ))
        
        return results
