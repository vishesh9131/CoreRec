"""
Vector store abstraction for fast similarity search.

Supports multiple backends: FAISS, Annoy, simple numpy.
Essential for retrieval stage in production recsys.
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging


class VectorIndex(ABC):
    """Base class for vector search indices."""
    
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric  # cosine, euclidean, dot
        self.log = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """Add vectors to the index."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Returns: (distances, ids)
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load index from disk."""
        pass


class NumpyIndex(VectorIndex):
    """
    Simple brute-force search using numpy.
    Good for small datasets (<100k items) or as fallback.
    """
    
    def __init__(self, dim: int, metric: str = "cosine"):
        super().__init__(dim, metric)
        self.vectors = None
        self.ids = None
    
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """Store vectors in memory."""
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        
        self.vectors = vectors.astype(np.float32)
        
        if ids is None:
            self.ids = np.arange(len(vectors))
        else:
            self.ids = np.array(ids)
        
        # normalize for cosine similarity
        if self.metric == "cosine":
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # avoid div by zero
            self.vectors = self.vectors / norms
        
        self.log.info(f"Added {len(vectors)} vectors to index")
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Brute force search."""
        if self.vectors is None:
            raise ValueError("Index is empty, call add() first")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # normalize query for cosine
        if self.metric == "cosine":
            query_norm = np.linalg.norm(query, axis=1, keepdims=True)
            query_norm = np.where(query_norm == 0, 1, query_norm)
            query = query / query_norm
        
        # compute similarities
        if self.metric == "cosine" or self.metric == "dot":
            scores = np.dot(query, self.vectors.T)[0]
        elif self.metric == "euclidean":
            scores = -np.linalg.norm(self.vectors - query, axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # get top k
        if k > len(scores):
            k = len(scores)
        
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        
        top_scores = scores[top_indices]
        top_ids = self.ids[top_indices]
        
        return top_scores, top_ids
    
    def save(self, path: str):
        """Save to npz file."""
        np.savez(path, vectors=self.vectors, ids=self.ids, 
                 dim=self.dim, metric=self.metric)
        self.log.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """Load from npz file."""
        data = np.load(path, allow_pickle=True)
        self.vectors = data['vectors']
        self.ids = data['ids']
        self.dim = int(data['dim'])
        self.metric = str(data['metric'])
        self.log.info(f"Loaded index from {path}")


class FAISSIndex(VectorIndex):
    """
    FAISS-based index for large-scale ANN search.
    Much faster than brute force, handles millions of vectors.
    """
    
    def __init__(self, dim: int, metric: str = "cosine", index_type: str = "flat"):
        super().__init__(dim, metric)
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        self.index_type = index_type
        self.index = None
        self.ids = None
        self._build_index()
    
    def _build_index(self):
        """Initialize FAISS index."""
        if self.index_type == "flat":
            # exact search
            if self.metric == "cosine":
                self.index = self.faiss.IndexFlatIP(self.dim)  # inner product
            else:
                self.index = self.faiss.IndexFlatL2(self.dim)
        
        elif self.index_type == "ivf":
            # approximate search with clustering
            quantizer = self.faiss.IndexFlatL2(self.dim)
            self.index = self.faiss.IndexIVFFlat(quantizer, self.dim, 100)
        
        elif self.index_type == "hnsw":
            # graph-based ANN (good quality/speed tradeoff)
            self.index = self.faiss.IndexHNSWFlat(self.dim, 32)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        self.log.info(f"Created FAISS index: {self.index_type}")
    
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """Add vectors to FAISS index."""
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        
        vectors = vectors.astype(np.float32)
        
        # normalize for cosine similarity
        if self.metric == "cosine":
            self.faiss.normalize_L2(vectors)
        
        # train if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.log.info("Training IVF index...")
            self.index.train(vectors)
        
        self.index.add(vectors)
        
        if ids is None:
            self.ids = np.arange(len(vectors))
        else:
            self.ids = np.array(ids)
        
        self.log.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS."""
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty")
        
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        
        if self.metric == "cosine":
            self.faiss.normalize_L2(query)
        
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(query, k)
        
        # map internal indices to our IDs
        result_ids = self.ids[indices[0]]
        result_scores = distances[0]
        
        return result_scores, result_ids
    
    def save(self, path: str):
        """Save FAISS index."""
        self.faiss.write_index(self.index, path)
        np.savez(path + "_meta.npz", ids=self.ids, dim=self.dim, 
                 metric=self.metric, index_type=self.index_type)
        self.log.info(f"Saved FAISS index to {path}")
    
    def load(self, path: str):
        """Load FAISS index."""
        self.index = self.faiss.read_index(path)
        meta = np.load(path + "_meta.npz", allow_pickle=True)
        self.ids = meta['ids']
        self.dim = int(meta['dim'])
        self.metric = str(meta['metric'])
        self.index_type = str(meta['index_type'])
        self.log.info(f"Loaded FAISS index from {path}")


class AnnoyIndex(VectorIndex):
    """
    Annoy index - tree-based ANN search.
    Smaller memory footprint than FAISS, good for medium-scale.
    """
    
    def __init__(self, dim: int, metric: str = "cosine"):
        super().__init__(dim, metric)
        
        try:
            from annoy import AnnoyIndex as _AnnoyIndex
            self.AnnoyIndex = _AnnoyIndex
        except ImportError:
            raise ImportError("Annoy not installed. Run: pip install annoy")
        
        # map metric names
        metric_map = {
            "cosine": "angular",
            "euclidean": "euclidean",
            "dot": "dot"
        }
        
        annoy_metric = metric_map.get(metric, "angular")
        self.index = self.AnnoyIndex(dim, annoy_metric)
        self.ids = []
        self.built = False
    
    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """Add vectors to Annoy index."""
        if self.built:
            raise ValueError("Cannot add to built index, create new one")
        
        if ids is None:
            ids = list(range(len(self.ids), len(self.ids) + len(vectors)))
        
        for i, vec in enumerate(vectors):
            self.index.add_item(len(self.ids), vec)
            self.ids.append(ids[i])
        
        self.log.info(f"Added {len(vectors)} vectors")
    
    def build(self, n_trees: int = 10):
        """Build index (must call before search)."""
        self.log.info(f"Building Annoy index with {n_trees} trees...")
        self.index.build(n_trees)
        self.built = True
        self.log.info("Index built")
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search using Annoy."""
        if not self.built:
            raise ValueError("Index not built, call build() first")
        
        if query.ndim > 1:
            query = query.flatten()
        
        indices, distances = self.index.get_nns_by_vector(
            query, k, include_distances=True
        )
        
        result_ids = np.array([self.ids[i] for i in indices])
        result_scores = np.array(distances)
        
        # convert distances to similarities for consistency
        if self.metric == "cosine":
            result_scores = 1.0 - result_scores  # angular distance to similarity
        else:
            result_scores = -result_scores  # negative distance = similarity
        
        return result_scores, result_ids
    
    def save(self, path: str):
        """Save Annoy index."""
        if not self.built:
            raise ValueError("Must build index before saving")
        
        self.index.save(path)
        np.savez(path + "_meta.npz", ids=self.ids, dim=self.dim, metric=self.metric)
        self.log.info(f"Saved Annoy index to {path}")
    
    def load(self, path: str):
        """Load Annoy index."""
        meta = np.load(path + "_meta.npz", allow_pickle=True)
        self.ids = meta['ids'].tolist()
        self.dim = int(meta['dim'])
        self.metric = str(meta['metric'])
        
        metric_map = {"cosine": "angular", "euclidean": "euclidean", "dot": "dot"}
        annoy_metric = metric_map.get(self.metric, "angular")
        self.index = self.AnnoyIndex(self.dim, annoy_metric)
        self.index.load(path)
        self.built = True
        self.log.info(f"Loaded Annoy index from {path}")


def create_index(backend: str = "numpy", **kwargs) -> VectorIndex:
    """
    Factory function to create vector index.
    
    backend: "numpy", "faiss", or "annoy"
    kwargs: passed to index constructor
    """
    
    if backend == "numpy":
        return NumpyIndex(**kwargs)
    elif backend == "faiss":
        return FAISSIndex(**kwargs)
    elif backend == "annoy":
        return AnnoyIndex(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")

