"""
Pretrained Embeddings

Load and use pretrained embeddings from various sources.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np


class PretrainedEmbeddings:
    """
    Load and query pretrained embedding tables.
    
    Supports loading from:
    - NumPy files (.npy, .npz)
    - Pickle files (.pkl)
    - Text files (word2vec format)
    
    Example:
        embs = PretrainedEmbeddings.load("item_embeddings.npy", item_ids)
        
        # get embedding for an item
        vec = embs.get(item_id=123)
        
        # find similar items
        similar = embs.most_similar(item_id=123, top_k=10)
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        ids: List[Any],
        normalize: bool = False,
    ):
        """
        Args:
            embeddings: array of shape (n_items, dim)
            ids: list of item identifiers (same order as embeddings)
            normalize: whether to L2-normalize embeddings
        """
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.ids = list(ids)
        self._id_to_idx = {id_: i for i, id_ in enumerate(self.ids)}
        
        if normalize:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            self.embeddings = self.embeddings / norms
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        ids: Optional[List[Any]] = None,
        normalize: bool = False,
    ) -> "PretrainedEmbeddings":
        """
        Load embeddings from file.
        
        Args:
            path: path to embedding file
            ids: item identifiers (required for .npy, optional for .npz with 'ids')
            normalize: whether to normalize
        
        Returns:
            PretrainedEmbeddings instance
        """
        path = Path(path)
        
        if path.suffix == '.npy':
            embeddings = np.load(path)
            if ids is None:
                ids = list(range(len(embeddings)))
        
        elif path.suffix == '.npz':
            data = np.load(path)
            embeddings = data['embeddings']
            if ids is None:
                ids = list(data.get('ids', range(len(embeddings))))
        
        elif path.suffix == '.pkl':
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                embeddings = data.get('embeddings', data.get('vectors'))
                ids = data.get('ids', data.get('keys', list(range(len(embeddings)))))
            else:
                embeddings = data
                ids = list(range(len(embeddings))) if ids is None else ids
        
        elif path.suffix in ['.txt', '.vec']:
            embeddings, ids = cls._load_text_format(path)
        
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return cls(embeddings, ids, normalize=normalize)
    
    @staticmethod
    def _load_text_format(path: Path) -> tuple:
        """Load word2vec-style text format."""
        embeddings = []
        ids = []
        
        with open(path, 'r', encoding='utf-8') as f:
            # first line might be header (num_vectors dim)
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                # header line, skip
                pass
            else:
                # not header, this is data
                ids.append(first_line[0])
                embeddings.append([float(x) for x in first_line[1:]])
            
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    ids.append(parts[0])
                    embeddings.append([float(x) for x in parts[1:]])
        
        return np.array(embeddings), ids
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self.embeddings.shape[1]
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __contains__(self, item_id: Any) -> bool:
        return item_id in self._id_to_idx
    
    def get(self, item_id: Any, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get embedding for an item.
        
        Args:
            item_id: item identifier
            default: value to return if not found
        
        Returns:
            embedding vector or default
        """
        idx = self._id_to_idx.get(item_id)
        if idx is None:
            return default
        return self.embeddings[idx]
    
    def get_batch(
        self,
        item_ids: List[Any],
        default: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get embeddings for multiple items.
        
        Args:
            item_ids: list of item identifiers
            default: default embedding for missing items
        
        Returns:
            array of shape (len(item_ids), dim)
        """
        result = []
        
        for item_id in item_ids:
            emb = self.get(item_id)
            if emb is None:
                if default is not None:
                    emb = default
                else:
                    emb = np.zeros(self.dim)
            result.append(emb)
        
        return np.stack(result)
    
    def most_similar(
        self,
        query: Union[Any, np.ndarray],
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> List[tuple]:
        """
        Find most similar items.
        
        Args:
            query: item_id or embedding vector
            top_k: number of results
            exclude_self: exclude the query item from results
        
        Returns:
            list of (item_id, similarity) tuples
        """
        if isinstance(query, np.ndarray):
            query_emb = query
            query_id = None
        else:
            query_emb = self.get(query)
            query_id = query
            
            if query_emb is None:
                raise KeyError(f"Item not found: {query}")
        
        # compute similarities
        sims = self.embeddings @ query_emb
        
        # get top k
        indices = np.argsort(sims)[::-1]
        
        results = []
        for idx in indices:
            item_id = self.ids[idx]
            
            if exclude_self and item_id == query_id:
                continue
            
            results.append((item_id, float(sims[idx])))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save embeddings to file.
        
        Args:
            path: output path (.npz format)
        """
        path = Path(path)
        
        np.savez(
            path,
            embeddings=self.embeddings,
            ids=np.array(self.ids, dtype=object),
        )
    
    def __repr__(self) -> str:
        return f"PretrainedEmbeddings(n={len(self)}, dim={self.dim})"
