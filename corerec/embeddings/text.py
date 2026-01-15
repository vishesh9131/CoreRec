"""
Text Embeddings

Encode text into dense vectors using transformer models.
"""

from typing import Any, List, Optional, Union
import numpy as np


class TextEncoder:
    """
    Encode text into embeddings using sentence transformers or similar.
    
    Wraps sentence-transformers or other text encoding models
    with a simple interface.
    
    Example:
        encoder = TextEncoder("sentence-transformers/all-MiniLM-L6-v2")
        
        # single text
        emb = encoder.encode("red running shoes")
        
        # batch
        embs = encoder.encode(["red shoes", "blue jacket", "green hat"])
    """
    
    def __init__(
        self,
        model: Optional[Union[str, Any]] = None,
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Args:
            model: model name or pre-loaded model instance
            device: 'cpu' or 'cuda'
            normalize: whether to L2-normalize embeddings
        """
        self.model_name = model if isinstance(model, str) else None
        self._model = None if isinstance(model, str) else model
        self.device = device
        self.normalize = normalize
        self._dim: Optional[int] = None
    
    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            self._model = self._load_model(self.model_name)
        return self._model
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._dim is None:
            _ = self.model  # ensure loaded
            if hasattr(self._model, 'get_sentence_embedding_dimension'):
                self._dim = self._model.get_sentence_embedding_dimension()
            else:
                # probe by encoding something
                test = self.encode("test")
                self._dim = len(test)
        return self._dim
    
    def _load_model(self, model_name: str):
        """Load the embedding model."""
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: single string or list of strings
            batch_size: batch size for encoding
            show_progress: show progress bar
        
        Returns:
            numpy array of shape (n, dim) or (dim,) for single text
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            embeddings = embeddings / norms
        
        if single:
            return embeddings[0]
        return embeddings
    
    def similarity(
        self,
        text_a: Union[str, np.ndarray],
        text_b: Union[str, np.ndarray],
    ) -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text_a: text or embedding
            text_b: text or embedding
        
        Returns:
            cosine similarity score
        """
        if isinstance(text_a, str):
            emb_a = self.encode(text_a)
        else:
            emb_a = text_a
        
        if isinstance(text_b, str):
            emb_b = self.encode(text_b)
        else:
            emb_b = text_b
        
        # cosine similarity (already normalized if self.normalize=True)
        return float(np.dot(emb_a, emb_b))
    
    def __repr__(self) -> str:
        return f"TextEncoder(model={self.model_name}, dim={self._dim})"
