"""
Multimodal Embeddings

Combine embeddings from different modalities (text, image, etc).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np


class MultimodalEncoder:
    """
    Combine embeddings from multiple modalities into a single vector.
    
    Useful when items have multiple representations:
    - Product: title (text) + image + category (categorical)
    - Video: title (text) + thumbnail (image) + transcript (text)
    - Music: artist (text) + audio features + genre
    
    Fusion strategies:
    - 'concat': concatenate all embeddings
    - 'average': average embeddings (requires same dim)
    - 'weighted': weighted average
    - 'attention': learned attention weights (requires training)
    
    Example:
        encoder = MultimodalEncoder(
            encoders={
                'text': text_encoder,
                'image': image_encoder,
            },
            fusion='concat',
        )
        
        item_data = {
            'text': "red running shoes",
            'image': image_array,
        }
        embedding = encoder.encode(item_data)
    """
    
    def __init__(
        self,
        encoders: Optional[Dict[str, Any]] = None,
        fusion: str = "concat",
        weights: Optional[Dict[str, float]] = None,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            encoders: dict of modality_name -> encoder
            fusion: 'concat', 'average', or 'weighted'
            weights: weights per modality (for weighted fusion)
            output_dim: target output dimension (for projection)
        """
        self.encoders = encoders or {}
        self.fusion = fusion
        self.weights = weights or {k: 1.0 for k in self.encoders}
        self.output_dim = output_dim
        
        self._projection = None  # learned projection if needed
    
    def add_encoder(
        self,
        name: str,
        encoder: Any,
        weight: float = 1.0
    ) -> "MultimodalEncoder":
        """
        Add a modality encoder.
        
        Args:
            name: modality name
            encoder: encoder with encode() method
            weight: weight for fusion
        """
        self.encoders[name] = encoder
        self.weights[name] = weight
        return self
    
    def encode(
        self,
        data: Dict[str, Any],
        missing_ok: bool = True,
    ) -> np.ndarray:
        """
        Encode multimodal data into a single embedding.
        
        Args:
            data: dict of modality_name -> modality_data
            missing_ok: if True, skip missing modalities
        
        Returns:
            fused embedding vector
        """
        embeddings = {}
        
        for name, encoder in self.encoders.items():
            if name not in data:
                if missing_ok:
                    continue
                raise ValueError(f"Missing modality: {name}")
            
            modality_data = data[name]
            
            # skip None values
            if modality_data is None:
                continue
            
            # encode
            if hasattr(encoder, 'encode'):
                emb = encoder.encode(modality_data)
            elif callable(encoder):
                emb = encoder(modality_data)
            else:
                raise ValueError(f"Encoder for {name} must have encode() or be callable")
            
            embeddings[name] = np.asarray(emb)
        
        if not embeddings:
            raise ValueError("No modalities to encode")
        
        # fuse embeddings
        return self._fuse(embeddings)
    
    def _fuse(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse embeddings based on strategy."""
        if self.fusion == "concat":
            return self._fuse_concat(embeddings)
        elif self.fusion == "average":
            return self._fuse_average(embeddings)
        elif self.fusion == "weighted":
            return self._fuse_weighted(embeddings)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion}")
    
    def _fuse_concat(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate embeddings."""
        # maintain consistent order
        ordered = [embeddings[k] for k in sorted(embeddings.keys())]
        return np.concatenate(ordered)
    
    def _fuse_average(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Average embeddings (must have same dim)."""
        emb_list = list(embeddings.values())
        
        # check dimensions
        dims = [e.shape for e in emb_list]
        if len(set(dims)) > 1:
            raise ValueError(
                f"Cannot average embeddings with different dims: {dims}. "
                "Use 'concat' or project to common dimension."
            )
        
        return np.mean(emb_list, axis=0)
    
    def _fuse_weighted(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of embeddings."""
        emb_list = list(embeddings.values())
        
        # check dimensions
        dims = [e.shape for e in emb_list]
        if len(set(dims)) > 1:
            raise ValueError(f"Cannot weight-average embeddings with different dims: {dims}")
        
        total_weight = 0.0
        weighted_sum = np.zeros_like(emb_list[0])
        
        for name, emb in embeddings.items():
            w = self.weights.get(name, 1.0)
            weighted_sum += w * emb
            total_weight += w
        
        return weighted_sum / total_weight
    
    def encode_batch(
        self,
        data_batch: List[Dict[str, Any]],
        missing_ok: bool = True,
    ) -> np.ndarray:
        """
        Encode a batch of multimodal data.
        
        Args:
            data_batch: list of data dicts
            missing_ok: skip missing modalities
        
        Returns:
            array of shape (batch_size, embedding_dim)
        """
        return np.stack([
            self.encode(data, missing_ok=missing_ok)
            for data in data_batch
        ])
    
    @property
    def embedding_dim(self) -> int:
        """Get output embedding dimension."""
        if self.output_dim:
            return self.output_dim
        
        if self.fusion == "concat":
            total = 0
            for encoder in self.encoders.values():
                if hasattr(encoder, 'embedding_dim'):
                    total += encoder.embedding_dim
            return total
        else:
            # for average/weighted, all dims must match
            for encoder in self.encoders.values():
                if hasattr(encoder, 'embedding_dim'):
                    return encoder.embedding_dim
        
        return 0
    
    def __repr__(self) -> str:
        modalities = list(self.encoders.keys())
        return f"MultimodalEncoder(modalities={modalities}, fusion={self.fusion})"
