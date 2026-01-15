"""
Embeddings Module - Representation Learning

Provides utilities for creating and managing embeddings
from various modalities (text, images, behavior).

Usage:
    from corerec.embeddings import TextEncoder, MultimodalEncoder
    
    encoder = TextEncoder(model="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(["red shoes", "blue jacket"])
"""

from .text import TextEncoder
from .multimodal import MultimodalEncoder
from .pretrained import PretrainedEmbeddings

__all__ = [
    "TextEncoder",
    "MultimodalEncoder",
    "PretrainedEmbeddings",
]
