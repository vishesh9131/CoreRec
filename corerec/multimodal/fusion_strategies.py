"""
Multi-modal fusion strategies for combining different data types.

Modern recsys uses text, images, audio, user behavior all together.
This module provides ways to combine these embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from enum import Enum


class FusionStrategy(Enum):
    """Different ways to combine modalities."""
    CONCAT = "concat"  # simple concatenation
    WEIGHTED = "weighted"  # learned weights
    ATTENTION = "attention"  # attention-based fusion
    GATED = "gated"  # gated fusion (like in VQA)
    BILINEAR = "bilinear"  # bilinear pooling
    TUCKER = "tucker"  # tensor decomposition


class ConcatFusion(nn.Module):
    """
    Simplest fusion: concatenate embeddings and project.
    Fast but doesn't model cross-modal interactions well.
    """
    
    def __init__(self, input_dims: Dict[str, int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        total_dim = sum(input_dims.values())
        
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        embeddings: dict with keys matching input_dims, values are [batch, dim] tensors
        """
        # concatenate in consistent order
        ordered_keys = sorted(self.input_dims.keys())
        concat = torch.cat([embeddings[k] for k in ordered_keys], dim=-1)
        
        return self.projection(concat)


class WeightedFusion(nn.Module):
    """
    Learned weighted sum of modalities.
    Good when modalities have different importance.
    """
    
    def __init__(self, modalities: List[str], embedding_dim: int):
        super().__init__()
        
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        
        # learnable weights (initialized to equal)
        self.weights = nn.Parameter(torch.ones(len(modalities)) / len(modalities))
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        embeddings: dict with [batch, embedding_dim] tensors
        All embeddings must have same dimension.
        """
        # normalize weights
        weights = F.softmax(self.weights, dim=0)
        
        # weighted sum
        result = torch.zeros_like(embeddings[self.modalities[0]])
        
        for i, modality in enumerate(self.modalities):
            if modality in embeddings:
                result += weights[i] * embeddings[modality]
        
        return result


class AttentionFusion(nn.Module):
    """
    Use attention to dynamically weight modalities.
    More flexible than fixed weights, adapts per instance.
    """
    
    def __init__(self, embedding_dim: int, num_modalities: int, num_heads: int = 4):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # use multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # learnable query
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # output projection
        self.output = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Use attention to aggregate modality embeddings.
        """
        batch_size = next(iter(embeddings.values())).size(0)
        
        # stack modalities [batch, num_modalities, embed_dim]
        modality_stack = torch.stack(list(embeddings.values()), dim=1)
        
        # expand query for batch
        query = self.query.expand(batch_size, -1, -1)
        
        # apply attention: query attends to modalities
        attended, _ = self.attention(query, modality_stack, modality_stack)
        
        # squeeze and project
        result = self.output(attended.squeeze(1))
        
        return result


class GatedFusion(nn.Module):
    """
    Gated fusion - learns to filter/select from each modality.
    
    Used in VQA and other multi-modal tasks.
    Each modality gets a gate that controls its contribution.
    """
    
    def __init__(self, modalities: List[str], embedding_dim: int):
        super().__init__()
        
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        
        # gate for each modality
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            for name in modalities
        })
        
        # combine gated representations
        self.combine = nn.Linear(embedding_dim * len(modalities), embedding_dim)
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply gating to each modality then combine.
        """
        # compute global context (mean of all modalities)
        context = torch.mean(torch.stack(list(embeddings.values())), dim=0)
        
        gated = []
        
        for modality in self.modalities:
            if modality in embeddings:
                emb = embeddings[modality]
                
                # gate input: concat embedding with context
                gate_input = torch.cat([emb, context], dim=-1)
                gate = self.gates[modality](gate_input)
                
                # apply gate
                gated_emb = gate * emb
                gated.append(gated_emb)
        
        # concatenate and project
        combined = torch.cat(gated, dim=-1)
        result = self.combine(combined)
        
        return result


class BilinearFusion(nn.Module):
    """
    Bilinear pooling for two modalities.
    Captures multiplicative interactions between features.
    
    Commonly used for VQA (vision + language).
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        
        # bilinear layer: computes x1^T W x2
        self.bilinear = nn.Bilinear(dim1, dim2, output_dim)
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        emb1: [batch, dim1]
        emb2: [batch, dim2]
        """
        return self.bilinear(emb1, emb2)


class MultiModalFusion(nn.Module):
    """
    Unified interface for multi-modal fusion.
    Handles arbitrary number of modalities with flexible strategies.
    """
    
    def __init__(self, 
                 modality_dims: Dict[str, int],
                 output_dim: int,
                 strategy: str = "attention",
                 dropout: float = 0.1):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.strategy = strategy
        
        # project each modality to common dimension first
        self.modality_projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })
        
        # fusion module
        if strategy == "concat":
            # after projection, concat and project again
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * len(modality_dims), output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )
        
        elif strategy == "weighted":
            self.fusion = WeightedFusion(
                modalities=list(modality_dims.keys()),
                embedding_dim=output_dim
            )
        
        elif strategy == "attention":
            self.fusion = AttentionFusion(
                embedding_dim=output_dim,
                num_modalities=len(modality_dims)
            )
        
        elif strategy == "gated":
            self.fusion = GatedFusion(
                modalities=list(modality_dims.keys()),
                embedding_dim=output_dim
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def forward(self, raw_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse embeddings from different modalities.
        
        raw_embeddings: dict with modality name -> [batch, modality_dim] tensor
        Returns: [batch, output_dim] fused representation
        """
        # project to common space
        projected = {}
        for name, emb in raw_embeddings.items():
            if name in self.modality_projections:
                projected[name] = self.modality_projections[name](emb)
        
        # fuse
        if self.strategy == "concat":
            concat_input = torch.cat(list(projected.values()), dim=-1)
            return self.fusion(concat_input)
        else:
            return self.fusion(projected)


# Example usage patterns:
"""
# Text + Image fusion for item representation
fusion = MultiModalFusion(
    modality_dims={
        'text': 768,  # BERT output
        'image': 2048,  # ResNet output
    },
    output_dim=256,
    strategy='attention'
)

item_emb = fusion({
    'text': text_embeddings,  # [batch, 768]
    'image': image_embeddings  # [batch, 2048]
})


# User features: demographics + behavior + context
user_fusion = MultiModalFusion(
    modality_dims={
        'demographics': 32,
        'behavior': 128,
        'context': 64
    },
    output_dim=128,
    strategy='gated'
)

user_emb = user_fusion({
    'demographics': demo_features,
    'behavior': behavior_history,
    'context': context_features
})
"""

