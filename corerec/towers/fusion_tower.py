"""
Fusion tower for recommendation systems.

This module provides a tower for fusing user/item or multi-modal features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional
import logging

from corerec.towers.base_tower import AbstractTower


class FusionTower(AbstractTower):
    """
    Fusion tower for recommendation systems.
    
    This tower fuses representations from multiple towers (e.g., user and item
    towers, or towers for different modalities) into a unified representation.
    
    Architecture:
        Tower 1 Output       Tower 2 Output     ...
              ↓                   ↓
        [Fusion Strategy: concat, attention, gating, etc.]
                        ↓
                 [Dense Layers]
                        ↓
                Fused Representation
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the fusion tower.
        
        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration including:
                - input_dims (Dict[str, int]): Dictionary mapping source names to dimensions
                - output_dim (int): Output dimension
                - fusion_type (str): Fusion strategy ('concat', 'attention', 'gating', 'sum', 'bilinear')
                - hidden_dims (List[int]): Dimensions of hidden layers after fusion
                - dropout (float): Dropout rate
                - use_batch_norm (bool): Whether to use batch normalization
                - fusion_kwargs (Dict[str, Any]): Additional arguments for specific fusion methods
        """
        super().__init__(name, config)
        
        # Get configuration
        self.input_dims = config.get('input_dims', {})
        self.output_dim = config.get('output_dim', 512)
        self.fusion_type = config.get('fusion_type', 'concat').lower()
        self.hidden_dims = config.get('hidden_dims', [])
        self.dropout_rate = config.get('dropout', 0.1)
        self.use_batch_norm = config.get('use_batch_norm', False)
        self.fusion_kwargs = config.get('fusion_kwargs', {})
        
        # Create fusion layer
        self.fusion_layer = self._create_fusion_layer()
        
        # Create projection layers after fusion
        self.projection = self._create_projection_layers()
    
    def _create_fusion_layer(self) -> nn.Module:
        """Create fusion layer based on fusion type.
        
        Returns:
            nn.Module: Fusion layer
        """
        if self.fusion_type == 'concat':
            return ConcatFusion()
        elif self.fusion_type == 'attention':
            return AttentionFusion(self.input_dims)
        elif self.fusion_type == 'gating':
            return GatingFusion(self.input_dims)
        elif self.fusion_type == 'sum':
            return SumFusion()
        elif self.fusion_type == 'bilinear':
            # Only supports two sources
            if len(self.input_dims) != 2:
                raise ValueError("BilinearFusion only supports two input sources")
            source_names = list(self.input_dims.keys())
            return BilinearFusion(
                self.input_dims[source_names[0]],
                self.input_dims[source_names[1]],
                self.fusion_kwargs.get('bilinear_dim', 512)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
    
    def _create_projection_layers(self) -> nn.Sequential:
        """Create projection layers after fusion.
        
        Returns:
            nn.Sequential: Projection layers
        """
        # Calculate fusion output dimension
        if self.fusion_type == 'concat':
            fusion_output_dim = sum(self.input_dims.values())
        elif self.fusion_type in ['attention', 'gating', 'sum']:
            # All input dimensions must be the same
            dims = list(self.input_dims.values())
            if not all(d == dims[0] for d in dims):
                raise ValueError(f"For {self.fusion_type} fusion, all input dimensions must be the same")
            fusion_output_dim = dims[0]
        elif self.fusion_type == 'bilinear':
            fusion_output_dim = self.fusion_kwargs.get('bilinear_dim', 512)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        # Create layers
        if not self.hidden_dims:
            # Simple projection
            if fusion_output_dim != self.output_dim:
                return nn.Linear(fusion_output_dim, self.output_dim)
            else:
                return nn.Identity()
        else:
            # MLP projection
            layers = []
            prev_dim = fusion_output_dim
            
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            
            # Output layer
            if prev_dim != self.output_dim:
                layers.append(nn.Linear(prev_dim, self.output_dim))
            
            return nn.Sequential(*layers)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the fusion tower.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Fused representation
        """
        # Check if all required sources are present
        for source_name in self.input_dims:
            if source_name not in inputs:
                raise ValueError(f"Missing input source: {source_name}")
        
        # Apply fusion
        fused = self.fusion_layer(inputs)
        
        # Apply projection
        output = self.projection(fused)
        
        return output
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode inputs using the fusion tower.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Fused representation
        """
        with torch.no_grad():
            output = self.forward(inputs)
        
        return output


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion.
    
    This fusion method simply concatenates the input representations.
    """
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the concatenation fusion.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Concatenated representation
        """
        # Convert to list of tensors
        tensors = list(inputs.values())
        
        # Concatenate tensors
        return torch.cat(tensors, dim=1)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion.
    
    This fusion method uses attention to dynamically weight the importance
    of different input representations.
    """
    
    def __init__(self, input_dims: Dict[str, int]):
        """Initialize the attention fusion.
        
        Args:
            input_dims (Dict[str, int]): Dictionary mapping source names to dimensions
        """
        super().__init__()
        
        # Check if all dimensions are the same
        dims = list(input_dims.values())
        if not all(d == dims[0] for d in dims):
            raise ValueError("All input dimensions must be the same for attention fusion")
        
        self.dim = dims[0]
        self.source_names = list(input_dims.keys())
        
        # Create query vector
        self.query = nn.Parameter(torch.randn(self.dim))
        
        # Create key projections for each source
        self.key_projections = nn.ModuleDict({
            source_name: nn.Linear(self.dim, self.dim)
            for source_name in self.source_names
        })
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the attention fusion.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Attention-fused representation
        """
        batch_size = next(iter(inputs.values())).size(0)
        
        # Generate keys for attention
        keys = []
        values = []
        
        for source_name in self.source_names:
            # Get input for this source
            source_input = inputs[source_name]
            
            # Generate key
            key = self.key_projections[source_name](source_input)
            keys.append(key)
            
            # Use input as value
            values.append(source_input)
        
        # Stack keys and values
        keys = torch.stack(keys, dim=1)  # [batch_size, num_sources, dim]
        values = torch.stack(values, dim=1)  # [batch_size, num_sources, dim]
        
        # Expand query
        query = self.query.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)  # [batch_size, dim, 1]
        
        # Compute attention scores
        attn_scores = torch.bmm(keys, query).squeeze(2)  # [batch_size, num_sources]
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch_size, num_sources, 1]
        
        # Apply attention weights
        weighted_values = values * attn_weights
        
        # Sum weighted values
        fused = weighted_values.sum(dim=1)  # [batch_size, dim]
        
        return fused


class GatingFusion(nn.Module):
    """
    Gating-based fusion.
    
    This fusion method uses gates to control the information flow from
    different input representations.
    """
    
    def __init__(self, input_dims: Dict[str, int]):
        """Initialize the gating fusion.
        
        Args:
            input_dims (Dict[str, int]): Dictionary mapping source names to dimensions
        """
        super().__init__()
        
        # Check if all dimensions are the same
        dims = list(input_dims.values())
        if not all(d == dims[0] for d in dims):
            raise ValueError("All input dimensions must be the same for gating fusion")
        
        self.dim = dims[0]
        self.source_names = list(input_dims.keys())
        
        # Create gate networks for each source
        self.gate_networks = nn.ModuleDict({
            source_name: nn.Sequential(
                nn.Linear(self.dim, self.dim // 4),
                nn.ReLU(),
                nn.Linear(self.dim // 4, self.dim),
                nn.Sigmoid()
            )
            for source_name in self.source_names
        })
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the gating fusion.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Gating-fused representation
        """
        # Generate gates and apply them
        gated_outputs = []
        
        for source_name in self.source_names:
            # Get input for this source
            source_input = inputs[source_name]
            
            # Generate gate
            gate = self.gate_networks[source_name](source_input)
            
            # Apply gate
            gated_output = source_input * gate
            gated_outputs.append(gated_output)
        
        # Sum gated outputs
        fused = sum(gated_outputs)
        
        return fused


class SumFusion(nn.Module):
    """
    Sum-based fusion.
    
    This fusion method simply sums the input representations element-wise.
    """
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the sum fusion.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Sum-fused representation
        """
        # Convert to list of tensors and sum
        return sum(inputs.values())


class BilinearFusion(nn.Module):
    """
    Bilinear fusion.
    
    This fusion method uses bilinear transformation to capture pairwise
    interactions between two input sources.
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: int):
        """Initialize the bilinear fusion.
        
        Args:
            dim1 (int): Dimension of first input
            dim2 (int): Dimension of second input
            output_dim (int): Output dimension
        """
        super().__init__()
        
        # Create bilinear layer
        self.bilinear = nn.Bilinear(dim1, dim2, output_dim)
        
        # Remember source indices
        self.source_names = None
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the bilinear fusion.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary mapping source names to tensors
            
        Returns:
            torch.Tensor: Bilinear-fused representation
        """
        # Get source names if not already set
        if self.source_names is None:
            self.source_names = list(inputs.keys())
            if len(self.source_names) != 2:
                raise ValueError("BilinearFusion requires exactly two input sources")
        
        # Get inputs
        x1 = inputs[self.source_names[0]]
        x2 = inputs[self.source_names[1]]
        
        # Apply bilinear transformation
        return self.bilinear(x1, x2) 