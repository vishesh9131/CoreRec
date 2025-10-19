"""
Collisionless embedding tables for large-scale recommendation systems.

This module provides a collisionless embedding table implementation
for handling large categorical features efficiently.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import xxhash


class CollisionlessEmbedding(nn.Module):
    """
    Collisionless embedding table for large-scale recommendation systems.
    
    This implementation uses hashing to map large categorical features
    to embedding vectors while minimizing collisions.
    
    Attributes:
        num_embeddings (int): Number of embeddings in the table
        embedding_dim (int): Dimension of each embedding vector
        hash_function (str): Hash function to use ('xxh64', 'xxh32', 'murmur')
        num_hash_functions (int): Number of hash functions to use
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                hash_function: str = 'xxh64', num_hash_functions: int = 2,
                seed: int = 42):
        """Initialize the collisionless embedding table.
        
        Args:
            num_embeddings (int): Number of embeddings in the table
            embedding_dim (int): Dimension of each embedding vector
            hash_function (str): Hash function to use ('xxh64', 'xxh32', 'murmur')
            num_hash_functions (int): Number of hash functions to use
            seed (int): Random seed for reproducibility
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hash_function = hash_function
        self.num_hash_functions = num_hash_functions
        self.seed = seed
        
        # Create embedding tables
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim // num_hash_functions)
            for _ in range(num_hash_functions)
        ])
        
        # Initialize hash seeds
        self.hash_seeds = [seed + i for i in range(num_hash_functions)]
        
    def _hash_id(self, id_value: int, seed: int) -> int:
        """Hash a categorical ID to an index.
        
        Args:
            id_value (int): Categorical ID to hash
            seed (int): Seed for the hash function
            
        Returns:
            int: Hashed index in the range [0, num_embeddings)
        """
        if self.hash_function == 'xxh64':
            hasher = xxhash.xxh64(seed=seed)
            hasher.update(str(id_value).encode())
            return hasher.intdigest() % self.num_embeddings
        elif self.hash_function == 'xxh32':
            hasher = xxhash.xxh32(seed=seed)
            hasher.update(str(id_value).encode())
            return hasher.intdigest() % self.num_embeddings
        else:  # Default hash
            np.random.seed(seed)
            return hash(str(id_value) + str(seed)) % self.num_embeddings
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the collisionless embedding table.
        
        Args:
            input_ids (torch.Tensor): Input IDs of shape [batch_size, ...]
            
        Returns:
            torch.Tensor: Embedded vectors of shape [batch_size, ..., embedding_dim]
        """
        # Save original shape
        original_shape = input_ids.shape
        
        # Flatten input for processing
        flat_ids = input_ids.view(-1)
        
        # Initialize output embeddings
        output_embeddings = torch.zeros(flat_ids.shape[0], self.embedding_dim, 
                                        device=input_ids.device)
        
        # Compute embeddings from each hash function
        start_dim = 0
        for i, (table, seed) in enumerate(zip(self.embedding_tables, self.hash_seeds)):
            # Hash input IDs
            hashed_ids = torch.tensor(
                [self._hash_id(id_val.item(), seed) for id_val in flat_ids],
                device=input_ids.device
            )
            
            # Get embeddings for hashed IDs
            embeddings = table(hashed_ids)
            
            # Calculate dimensions for this segment
            dim_size = table.embedding_dim
            
            # Add to output
            output_embeddings[:, start_dim:start_dim + dim_size] = embeddings
            start_dim += dim_size
        
        # Reshape to original dimensions plus embedding dim
        output_shape = original_shape + (self.embedding_dim,)
        return output_embeddings.view(output_shape)


class BloomEmbedding(nn.Module):
    """
    Bloom filter inspired embedding for massive-scale categorical features.
    
    This uses multiple small hash functions to approximate embeddings for
    extremely large categorical spaces (e.g., billions of IDs).
    """
    
    def __init__(self, embedding_dim: int, num_hash_functions: int = 4, 
                table_size: int = 1000000, seed: int = 42):
        """Initialize the bloom embedding table.
        
        Args:
            embedding_dim (int): Dimension of each embedding vector
            num_hash_functions (int): Number of hash functions to use
            table_size (int): Size of each hash table
            seed (int): Random seed for reproducibility
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_hash_functions = num_hash_functions
        self.table_size = table_size
        self.seed = seed
        
        # Each hash function gets a separate embedding table
        self.embedding_tables = nn.ModuleList([
            nn.Embedding(table_size, embedding_dim // num_hash_functions)
            for _ in range(num_hash_functions)
        ])
        
        # Initialize hash seeds
        self.hash_seeds = [seed + i for i in range(num_hash_functions)]
        
    def _hash_id(self, id_value: int, seed: int) -> int:
        """Hash a categorical ID to an index.
        
        Args:
            id_value (int): Categorical ID to hash
            seed (int): Seed for the hash function
            
        Returns:
            int: Hashed index in the range [0, table_size)
        """
        hasher = xxhash.xxh64(seed=seed)
        hasher.update(str(id_value).encode())
        return hasher.intdigest() % self.table_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bloom embedding table.
        
        Args:
            input_ids (torch.Tensor): Input IDs of shape [batch_size, ...]
            
        Returns:
            torch.Tensor: Embedded vectors of shape [batch_size, ..., embedding_dim]
        """
        # Save original shape
        original_shape = input_ids.shape
        
        # Flatten input for processing
        flat_ids = input_ids.view(-1)
        
        # Initialize output embeddings
        output_embeddings = torch.zeros(flat_ids.shape[0], self.embedding_dim, 
                                        device=input_ids.device)
        
        # Process in chunks for efficiency
        chunk_size = 10000
        for chunk_start in range(0, flat_ids.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, flat_ids.shape[0])
            chunk_ids = flat_ids[chunk_start:chunk_end]
            
            # Compute embeddings from each hash function
            start_dim = 0
            for i, (table, seed) in enumerate(zip(self.embedding_tables, self.hash_seeds)):
                # Hash input IDs
                hashed_ids = torch.tensor(
                    [self._hash_id(id_val.item(), seed) for id_val in chunk_ids],
                    device=input_ids.device
                )
                
                # Get embeddings for hashed IDs
                embeddings = table(hashed_ids)
                
                # Calculate dimensions for this segment
                dim_size = table.embedding_dim
                
                # Add to output
                output_embeddings[chunk_start:chunk_end, start_dim:start_dim + dim_size] = embeddings
                start_dim += dim_size
        
        # Reshape to original dimensions plus embedding dim
        output_shape = original_shape + (self.embedding_dim,)
        return output_embeddings.view(output_shape) 