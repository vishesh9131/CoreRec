"""
Dataset classes for CoreRec framework.

This module contains dataset classes for recommendation tasks.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import logging
import random
from collections import defaultdict


class RecommendationDataset(Dataset):
    """
    Base dataset class for recommendation tasks.
    
    This dataset handles user-item interactions with optional features.
    
    Attributes:
        interactions (pd.DataFrame): User-item interaction data
        user_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping user IDs to feature tensors
        item_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping item IDs to feature tensors
        num_negatives (int): Number of negative samples per positive sample
        negative_sampling_strategy (str): Strategy for negative sampling
    """
    
    def __init__(
        self, 
        interactions: pd.DataFrame,
        user_features: Optional[Dict[Any, torch.Tensor]] = None,
        item_features: Optional[Dict[Any, torch.Tensor]] = None,
        num_negatives: int = 4,
        negative_sampling_strategy: str = 'random',
        user_id_col: str = 'user_id',
        item_id_col: str = 'item_id',
        rating_col: Optional[str] = 'rating',
        transform: Optional[Callable] = None
    ):
        """Initialize the recommendation dataset.
        
        Args:
            interactions (pd.DataFrame): User-item interaction data with columns:
                - user_id_col: Column name for user IDs
                - item_id_col: Column name for item IDs
                - rating_col: Optional column name for ratings
            user_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping user IDs to feature tensors
            item_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping item IDs to feature tensors
            num_negatives (int): Number of negative samples per positive sample
            negative_sampling_strategy (str): Strategy for negative sampling ('random', 'popular')
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            rating_col (Optional[str]): Column name for ratings
            transform (Optional[Callable]): Optional transform to apply to the data
        """
        self.interactions = interactions
        self.user_features = user_features
        self.item_features = item_features
        self.num_negatives = num_negatives
        self.negative_sampling_strategy = negative_sampling_strategy
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col
        self.transform = transform
        
        # Extract unique users and items
        self.users = interactions[user_id_col].unique()
        self.items = interactions[item_id_col].unique()
        
        # Create user-item interaction dictionary
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            user_id = row[user_id_col]
            item_id = row[item_id_col]
            self.user_items[user_id].add(item_id)
        
        # Create positive samples
        self.samples = []
        for _, row in interactions.iterrows():
            user_id = row[user_id_col]
            item_id = row[item_id_col]
            rating = row[rating_col] if rating_col is not None else 1.0
            self.samples.append((user_id, item_id, rating))
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.samples) * (1 + self.num_negatives)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - user_id: User ID
                - item_id: Item ID
                - user_features: User features (if available)
                - item_features: Item features (if available)
                - label: Binary label (1 for positive, 0 for negative)
                - rating: Rating (if available)
        """
        # Determine if positive or negative sample
        is_positive = idx < len(self.samples)
        sample_idx = idx % len(self.samples)
        
        if is_positive:
            # Positive sample
            user_id, item_id, rating = self.samples[sample_idx]
            label = 1.0
        else:
            # Negative sample
            user_id, positive_item_id, rating = self.samples[sample_idx]
            item_id = self._sample_negative(user_id, positive_item_id)
            label = 0.0
        
        # Create sample
        sample = {
            'user_id': user_id,
            'item_id': item_id,
            'label': torch.tensor(label, dtype=torch.float32),
            'rating': torch.tensor(rating, dtype=torch.float32) if self.rating_col is not None else None
        }
        
        # Add features if available
        if self.user_features is not None:
            sample['user_features'] = self.user_features.get(user_id, torch.zeros(next(iter(self.user_features.values())).shape))
            
        if self.item_features is not None:
            sample['item_features'] = self.item_features.get(item_id, torch.zeros(next(iter(self.item_features.values())).shape))
        
        # Apply transform if available
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def _sample_negative(self, user_id: Any, positive_item_id: Any) -> Any:
        """Sample a negative item for a user.
        
        Args:
            user_id (Any): User ID
            positive_item_id (Any): Positive item ID to avoid
            
        Returns:
            Any: Negative item ID
        """
        # Get interacted items for user
        interacted_items = self.user_items[user_id]
        
        # Sample negative item
        if self.negative_sampling_strategy == 'random':
            # Random sampling
            while True:
                negative_item = random.choice(self.items)
                if negative_item not in interacted_items and negative_item != positive_item_id:
                    return negative_item
        else:
            # Popular item sampling (not implemented, fallback to random)
            return self._sample_negative(user_id, positive_item_id)


class DSSMDataset(RecommendationDataset):
    """
    Dataset for DSSM (Deep Structured Semantic Model) training.
    
    This dataset is specifically designed for DSSM training,
    where user and item features are required.
    """
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - user_features: User features
                - item_features: Item features
                - label: Binary label (1 for positive, 0 for negative)
        """
        # Get base sample
        sample = super().__getitem__(idx)
        
        # Ensure features are available
        if 'user_features' not in sample or 'item_features' not in sample:
            raise ValueError("User and item features are required for DSSM training")
        
        # Create DSSM-specific sample
        dssm_sample = {
            'user_features': sample['user_features'],
            'item_features': sample['item_features'],
            'labels': sample['label']
        }
        
        return dssm_sample


class SequentialRecommendationDataset(Dataset):
    """
    Dataset for sequential recommendation tasks.
    
    This dataset handles sequences of user-item interactions for
    sequential recommendation models like SASRec, BERT4Rec, etc.
    
    Attributes:
        interactions (pd.DataFrame): User-item interaction data
        user_seq (Dict[Any, List[Any]]): Dict mapping user IDs to item sequence
        max_seq_length (int): Maximum sequence length
        mask_prob (float): Probability of masking items for BERT-style training
    """
    
    def __init__(
        self, 
        interactions: pd.DataFrame,
        max_seq_length: int = 50,
        mask_prob: float = 0.2,
        user_id_col: str = 'user_id',
        item_id_col: str = 'item_id',
        timestamp_col: str = 'timestamp',
        item_features: Optional[Dict[Any, torch.Tensor]] = None,
        mode: str = 'next_item'  # 'next_item', 'mask', 'sequence'
    ):
        """Initialize the sequential recommendation dataset.
        
        Args:
            interactions (pd.DataFrame): User-item interaction data with columns:
                - user_id_col: Column name for user IDs
                - item_id_col: Column name for item IDs
                - timestamp_col: Column name for timestamps
            max_seq_length (int): Maximum sequence length
            mask_prob (float): Probability of masking items for BERT-style training
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            timestamp_col (str): Column name for timestamps
            item_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping item IDs to feature tensors
            mode (str): Training mode ('next_item', 'mask', 'sequence')
        """
        self.interactions = interactions
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.timestamp_col = timestamp_col
        self.item_features = item_features
        self.mode = mode
        
        # Extract unique users and items
        self.users = interactions[user_id_col].unique()
        self.items = interactions[item_id_col].unique()
        
        # Create item mapping for padding and masking
        self.item_map = {item: idx + 1 for idx, item in enumerate(self.items)}
        self.item_map['PAD'] = 0
        self.item_map['MASK'] = len(self.items) + 1
        
        # Create user-item sequence dictionary
        self.user_seq = self._create_user_sequences()
        
        # Create samples from sequences
        self.samples = []
        for user_id, seq in self.user_seq.items():
            if len(seq) > 1:  # Need at least 2 items
                self.samples.append(user_id)
    
    def _create_user_sequences(self) -> Dict[Any, List[Any]]:
        """Create user-item sequences from interactions.
        
        Returns:
            Dict[Any, List[Any]]: Dict mapping user IDs to item sequences
        """
        # Sort interactions by user and timestamp
        sorted_interactions = self.interactions.sort_values(
            [self.user_id_col, self.timestamp_col]
        )
        
        # Create user sequences
        user_seq = defaultdict(list)
        for _, row in sorted_interactions.iterrows():
            user_id = row[self.user_id_col]
            item_id = row[self.item_id_col]
            user_seq[user_id].append(item_id)
        
        # Truncate sequences to max_seq_length
        for user_id in user_seq:
            user_seq[user_id] = user_seq[user_id][-self.max_seq_length:]
        
        return user_seq
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing sequence data
        """
        user_id = self.samples[idx]
        seq = self.user_seq[user_id]
        
        if self.mode == 'next_item':
            # Train to predict the next item
            input_seq = seq[:-1]
            target = seq[-1]
            
            # Pad sequence if needed
            if len(input_seq) < self.max_seq_length - 1:
                input_seq = ['PAD'] * (self.max_seq_length - 1 - len(input_seq)) + input_seq
            
            # Convert to tensor
            input_seq = torch.tensor([self.item_map.get(item, 0) for item in input_seq], dtype=torch.long)
            target = torch.tensor(self.item_map.get(target, 0), dtype=torch.long)
            
            return {
                'user_id': user_id,
                'input_seq': input_seq,
                'target': target
            }
            
        elif self.mode == 'mask':
            # BERT-style masked item prediction
            tokens = seq.copy()
            
            # Randomly mask some tokens
            masked_pos = []
            masked_tokens = []
            for i in range(len(tokens)):
                if random.random() < self.mask_prob:
                    masked_pos.append(i)
                    masked_tokens.append(tokens[i])
                    tokens[i] = 'MASK'
            
            # Pad sequence if needed
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
            if len(tokens) < self.max_seq_length:
                attention_mask[:(self.max_seq_length - len(tokens))] = 0
                tokens = ['PAD'] * (self.max_seq_length - len(tokens)) + tokens
                
            # Convert to tensors
            tokens = torch.tensor([self.item_map.get(item, 0) for item in tokens], dtype=torch.long)
            masked_pos = torch.tensor(masked_pos, dtype=torch.long)
            masked_tokens = torch.tensor([self.item_map.get(item, 0) for item in masked_tokens], dtype=torch.long)
            
            return {
                'user_id': user_id,
                'tokens': tokens,
                'attention_mask': attention_mask,
                'masked_pos': masked_pos,
                'masked_tokens': masked_tokens
            }
        
        else:  # 'sequence'
            # Full sequence prediction
            tokens = seq.copy()
            
            # Pad sequence if needed
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)
            if len(tokens) < self.max_seq_length:
                attention_mask[:(self.max_seq_length - len(tokens))] = 0
                tokens = ['PAD'] * (self.max_seq_length - len(tokens)) + tokens
            
            # Convert to tensor
            tokens = torch.tensor([self.item_map.get(item, 0) for item in tokens], dtype=torch.long)
            
            return {
                'user_id': user_id,
                'tokens': tokens,
                'attention_mask': attention_mask
            } 