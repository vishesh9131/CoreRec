"""
Multimodal dataset for recommendation systems.

This module provides dataset classes for multimodal recommendation tasks,
supporting both text and image data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import logging
import random
from PIL import Image
import os
import json
from collections import defaultdict

from corerec.data.datasets import RecommendationDataset


class MultimodalRecommendationDataset(RecommendationDataset):
    """
    Dataset for multimodal recommendation tasks.
    
    This dataset handles user-item interactions with both text and image features.
    
    Attributes:
        interactions (pd.DataFrame): User-item interaction data
        user_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping user IDs to feature tensors
        item_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping item IDs to feature tensors
        item_texts (Optional[Dict[Any, str]]): Dict mapping item IDs to text descriptions
        item_image_paths (Optional[Dict[Any, str]]): Dict mapping item IDs to image paths
        image_transform (Optional[Callable]): Transform to apply to images
    """
    
    def __init__(
        self, 
        interactions: pd.DataFrame,
        user_features: Optional[Dict[Any, torch.Tensor]] = None,
        item_features: Optional[Dict[Any, torch.Tensor]] = None,
        item_texts: Optional[Dict[Any, str]] = None,
        item_image_paths: Optional[Dict[Any, str]] = None,
        image_transform: Optional[Callable] = None,
        num_negatives: int = 4,
        negative_sampling_strategy: str = 'random',
        user_id_col: str = 'user_id',
        item_id_col: str = 'item_id',
        rating_col: Optional[str] = 'rating',
        transform: Optional[Callable] = None,
        text_max_length: int = 512,
        use_image_features: bool = True,
        use_text_features: bool = True,
        preload_images: bool = False
    ):
        """Initialize the multimodal recommendation dataset.
        
        Args:
            interactions (pd.DataFrame): User-item interaction data with columns:
                - user_id_col: Column name for user IDs
                - item_id_col: Column name for item IDs
                - rating_col: Optional column name for ratings
            user_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping user IDs to feature tensors
            item_features (Optional[Dict[Any, torch.Tensor]]): Dict mapping item IDs to feature tensors
            item_texts (Optional[Dict[Any, str]]): Dict mapping item IDs to text descriptions
            item_image_paths (Optional[Dict[Any, str]]): Dict mapping item IDs to image paths
            image_transform (Optional[Callable]): Transform to apply to images
            num_negatives (int): Number of negative samples per positive sample
            negative_sampling_strategy (str): Strategy for negative sampling ('random', 'popular')
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            rating_col (Optional[str]): Column name for ratings
            transform (Optional[Callable]): Optional transform to apply to the data
            text_max_length (int): Maximum text length
            use_image_features (bool): Whether to use image features
            use_text_features (bool): Whether to use text features
            preload_images (bool): Whether to preload images into memory
        """
        super().__init__(
            interactions=interactions,
            user_features=user_features,
            item_features=item_features,
            num_negatives=num_negatives,
            negative_sampling_strategy=negative_sampling_strategy,
            user_id_col=user_id_col,
            item_id_col=item_id_col,
            rating_col=rating_col,
            transform=transform
        )
        
        self.item_texts = item_texts or {}
        self.item_image_paths = item_image_paths or {}
        self.image_transform = image_transform
        self.text_max_length = text_max_length
        self.use_image_features = use_image_features
        self.use_text_features = use_text_features
        
        # Preload images if requested
        self.preloaded_images = {}
        if preload_images and self.item_image_paths:
            self._preload_images()
    
    def _preload_images(self):
        """Preload images into memory."""
        for item_id, image_path in self.item_image_paths.items():
            try:
                image = self._load_image(image_path)
                self.preloaded_images[item_id] = image
            except Exception as e:
                logging.warning(f"Failed to load image for item {item_id}: {e}")
    
    def _load_image(self, image_path: str) -> Any:
        """Load an image from path.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            Any: Loaded image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
            return image
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - user_id: User ID
                - item_id: Item ID
                - user_features: User features (if available)
                - item_features: Item features (if available)
                - text: Item text description (if available)
                - image: Item image (if available)
                - label: Binary label (1 for positive, 0 for negative)
                - rating: Rating (if available)
        """
        # Get base sample
        sample = super().__getitem__(idx)
        item_id = sample['item_id']
        
        # Add text if available
        if self.use_text_features and item_id in self.item_texts:
            sample['text'] = self.item_texts[item_id]
        
        # Add image if available
        if self.use_image_features and item_id in self.item_image_paths:
            if item_id in self.preloaded_images:
                sample['image'] = self.preloaded_images[item_id]
            else:
                image_path = self.item_image_paths[item_id]
                sample['image'] = self._load_image(image_path)
        
        return sample


class MultimodalSequentialDataset(Dataset):
    """
    Dataset for multimodal sequential recommendation tasks.
    
    This dataset handles sequences of user-item interactions with both text and image features.
    
    Attributes:
        interactions (pd.DataFrame): User-item interaction data
        user_seq (Dict[Any, List[Any]]): Dict mapping user IDs to item sequence
        item_texts (Dict[Any, str]): Dict mapping item IDs to text descriptions
        item_image_paths (Dict[Any, str]): Dict mapping item IDs to image paths
        max_seq_length (int): Maximum sequence length
        image_transform (Optional[Callable]): Transform to apply to images
    """
    
    def __init__(
        self, 
        interactions: pd.DataFrame,
        item_texts: Dict[Any, str],
        item_image_paths: Dict[Any, str],
        max_seq_length: int = 50,
        user_id_col: str = 'user_id',
        item_id_col: str = 'item_id',
        timestamp_col: str = 'timestamp',
        image_transform: Optional[Callable] = None,
        mode: str = 'next_item',  # 'next_item', 'mask', 'sequence'
        use_image_features: bool = True,
        use_text_features: bool = True,
        preload_images: bool = False
    ):
        """Initialize the multimodal sequential dataset.
        
        Args:
            interactions (pd.DataFrame): User-item interaction data with columns:
                - user_id_col: Column name for user IDs
                - item_id_col: Column name for item IDs
                - timestamp_col: Column name for timestamps
            item_texts (Dict[Any, str]): Dict mapping item IDs to text descriptions
            item_image_paths (Dict[Any, str]): Dict mapping item IDs to image paths
            max_seq_length (int): Maximum sequence length
            user_id_col (str): Column name for user IDs
            item_id_col (str): Column name for item IDs
            timestamp_col (str): Column name for timestamps
            image_transform (Optional[Callable]): Transform to apply to images
            mode (str): Training mode ('next_item', 'mask', 'sequence')
            use_image_features (bool): Whether to use image features
            use_text_features (bool): Whether to use text features
            preload_images (bool): Whether to preload images into memory
        """
        self.interactions = interactions
        self.max_seq_length = max_seq_length
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.timestamp_col = timestamp_col
        self.mode = mode
        self.item_texts = item_texts
        self.item_image_paths = item_image_paths
        self.image_transform = image_transform
        self.use_image_features = use_image_features
        self.use_text_features = use_text_features
        
        # Extract unique users and items
        self.users = interactions[user_id_col].unique()
        self.items = interactions[item_id_col].unique()
        
        # Create user-item sequence dictionary
        self.user_seq = self._create_user_sequences()
        
        # Create samples from sequences
        self.samples = []
        for user_id, seq in self.user_seq.items():
            if len(seq) > 1:  # Need at least 2 items
                self.samples.append(user_id)
        
        # Preload images if requested
        self.preloaded_images = {}
        if preload_images and self.item_image_paths:
            self._preload_images()
    
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
    
    def _preload_images(self):
        """Preload images into memory."""
        for item_id, image_path in self.item_image_paths.items():
            try:
                image = self._load_image(image_path)
                self.preloaded_images[item_id] = image
            except Exception as e:
                logging.warning(f"Failed to load image for item {item_id}: {e}")
    
    def _load_image(self, image_path: str) -> Any:
        """Load an image from path.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            Any: Loaded image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform:
                image = self.image_transform(image)
            return image
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            Dict[str, Any]: Dictionary containing sequence data
        """
        user_id = self.samples[idx]
        seq = self.user_seq[user_id]
        
        if self.mode == 'next_item':
            # Train to predict the next item
            input_seq = seq[:-1]
            target_item = seq[-1]
            
            # Create base sample
            sample = {
                'user_id': user_id,
                'input_seq': input_seq,
                'target_item': target_item
            }
            
            # Add text and image data for input sequence
            if self.use_text_features:
                sample['input_texts'] = [self.item_texts.get(item_id, "") for item_id in input_seq]
            
            if self.use_image_features:
                # Get images for input sequence
                input_images = []
                for item_id in input_seq:
                    if item_id in self.preloaded_images:
                        input_images.append(self.preloaded_images[item_id])
                    elif item_id in self.item_image_paths:
                        image = self._load_image(self.item_image_paths[item_id])
                        input_images.append(image)
                    else:
                        input_images.append(None)
                
                sample['input_images'] = input_images
            
            # Add text and image data for target item
            if self.use_text_features:
                sample['target_text'] = self.item_texts.get(target_item, "")
            
            if self.use_image_features:
                if target_item in self.preloaded_images:
                    target_image = self.preloaded_images[target_item]
                elif target_item in self.item_image_paths:
                    target_image = self._load_image(self.item_image_paths[target_item])
                else:
                    target_image = None
                
                sample['target_image'] = target_image
            
            return sample
            
        elif self.mode == 'mask':
            # BERT-style masked item prediction
            masked_seq = seq.copy()
            
            # Randomly mask some items
            masked_positions = []
            masked_items = []
            for i in range(len(masked_seq)):
                if random.random() < 0.15:  # 15% masking rate
                    masked_positions.append(i)
                    masked_items.append(masked_seq[i])
                    masked_seq[i] = None  # Mask the item
            
            # Create base sample
            sample = {
                'user_id': user_id,
                'masked_seq': masked_seq,
                'masked_positions': masked_positions,
                'masked_items': masked_items
            }
            
            # Add text and image data for sequence
            if self.use_text_features:
                # Get texts for sequence, using empty string for masked items
                seq_texts = []
                for item_id in masked_seq:
                    if item_id is not None:
                        seq_texts.append(self.item_texts.get(item_id, ""))
                    else:
                        seq_texts.append("")  # Masked item
                
                sample['seq_texts'] = seq_texts
                
                # Get texts for masked items
                masked_texts = [self.item_texts.get(item_id, "") for item_id in masked_items]
                sample['masked_texts'] = masked_texts
            
            if self.use_image_features:
                # Get images for sequence
                seq_images = []
                for item_id in masked_seq:
                    if item_id is not None:
                        if item_id in self.preloaded_images:
                            seq_images.append(self.preloaded_images[item_id])
                        elif item_id in self.item_image_paths:
                            image = self._load_image(self.item_image_paths[item_id])
                            seq_images.append(image)
                        else:
                            seq_images.append(None)
                    else:
                        seq_images.append(None)  # Masked item
                
                sample['seq_images'] = seq_images
                
                # Get images for masked items
                masked_images = []
                for item_id in masked_items:
                    if item_id in self.preloaded_images:
                        masked_images.append(self.preloaded_images[item_id])
                    elif item_id in self.item_image_paths:
                        image = self._load_image(self.item_image_paths[item_id])
                        masked_images.append(image)
                    else:
                        masked_images.append(None)
                
                sample['masked_images'] = masked_images
            
            return sample
        
        else:  # 'sequence'
            # Full sequence prediction
            
            # Create base sample
            sample = {
                'user_id': user_id,
                'sequence': seq
            }
            
            # Add text and image data for sequence
            if self.use_text_features:
                sample['seq_texts'] = [self.item_texts.get(item_id, "") for item_id in seq]
            
            if self.use_image_features:
                # Get images for sequence
                seq_images = []
                for item_id in seq:
                    if item_id in self.preloaded_images:
                        seq_images.append(self.preloaded_images[item_id])
                    elif item_id in self.item_image_paths:
                        image = self._load_image(self.item_image_paths[item_id])
                        seq_images.append(image)
                    else:
                        seq_images.append(None)
                
                sample['seq_images'] = seq_images
            
            return sample


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of multimodal samples.
    
    This function handles collating samples with text and image data.
    
    Args:
        batch (List[Dict[str, Any]]): List of samples
        
    Returns:
        Dict[str, Any]: Collated batch
    """
    # Initialize result
    result = defaultdict(list)
    
    # Collect values for each key
    for sample in batch:
        for key, value in sample.items():
            result[key].append(value)
    
    # Special handling for sequence data
    for key in list(result.keys()):
        if key in ['input_seq', 'masked_seq', 'sequence', 'masked_positions', 'masked_items']:
            # Keep as lists
            continue
        elif key in ['input_texts', 'seq_texts', 'masked_texts']:
            # Keep as lists of strings
            continue
        elif key in ['input_images', 'seq_images', 'masked_images']:
            # Keep as lists of images
            continue
        elif key in ['user_id', 'item_id', 'target_item']:
            # Convert to tensor if possible
            try:
                result[key] = torch.tensor(result[key])
            except:
                # Keep as list if conversion fails
                pass
        elif isinstance(result[key][0], torch.Tensor):
            # Stack tensors
            result[key] = torch.stack(result[key])
        elif key in ['label', 'rating']:
            # Convert to tensor
            result[key] = torch.tensor(result[key])
    
    return dict(result) 