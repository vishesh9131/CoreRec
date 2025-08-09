"""
Transformer tower for recommendation systems.

This module provides a transformer-based tower for recommendation systems,
which can encode text, sequences, or other data using transformer architectures.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
import logging
import os
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,    #initionally not used
    PreTrainedTokenizer #initionally not used
)

from corerec.towers.base_tower import AbstractTower


class TransformerTower(AbstractTower):
    """
    Transformer-based tower for recommendation systems.
    
    This tower uses transformer models like BERT, RoBERTa, or other pre-trained
    transformers to encode text, sequences, or other data.
    
    Attributes:
        name (str): Name of the tower
        config (Dict[str, Any]): Tower configuration
        model (PreTrainedModel): Pre-trained transformer model
        tokenizer (PreTrainedTokenizer): Tokenizer for the transformer model
        pooling_strategy (str): Strategy for pooling the transformer outputs
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the transformer tower.
        
        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration including:
                - model_name (str): Name or path of the pre-trained model
                - pooling_strategy (str): Strategy for pooling ('cls', 'mean', 'max')
                - max_length (int): Maximum sequence length
                - output_dim (int): Output dimension
                - trainable (bool): Whether to fine-tune the transformer
        """
        super().__init__(name, config)
        
        # Get configuration
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.pooling_strategy = config.get('pooling_strategy', 'cls')
        self.max_length = config.get('max_length', 128)
        self.output_dim = config.get('output_dim', 768)
        self.trainable = config.get('trainable', False)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configure model
        model_config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, config=model_config)
        
        # Freeze model if not trainable
        if not self.trainable:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get model dimension
        self.model_dim = self.model.config.hidden_size
        
        # Create projection layer if needed
        if self.model_dim != self.output_dim:
            self.projection = nn.Linear(self.model_dim, self.output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the transformer tower.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs including:
                - input_ids (torch.Tensor): Token IDs
                - attention_mask (torch.Tensor): Attention mask
                - token_type_ids (torch.Tensor): Token type IDs (for BERT)
            
        Returns:
            torch.Tensor: Encoded representations
        """
        # Check if inputs are already tokenized
        if 'input_ids' in inputs and 'attention_mask' in inputs:
            # Use provided inputs
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs.get('token_type_ids', None)
        elif 'text' in inputs:
            # Tokenize text
            tokenized = self.tokenizer(
                inputs['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to the same device as the model
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Extract inputs
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            token_type_ids = tokenized.get('token_type_ids', None)
        else:
            raise ValueError("Inputs must contain either 'input_ids' and 'attention_mask' or 'text'")
        
        # Forward pass through transformer
        if token_type_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        if self.pooling_strategy == 'cls':
            # Use CLS token
            pooled = hidden_states[:, 0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling
            pooled = self._mean_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == 'max':
            # Max pooling
            pooled = self._max_pooling(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        # Apply projection
        embeddings = self.projection(pooled)
        
        return embeddings
    
    def _mean_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from transformer
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Mean-pooled representation
        """
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Apply mask and compute mean
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def _max_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply max pooling to hidden states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from transformer
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Max-pooled representation
        """
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Apply mask
        hidden_states[input_mask_expanded == 0] = -1e9
        
        # Max pooling
        max_embeddings, _ = torch.max(hidden_states, dim=1)
        
        return max_embeddings
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode a batch of texts.
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to encode
            
        Returns:
            torch.Tensor: Encoded representations
        """
        # Set model to eval mode
        self.model.eval()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        
        # Forward pass
        with torch.no_grad():
            embeddings = self.forward(tokenized)
        
        return embeddings
    
    def save(self, path: str):
        """Save the tower.
        
        Args:
            path (str): Path to save the tower
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save_pretrained(f"{path}_model")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{path}_tokenizer")
        
        # Save config
        torch.save({
            'name': self.name,
            'config': self.config,
            'model_path': f"{path}_model",
            'tokenizer_path': f"{path}_tokenizer",
            'projection': self.projection.state_dict() if not isinstance(self.projection, nn.Identity) else None
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'TransformerTower':
        """Load the tower.
        
        Args:
            path (str): Path to load the tower from
            
        Returns:
            TransformerTower: Loaded tower
        """
        # Load config
        checkpoint = torch.load(path, map_location='cpu')
        
        # Create model with loaded config
        model = cls(
            name=checkpoint['name'],
            config=checkpoint['config']
        )
        
        # Load model
        model.model = AutoModel.from_pretrained(checkpoint['model_path'])
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_path'])
        
        # Load projection
        if checkpoint['projection'] is not None:
            model.projection.load_state_dict(checkpoint['projection'])
        
        return model
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model.
        
        Returns:
            torch.device: Device of the model
        """
        return next(self.model.parameters()).device


class BERTTower(TransformerTower):
    """
    BERT-based tower for recommendation systems.
    
    This is a specialized transformer tower using BERT.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the BERT tower.
        
        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration
        """
        # Set default model name if not provided
        if 'model_name' not in config:
            config['model_name'] = 'bert-base-uncased'
        
        super().__init__(name, config)


class RoBERTaTower(TransformerTower):
    """
    RoBERTa-based tower for recommendation systems.
    
    This is a specialized transformer tower using RoBERTa.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the RoBERTa tower.
        
        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration
        """
        # Set default model name if not provided
        if 'model_name' not in config:
            config['model_name'] = 'roberta-base'
        
        super().__init__(name, config)


class T5Tower(TransformerTower):
    """
    T5-based tower for recommendation systems.
    
    This is a specialized transformer tower using T5.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the T5 tower.
        
        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration
        """
        # Set default model name if not provided
        if 'model_name' not in config:
            config['model_name'] = 't5-base'
        
        super().__init__(name, config)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the T5 tower.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs
            
        Returns:
            torch.Tensor: Encoded representations
        """
        # Check if inputs are already tokenized
        if 'input_ids' in inputs and 'attention_mask' in inputs:
            # Use provided inputs
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        elif 'text' in inputs:
            # Tokenize text
            tokenized = self.tokenizer(
                inputs['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to the same device as the model
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            
            # Extract inputs
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        else:
            raise ValueError("Inputs must contain either 'input_ids' and 'attention_mask' or 'text'")
        
        # Forward pass through transformer
        outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        if self.pooling_strategy == 'mean':
            # Mean pooling
            pooled = self._mean_pooling(hidden_states, attention_mask)
        elif self.pooling_strategy == 'max':
            # Max pooling
            pooled = self._max_pooling(hidden_states, attention_mask)
        else:
            # Use first token (T5 doesn't have a CLS token, so use first token)
            pooled = hidden_states[:, 0]
        
        # Apply projection
        embeddings = self.projection(pooled)
        
        return embeddings 