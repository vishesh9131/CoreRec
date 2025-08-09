"""
Encoder module for CoreRec framework.

This module contains the AbstractEncoder class and various concrete encoder
implementations like TextEncoder and VisionEncoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List, Tuple
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoFeatureExtractor

class AbstractEncoder(nn.Module, ABC):
    """
    Abstract encoder class for CoreRec framework.
    
    This class defines the interface for all encoders in the framework.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the abstract encoder.
        
        Args:
            name (str): Name of the encoder
            config (Dict[str, Any]): Encoder configuration
        """
        super().__init__()
        self.name = name
        self.config = config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Encoded representation
        """
        pass
    
    @abstractmethod
    def encode(self, input_data: Any) -> torch.Tensor:
        """Encode input data into a tensor representation.
        
        Args:
            input_data (Any): Input data to encode
            
        Returns:
            torch.Tensor: Encoded representation
        """
        pass


class TextEncoder(AbstractEncoder):
    """
    Text encoder using transformer models from HuggingFace.
    
    This encoder uses pre-trained language models to encode text data.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the text encoder.
        
        Args:
            name (str): Name of the encoder
            config (Dict[str, Any]): Encoder configuration including:
                - model_name (str): Name of the HuggingFace model
                - pooling (str): Pooling strategy ('cls', 'mean', 'max')
                - max_length (int): Maximum sequence length
                - trainable (bool): Whether to fine-tune the model
        """
        super().__init__(name, config)
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.pooling = config.get('pooling', 'cls')
        self.max_length = config.get('max_length', 512)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Freeze model if not trainable
        if not config.get('trainable', False):
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the text encoder.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            torch.Tensor: Encoded text representation
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply pooling
        if self.pooling == 'cls':
            # Use CLS token representation
            return outputs.last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            return sum_embeddings / sum_mask
        elif self.pooling == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative
            return torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text input into a tensor representation.
        
        Args:
            texts (Union[str, List[str]]): Text or list of texts to encode
            
        Returns:
            torch.Tensor: Encoded text representation
        """
        # Handle single text by wrapping it in a list
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            max_length=self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.forward(inputs['input_ids'], inputs['attention_mask'])
            
        return embeddings


class VisionEncoder(AbstractEncoder):
    """
    Vision encoder using transformer models from HuggingFace.
    
    This encoder uses pre-trained vision models to encode image data.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the vision encoder.
        
        Args:
            name (str): Name of the encoder
            config (Dict[str, Any]): Encoder configuration including:
                - model_name (str): Name of the HuggingFace model
                - image_size (int): Image size for resizing
                - trainable (bool): Whether to fine-tune the model
        """
        super().__init__(name, config)
        
        self.model_name = config.get('model_name', 'google/vit-base-patch16-224')
        self.image_size = config.get('image_size', 224)
        
        # Load model and processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        except Exception:
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Freeze model if not trainable
        if not config.get('trainable', False):
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass of the vision encoder.
        
        Args:
            pixel_values (torch.Tensor): Processed image pixel values
            
        Returns:
            torch.Tensor: Encoded image representation
        """
        outputs = self.model(pixel_values=pixel_values)
        
        # Get the pooled output (CLS token or global pool)
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        else:
            # Fallback to the CLS token for models without pooler
            return outputs.last_hidden_state[:, 0]
    
    def encode(self, images: Any) -> torch.Tensor:
        """Encode image input into a tensor representation.
        
        Args:
            images: Image or list of images to encode
            
        Returns:
            torch.Tensor: Encoded image representation
        """
        # Process images
        inputs = self.processor(images=images, return_tensors='pt')
        
        # Move inputs to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.forward(inputs['pixel_values'])
            
        return embeddings 