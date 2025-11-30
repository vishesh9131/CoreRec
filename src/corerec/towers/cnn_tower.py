"""
CNN tower for recommendation systems.

This module provides a CNN tower for encoding image features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional
import logging
import numpy as np
from PIL import Image
from torchvision import models, transforms

from corerec.towers.base_tower import AbstractTower


class CNNTower(AbstractTower):
    """
    Convolutional Neural Network (CNN) tower for recommendation systems.

    This tower uses a CNN to encode image features into a fixed-dimension
    representation for recommendation.

    Architecture:
        Image Input
              ↓
        [CNN Backbone]
              ↓
         [Pooling/GAP]
              ↓
        [Dense Layers]
              ↓
        Encoded Representation
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the CNN tower.

        Args:
            name (str): Name of the tower
            config (Dict[str, Any]): Tower configuration including:
                - backbone (str): CNN backbone ('resnet18', 'resnet50', 'efficientnet', 'mobilenet', etc.)
                - pretrained (bool): Whether to use pretrained weights
                - freeze_backbone (bool): Whether to freeze backbone weights
                - output_dim (int): Output dimension
                - pooling (str): Pooling strategy ('avg', 'max', 'attention')
                - hidden_dims (List[int]): Dimensions of hidden layers after pooling
                - image_size (int): Size to resize images to
                - use_augmentation (bool): Whether to use data augmentation during training
        """
        super().__init__(name, config)

        # Get configuration
        self.backbone_name = config.get("backbone", "resnet18")
        self.pretrained = config.get("pretrained", True)
        self.freeze_backbone = config.get("freeze_backbone", True)
        self.output_dim = config.get("output_dim", 512)
        self.pooling = config.get("pooling", "avg")
        self.hidden_dims = config.get("hidden_dims", [])
        self.image_size = config.get("image_size", 224)
        self.use_augmentation = config.get("use_augmentation", False)

        # Create CNN backbone
        self.backbone, self.backbone_dim = self._create_backbone()

        # Freeze backbone if needed
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create pooling layer
        if self.pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif self.pooling == "attention":
            self.pool = AttentionPool2d(self.backbone_dim)
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")

        # Create projection layers
        if not self.hidden_dims:
            # Simple projection
            if self.backbone_dim != self.output_dim:
                self.projection = nn.Linear(self.backbone_dim, self.output_dim)
            else:
                self.projection = nn.Identity()
        else:
            # MLP projection
            layers = []
            prev_dim = self.backbone_dim

            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
                prev_dim = hidden_dim

            # Final layer
            layers.append(nn.Linear(prev_dim, self.output_dim))

            self.projection = nn.Sequential(*layers)

        # Create image transforms
        self.transform = self._create_transforms()

    def _create_backbone(self) -> Tuple[nn.Module, int]:
        """Create CNN backbone.

        Returns:
            Tuple[nn.Module, int]: CNN backbone and its output dimension
        """
        if self.backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=self.pretrained)
            backbone_dim = 512
            backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc
        elif self.backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=self.pretrained)
            backbone_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc
        elif self.backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=self.pretrained)
            backbone_dim = 1280
            backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove classifier
        elif self.backbone_name == "mobilenet_v2":
            backbone = models.mobilenet_v2(pretrained=self.pretrained)
            backbone_dim = 1280
            backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove classifier
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")

        return backbone, backbone_dim

    def _create_transforms(self) -> transforms.Compose:
        """Create image transforms.

        Returns:
            transforms.Compose: Image transforms
        """
        # Create transforms
        if self.use_augmentation:
            # With augmentation for training
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            # Without augmentation for inference
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the CNN tower.

        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs including:
                - images (torch.Tensor): Image tensors [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Encoded representations
        """
        # Get images from inputs
        if "images" in inputs:
            images = inputs["images"]
        else:
            raise ValueError("Inputs must contain 'images'")

        # Pass through CNN backbone
        features = self.backbone(images)

        # Apply pooling
        if self.pooling == "attention":
            pooled = self.pool(features)
        else:
            pooled = self.pool(features).squeeze(-1).squeeze(-1)

        # Apply projection
        encoded = self.projection(pooled)

        return encoded

    def encode(self, images: Union[List[Image.Image], Image.Image, torch.Tensor]) -> torch.Tensor:
        """Encode images.

        Args:
            images (Union[List[Image.Image], Image.Image, torch.Tensor]): Images to encode
                - If List[Image.Image]: List of PIL images
                - If Image.Image: Single PIL image
                - If torch.Tensor: Already processed tensor [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Encoded representations
        """
        # Set to eval mode
        self.eval()

        # Process the input
        if isinstance(images, torch.Tensor):
            # Already processed tensor
            processed_images = images.to(self.device)
        elif isinstance(images, Image.Image):
            # Single PIL image
            processed_image = self.transform(images).unsqueeze(0)
            processed_images = processed_image.to(self.device)
        elif isinstance(images, list):
            # List of PIL images
            processed_images = []
            for img in images:
                if img is not None:
                    processed_image = self.transform(img)
                else:
                    # Create a blank image if None
                    processed_image = torch.zeros(3, self.image_size, self.image_size)
                processed_images.append(processed_image)
            processed_images = torch.stack(processed_images).to(self.device)
        else:
            raise ValueError(
                "Unsupported input type. Must be torch.Tensor, PIL.Image, or List[PIL.Image]"
            )

        # Forward pass
        with torch.no_grad():
            encoded = self.forward({"images": processed_images})

        return encoded


class AttentionPool2d(nn.Module):
    """
    Attention-based pooling for 2D feature maps.

    This pooling method uses attention to weight different spatial locations
    in the feature map, allowing the model to focus on the most important parts.
    """

    def __init__(self, in_channels: int):
        """Initialize the attention pooling.

        Args:
            in_channels (int): Number of input channels
        """
        super().__init__()

        # Create query vector
        self.query = nn.Parameter(torch.randn(in_channels))

        # Create attention layers
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Layer norm
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention pooling.

        Args:
            x (torch.Tensor): Input feature map [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Pooled feature vector [batch_size, channels]
        """
        batch_size, channels, height, width = x.size()

        # Create key and value
        key = self.key_conv(x).view(
            batch_size, channels, -1
        )  # [batch_size, channels, height*width]
        value = self.value_conv(x).view(
            batch_size, channels, -1
        )  # [batch_size, channels, height*width]

        # Attention mechanism
        query = (
            self.query.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        )  # [batch_size, channels, 1]
        attn_weights = torch.bmm(key.transpose(1, 2), query)  # [batch_size, height*width, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, height*width, 1]

        # Apply attention weights
        weighted = torch.bmm(value, attn_weights)  # [batch_size, channels, 1]
        weighted = weighted.squeeze(-1)  # [batch_size, channels]

        # Apply normalization
        pooled = self.norm(weighted)

        return pooled
