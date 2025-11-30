"""
Multimodal fusion model for recommendation systems.

This module provides a model that combines text and image modalities
for more effective recommendation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Union, Optional
import logging

from corerec.core.base_model import BaseModel
from corerec.core.encoders import TextEncoder, VisionEncoder


class ModalityFusion(nn.Module):
    """
    Base class for modality fusion techniques.

    This class defines the interface for fusing different modalities.
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        """Initialize the modality fusion module.

        Args:
            modality_dims (Dict[str, int]): Dictionary mapping modality names to dimensions
            output_dim (int): Dimension of the fused representation
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for fusing modalities.

        Args:
            modality_embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to embeddings

        Returns:
            torch.Tensor: Fused representation
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ConcatFusion(ModalityFusion):
    """
    Simple concatenation-based fusion with projection.

    This fusion technique concatenates the embeddings from different modalities
    and projects them to the desired output dimension.
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        """Initialize the concatenation fusion module.

        Args:
            modality_dims (Dict[str, int]): Dictionary mapping modality names to dimensions
            output_dim (int): Dimension of the fused representation
        """
        super().__init__(modality_dims, output_dim)

        # Calculate total input dimension
        total_dim = sum(modality_dims.values())

        # Create projection layer
        self.projection = nn.Sequential(
            nn.Linear(total_dim, output_dim), nn.ReLU(), nn.LayerNorm(output_dim)
        )

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for concatenation fusion.

        Args:
            modality_embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to embeddings

        Returns:
            torch.Tensor: Fused representation
        """
        # Ensure all modalities are present
        for modality in self.modality_dims:
            if modality not in modality_embeddings:
                raise ValueError(f"Missing modality: {modality}")

        # Concatenate embeddings
        embeddings = []
        for modality in self.modality_dims:
            embeddings.append(modality_embeddings[modality])

        concatenated = torch.cat(embeddings, dim=-1)

        # Project to output dimension
        return self.projection(concatenated)


class AttentionFusion(ModalityFusion):
    """
    Attention-based fusion of modalities.

    This fusion technique uses attention mechanisms to dynamically weight
    and combine different modalities.
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        """Initialize the attention fusion module.

        Args:
            modality_dims (Dict[str, int]): Dictionary mapping modality names to dimensions
            output_dim (int): Dimension of the fused representation
        """
        super().__init__(modality_dims, output_dim)

        # Create projection layers for each modality
        self.projections = nn.ModuleDict(
            {modality: nn.Linear(dim, output_dim) for modality, dim in modality_dims.items()}
        )

        # Create attention query vector
        self.query = nn.Parameter(torch.randn(output_dim))

        # Create attention layer
        self.attention = nn.Sequential(nn.Linear(output_dim, 64), nn.Tanh(), nn.Linear(64, 1))

        # Create output normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for attention fusion.

        Args:
            modality_embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to embeddings

        Returns:
            torch.Tensor: Fused representation
        """
        # Ensure all modalities are present
        for modality in self.modality_dims:
            if modality not in modality_embeddings:
                raise ValueError(f"Missing modality: {modality}")

        # Project each modality to the same dimension
        projected_embeddings = {}
        for modality, embedding in modality_embeddings.items():
            projected_embeddings[modality] = self.projections[modality](embedding)

        # Stack the projected embeddings
        stacked = torch.stack(
            [emb for emb in projected_embeddings.values()], dim=1
        )  # [batch_size, num_modalities, output_dim]

        # Compute attention scores
        attention_scores = self.attention(stacked).squeeze(-1)  # [batch_size, num_modalities]
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(
            -1
        )  # [batch_size, num_modalities, 1]

        # Apply attention weights
        weighted_embeddings = stacked * attention_weights

        # Sum the weighted embeddings
        fused = weighted_embeddings.sum(dim=1)  # [batch_size, output_dim]

        # Apply normalization
        return self.norm(fused)


class GatingFusion(ModalityFusion):
    """
    Gating-based fusion of modalities.

    This fusion technique uses gating mechanisms to control the flow of information
    from different modalities.
    """

    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        """Initialize the gating fusion module.

        Args:
            modality_dims (Dict[str, int]): Dictionary mapping modality names to dimensions
            output_dim (int): Dimension of the fused representation
        """
        super().__init__(modality_dims, output_dim)

        # Create projection layers for each modality
        self.projections = nn.ModuleDict(
            {modality: nn.Linear(dim, output_dim) for modality, dim in modality_dims.items()}
        )

        # Create gating networks for each modality
        self.gates = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, output_dim), nn.Sigmoid()
                )
                for modality, dim in modality_dims.items()
            }
        )

        # Create output normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for gating fusion.

        Args:
            modality_embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to embeddings

        Returns:
            torch.Tensor: Fused representation
        """
        # Ensure all modalities are present
        for modality in self.modality_dims:
            if modality not in modality_embeddings:
                raise ValueError(f"Missing modality: {modality}")

        # Project each modality to the same dimension
        projected_embeddings = {}
        for modality, embedding in modality_embeddings.items():
            projected_embeddings[modality] = self.projections[modality](embedding)

        # Compute gates for each modality
        gates = {}
        for modality, embedding in modality_embeddings.items():
            gates[modality] = self.gates[modality](embedding)

        # Apply gates to projected embeddings
        gated_embeddings = {}
        for modality in self.modality_dims:
            gated_embeddings[modality] = projected_embeddings[modality] * gates[modality]

        # Sum the gated embeddings
        fused = sum(gated_embeddings.values())

        # Apply normalization
        return self.norm(fused)


class MultimodalFusionModel(BaseModel):
    """
    Multimodal fusion model for recommendation.

    This model combines text and image modalities for recommendation.

    Attributes:
        name (str): Name of the model
        config (Dict[str, Any]): Model configuration
        text_encoder (TextEncoder): Encoder for text data
        vision_encoder (VisionEncoder): Encoder for image data
        fusion (ModalityFusion): Fusion module for combining modalities
        prediction_head (nn.Module): Prediction head for final output
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the multimodal fusion model.

        Args:
            name (str): Name of the model
            config (Dict[str, Any]): Model configuration including:
                - text_encoder_config (Dict[str, Any]): Configuration for the text encoder
                - vision_encoder_config (Dict[str, Any]): Configuration for the vision encoder
                - fusion_type (str): Type of fusion to use ('concat', 'attention', 'gating')
                - fusion_output_dim (int): Dimension of the fused representation
                - prediction_hidden_dims (List[int]): Hidden dimensions for the prediction head
        """
        super().__init__(name, config)

        # Create text encoder
        text_encoder_config = config.get("text_encoder_config", {})
        self.text_encoder = TextEncoder(name=f"{name}_text_encoder", config=text_encoder_config)

        # Create vision encoder
        vision_encoder_config = config.get("vision_encoder_config", {})
        self.vision_encoder = VisionEncoder(
            name=f"{name}_vision_encoder", config=vision_encoder_config
        )

        # Get dimensions
        text_dim = text_encoder_config.get("embedding_dim", 768)
        vision_dim = vision_encoder_config.get("embedding_dim", 768)

        # Create fusion module
        fusion_type = config.get("fusion_type", "concat").lower()
        fusion_output_dim = config.get("fusion_output_dim", 512)

        modality_dims = {"text": text_dim, "vision": vision_dim}

        if fusion_type == "concat":
            self.fusion = ConcatFusion(modality_dims, fusion_output_dim)
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(modality_dims, fusion_output_dim)
        elif fusion_type == "gating":
            self.fusion = GatingFusion(modality_dims, fusion_output_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        # Create prediction head
        prediction_hidden_dims = config.get("prediction_hidden_dims", [256, 128])

        layers = []
        input_dim = fusion_output_dim

        for hidden_dim in prediction_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.prediction_head = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass of the multimodal fusion model.

        Args:
            batch (Dict[str, Any]): Batch of data including:
                - text (List[str] or torch.Tensor): Text data
                - images: Image data

        Returns:
            torch.Tensor: Prediction scores
        """
        # Check if batch has all required modalities
        has_text = "text" in batch or "text_features" in batch
        has_vision = "images" in batch or "image_features" in batch

        if not has_text and not has_vision:
            raise ValueError("Batch must contain at least one modality (text or images)")

        # Encode modalities
        modality_embeddings = {}

        # Text encoding
        if has_text:
            if "text_features" in batch:
                # Use provided features
                modality_embeddings["text"] = batch["text_features"]
            else:
                # Encode text
                modality_embeddings["text"] = self.text_encoder.encode(batch["text"])

        # Vision encoding
        if has_vision:
            if "image_features" in batch:
                # Use provided features
                modality_embeddings["vision"] = batch["image_features"]
            else:
                # Encode images
                modality_embeddings["vision"] = self.vision_encoder.encode(batch["images"])

        # Handle missing modalities
        if "text" not in modality_embeddings:
            # Create zero embeddings for text
            batch_size = modality_embeddings["vision"].shape[0]
            text_dim = self.fusion.modality_dims["text"]
            modality_embeddings["text"] = torch.zeros(batch_size, text_dim, device=self.device)

        if "vision" not in modality_embeddings:
            # Create zero embeddings for vision
            batch_size = modality_embeddings["text"].shape[0]
            vision_dim = self.fusion.modality_dims["vision"]
            modality_embeddings["vision"] = torch.zeros(batch_size, vision_dim, device=self.device)

        # Fuse modalities
        fused = self.fusion(modality_embeddings)

        # Apply prediction head
        return self.prediction_head(fused).squeeze(-1)

    def train_step(
        self, batch: Dict[str, Any], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform a single training step.

        Args:
            batch (Dict[str, Any]): Batch of data including:
                - text (List[str] or torch.Tensor): Text data
                - images: Image data
                - labels (torch.Tensor): Target labels
            optimizer (torch.optim.Optimizer): Optimizer instance

        Returns:
            Dict[str, float]: Dictionary with loss values
        """
        # Extract labels
        labels = batch.pop("labels")

        # Forward pass
        predictions = self.forward(batch)

        # Compute loss
        loss = nn.BCEWithLogitsLoss()(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    def validate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch (Dict[str, Any]): Batch of data including:
                - text (List[str] or torch.Tensor): Text data
                - images: Image data
                - labels (torch.Tensor): Target labels

        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        # Extract labels
        labels = batch.pop("labels")

        # Forward pass
        with torch.no_grad():
            predictions = self.forward(batch)

            # Compute loss
            loss = nn.BCEWithLogitsLoss()(predictions, labels)

            # Compute accuracy
            binary_predictions = (torch.sigmoid(predictions) > 0.5).float()
            accuracy = (binary_predictions == labels).float().mean()

        return {"val_loss": loss.item(), "val_accuracy": accuracy.item()}

    @property
    def device(self) -> torch.device:
        """Get the device of the model.

        Returns:
            torch.device: Device of the model
        """
        return next(self.parameters()).device
