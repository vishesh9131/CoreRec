from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from corerec.api.base_recommender import BaseRecommender


class AttentionalGRU(nn.Module):
    """
    Attentional GRU Cell proposed in DIEN.

    Architecture:

    ┌───────────┐       ┌───────────┐
    │  Hidden   │       │ Attention │
    │  States   │       │  Scores   │
    └─────┬─────┘       └─────┬─────┘
          │                   │
          └────────┬──────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Attentional GRU  │
          │      Cell        │
          └────────┬─────────┘
                   │
                   ▼
            ┌─────────────┐
            │  Output     │
            └─────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize Attentional GRU Cell.

        Args:
            input_size: Input dimension.
            hidden_size: Hidden dimension.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset gate
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # Update gate
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        # Candidate activation
        self.new_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Initialize weights
        nn.init.xavier_normal_(self.reset_gate.weight)
        nn.init.xavier_normal_(self.update_gate.weight)
        nn.init.xavier_normal_(self.new_gate.weight)

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor, att_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Attentional GRU Cell.

        Args:
            inputs: Input tensor of shape (batch_size, input_size).
            hidden: Hidden state tensor of shape (batch_size, hidden_size).
            att_score: Attention score of shape (batch_size, 1).

        Returns:
            Output tensor of shape (batch_size, hidden_size).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Concatenate inputs and hidden state
        combined = torch.cat([inputs, hidden], dim=1)

        # Reset gate
        r = torch.sigmoid(self.reset_gate(combined))
        # Update gate
        z = torch.sigmoid(self.update_gate(combined)) * att_score
        # Candidate activation
        combined_r = torch.cat([inputs, r * hidden], dim=1)
        n = torch.tanh(self.new_gate(combined_r))

        # Compute output
        output = (1 - z) * hidden + z * n

        return output


class InterestExtractionLayer(nn.Module):
    """
    Interest Extraction Layer for DIEN.

    Architecture:

    ┌───────────┐
    │  Inputs   │
    └─────┬─────┘
          │
          ▼
    ┌─────────────┐
    │     GRU     │
    └──────┬──────┘
           │
           ▼
     ┌───────────┐
     │  Outputs  │
     └───────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize Interest Extraction Layer.

        Args:
            input_size: Input dimension.
            hidden_size: Hidden dimension.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Interest Extraction Layer.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_size).
            lengths: Sequence lengths of shape (batch_size,).

        Returns:
            hidden_states: All hidden states of shape (batch_size, seq_len, hidden_size).
            final_state: Final hidden state of shape (batch_size, hidden_size).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Pack padded sequence
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Forward pass through GRU
        outputs, final_state = self.gru(packed_inputs)

        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, final_state.squeeze(0)


class InterestEvolutionLayer(nn.Module):
    """
    Interest Evolution Layer for DIEN.

    Architecture:

    ┌───────────┐   ┌───────────┐
    │  Hidden   │   │  Target   │
    │  States   │   │  Item     │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
                  ▼
           ┌─────────────┐
           │  Attention  │
           │   Layer     │
           └──────┬──────┘
                  │
                  ▼
          ┌───────────────┐
          │ Attentional   │
          │     GRU       │
          └───────┬───────┘
                  │
                  ▼
         ┌─────────────────┐
    ┌────┤ Final Hidden    │
    │    └─────────────────┘
    │
    ▼
    Output

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, input_size: int, hidden_size: int, attention_size: int):
        """
        Initialize Interest Evolution Layer.

        Args:
            input_size: Input dimension.
            hidden_size: Hidden dimension.
            attention_size: Attention dimension.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size + hidden_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1, bias=False),
        )

        self.agru = AttentionalGRU(input_size, hidden_size)

        # Initialize attention weights
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(
        self, hidden_states: torch.Tensor, target_item: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through Interest Evolution Layer.

        Args:
            hidden_states: Hidden states from Interest Extraction Layer of shape (batch_size, seq_len, hidden_size).
            target_item: Target item embeddings of shape (batch_size, input_size).
            lengths: Sequence lengths of shape (batch_size,).

        Returns:
            Output hidden state of shape (batch_size, hidden_size).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Expand target item embedding to match sequence length
        expanded_target = target_item.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate hidden states and target item
        combined = torch.cat([hidden_states, expanded_target], dim=2)

        # Calculate attention scores
        att_scores = self.attention(combined).squeeze(-1)

        # Create mask based on lengths
        mask = torch.arange(seq_len, device=hidden_states.device).expand(
            batch_size, seq_len
        ) < lengths.unsqueeze(1)

        # Apply mask to attention scores
        att_scores = att_scores.masked_fill(~mask, -1e9)

        # Apply softmax to get normalized attention weights
        att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)

        # Initialize final hidden state with zeros
        h_final = torch.zeros(batch_size, hidden_size, device=hidden_states.device)

        # Process sequence through AGRU
        for i in range(seq_len):
            # Get slice of hidden states and attention weights
            h_i = hidden_states[:, i, :]
            a_i = att_weights[:, i, :]

            # Update hidden state using AGRU
            h_final = self.agru(h_i, h_final, a_i)

        return h_final


class DINAttention(nn.Module):
    """
    Deep Interest Network Attention mechanism.

    Architecture:

    ┌───────────┐   ┌───────────┐
    │ Candidate │   │   User    │
    │   Item    │   │ Behaviors │
    └─────┬─────┘   └─────┬─────┘
          │               │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │ Attention Net │
          └───────┬───────┘
                  │
                  ▼
         ┌─────────────────┐
    ┌────┤ Weighted Sum    │
    │    └─────────────────┘
    │
    ▼
    Output

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, embed_dim: int, attention_units: List[int]):
        """
        Initialize DIN Attention.

        Args:
            embed_dim: Dimension of embeddings.
            attention_units: List of attention network unit sizes.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.attention_layers = nn.ModuleList()

        # Input dimension: target + behavior + element-wise product
        input_dim = embed_dim * 3

        # Build attention network
        for i, units in enumerate(attention_units):
            self.attention_layers.append(nn.Linear(input_dim, units))
            self.attention_layers.append(nn.ReLU())
            input_dim = units

        # Output layer
        self.attention_layers.append(nn.Linear(input_dim, 1))

        # Initialize weights
        for layer in self.attention_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(
        self, target_item: torch.Tensor, behavior_items: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DIN Attention.

        Args:
            target_item: Target item embeddings of shape (batch_size, embed_dim).
            behavior_items: User behavior item embeddings of shape (batch_size, seq_len, embed_dim).
            mask: Mask tensor of shape (batch_size, seq_len).

        Returns:
            Weighted sum of behavior embeddings of shape (batch_size, embed_dim).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        batch_size, seq_len, embed_dim = behavior_items.size()

        # Expand target item to match behavior sequence
        target_expand = target_item.unsqueeze(1).expand(-1, seq_len, -1)

        # Element-wise product
        prod = target_expand * behavior_items

        # Concatenate features
        concat = torch.cat([target_expand, behavior_items, prod], dim=2)

        # Calculate attention scores
        for layer in self.attention_layers:
            concat = layer(concat)

        # Squeeze last dimension
        attention_scores = concat.squeeze(-1)

        # Apply mask
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)

        # Weighted sum
        weighted_sum = (attention_weights * behavior_items).sum(dim=1)

        return weighted_sum


class DIENModel(nn.Module):
    """
    Deep Interest Evolution Network (DIEN) model.

    Architecture:

                 ┌───────────────┐
                 │   Embedding   │
                 │     Layer     │
                 └───────┬───────┘
                         │
                         ▼
    ┌───────────┐   ┌─────────────┐   ┌───────────┐
    │ Behavior  │   │  Interest   │   │  Target   │
    │ Sequence  │──▶│ Extraction  │   │   Item    │
    └───────────┘   └──────┬──────┘   └─────┬─────┘
                           │                 │
                           ▼                 │
                   ┌───────────────┐         │
                   │   Interest    │         │
                   │  Evolution    │◀────────┘
                   └───────┬───────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Concat with │
                    │ Other Feats │
                    └──────┬──────┘
                           │
                           ▼
                     ┌───────────┐
                     │    MLP    │
                     └─────┬─────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Output    │
                    └─────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.1,
        attention_dims: List[int] = [64, 32],
        gru_hidden_dim: int = 64,
        aux_loss_weight: float = 0.2,
    ):
        """
        Initialize DIEN model.

        Args:
            field_dims: List of field dimensions.
            embed_dim: Dimension of embeddings.
            mlp_dims: List of MLP layer dimensions.
            dropout: Dropout rate.
            attention_dims: Attention network dimensions.
            gru_hidden_dim: GRU hidden dimension.
            aux_loss_weight: Auxiliary loss weight.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.aux_loss_weight = aux_loss_weight

        # Embedding layers
        self.embedding = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in field_dims])

        # Embedding for numerical features
        self.numerical_embedding = nn.Linear(1, embed_dim)

        # Interest extraction layer
        self.interest_extractor = InterestExtractionLayer(embed_dim, gru_hidden_dim)

        # Interest evolution layer
        self.interest_evolution = InterestEvolutionLayer(
            embed_dim, gru_hidden_dim, attention_dims[0]
        )

        # Auxiliary loss network (next item prediction)
        self.aux_net = nn.Linear(gru_hidden_dim, embed_dim)

        # DIN attention layer for behavioral pooling
        self.din_attention = DINAttention(embed_dim, attention_dims)

        # MLP layers
        self.mlp = nn.ModuleList()
        # Input: evolved_state (gru_hidden_dim) + target_item_embed (embed_dim) + other_features (variable)
        # For now, assume other_features is embed_dim (will be adjusted based on actual features)
        # If there are additional fields/numerical features, they need to be accounted for
        input_dim = gru_hidden_dim + embed_dim + embed_dim  # Interest + target item + other features (default to embed_dim)

        for dim in mlp_dims:
            self.mlp.append(nn.Linear(input_dim, dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(dropout))
            input_dim = dim

        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize embeddings
        for embedding in self.embedding:
            nn.init.xavier_normal_(embedding.weight)

        # Initialize MLP weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.xavier_normal_(self.numerical_embedding.weight)
        nn.init.xavier_normal_(self.aux_net.weight)

    def forward(
        self,
        categorical_input: torch.Tensor,
        seq_input: torch.Tensor,
        seq_lengths: torch.Tensor,
        numerical_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through DIEN model.

        Args:
            categorical_input: Categorical input of shape (batch_size, num_fields).
            seq_input: Sequence input of shape (batch_size, seq_len).
            seq_lengths: Sequence lengths of shape (batch_size,).
            numerical_input: Numerical input of shape (batch_size, num_numerical).

        Returns:
            output: Predicted probability.
            aux_loss: Auxiliary loss (optional, during training).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings for regular fields (user, item, etc.)
        field_embeds = []
        for i, emb in enumerate(self.embedding):
            field_embeds.append(emb(categorical_input[:, i]))

        # Get embeddings for numerical features
        numerical_embeds = []
        for i in range(numerical_input.size(1)):
            numerical_embeds.append(self.numerical_embedding(numerical_input[:, i].unsqueeze(1)))

        # Split embeddings
        user_embed = field_embeds[0]
        target_item_embed = field_embeds[1]
        
        # Other features: remaining categorical fields (if any) + numerical features
        # Exclude user (index 0) and target_item (index 1) from field_embeds
        other_field_embeds = field_embeds[2:] if len(field_embeds) > 2 else []
        other_numerical_embeds = numerical_embeds
        
        # Combine other features (excluding user and target_item)
        if other_field_embeds and other_numerical_embeds:
            other_features_raw = torch.cat(other_field_embeds + other_numerical_embeds, dim=1)
        elif other_field_embeds:
            other_features_raw = torch.cat(other_field_embeds, dim=1)
        elif other_numerical_embeds:
            other_features_raw = torch.cat(other_numerical_embeds, dim=1)
        else:
            # No other features, use zero tensor
            other_features_raw = torch.zeros(categorical_input.size(0), self.embed_dim, 
                                            device=categorical_input.device)
        
        # Project other_features to embed_dim to match MLP input size
        if other_features_raw.size(1) != self.embed_dim:
            if other_features_raw.size(1) > self.embed_dim:
                # Too large: use mean pooling to reduce to embed_dim
                # Reshape to (batch, num_features, embed_dim) and take mean
                num_features = other_features_raw.size(1) // self.embed_dim
                remainder = other_features_raw.size(1) % self.embed_dim
                if remainder == 0 and num_features > 0:
                    # Can evenly divide, use mean pooling
                    other_features = other_features_raw.view(
                        categorical_input.size(0), num_features, self.embed_dim
                    ).mean(dim=1)
                else:
                    # Use linear projection
                    if not hasattr(self, 'other_features_proj'):
                        self.other_features_proj = nn.Linear(
                            other_features_raw.size(1), self.embed_dim
                        ).to(other_features_raw.device)
                        # Initialize weights
                        nn.init.xavier_normal_(self.other_features_proj.weight)
                        if self.other_features_proj.bias is not None:
                            nn.init.zeros_(self.other_features_proj.bias)
                    other_features = self.other_features_proj(other_features_raw)
            else:
                # Too small: pad with zeros
                padding = torch.zeros(
                    categorical_input.size(0), 
                    self.embed_dim - other_features_raw.size(1),
                    device=other_features_raw.device
                )
                other_features = torch.cat([other_features_raw, padding], dim=1)
        else:
            other_features = other_features_raw

        # Get item embeddings for sequence
        seq_embeds = self.embedding[1](seq_input)

        # Create mask for sequence
        mask = torch.arange(seq_input.size(1), device=seq_input.device).expand(
            seq_input.size(0), seq_input.size(1)
        ) < seq_lengths.unsqueeze(1)

        # Interest extraction
        hidden_states, final_state = self.interest_extractor(seq_embeds, seq_lengths)

        # Interest evolution
        evolved_state = self.interest_evolution(hidden_states, target_item_embed, seq_lengths)

        # Compute auxiliary loss (next item prediction)
        aux_loss = None
        if self.training:
            # Only compute auxiliary loss if sequences have at least 2 elements
            # (need at least 1 for prediction and 1 for target)
            # Check if we have valid sequences (length > 1) and if slicing won't result in empty tensors
            if hidden_states.size(1) > 1 and seq_embeds.size(1) > 1:
                # Predict next item embedding (exclude last timestep)
                pred_next_embed = self.aux_net(hidden_states[:, :-1, :])

                # Get actual next item embedding (exclude first timestep)
                next_item_embed = seq_embeds[:, 1:, :]

                # Ensure both have the same sequence length dimension
                if pred_next_embed.size(1) == next_item_embed.size(1) and pred_next_embed.size(1) > 0:
                    # Compute similarity
                    pred_norm = F.normalize(pred_next_embed, p=2, dim=2)
                    target_norm = F.normalize(next_item_embed, p=2, dim=2)

                    aux_similarity = torch.sum(pred_norm * target_norm, dim=2)

                    # Apply mask to exclude padding (mask for positions 1 onwards)
                    aux_mask = mask[:, 1:]
                    # Ensure mask matches the sequence length
                    if aux_mask.size(1) == aux_similarity.size(1):
                        aux_similarity = aux_similarity.masked_fill(~aux_mask, 0)

                        # Compute mean similarity only over valid positions
                        valid_positions = torch.sum(aux_mask.float())
                        if valid_positions > 0:
                            aux_loss = -torch.sum(aux_similarity) / valid_positions
                        else:
                            aux_loss = None
                    else:
                        # Mask size mismatch, skip auxiliary loss
                        aux_loss = None
                else:
                    # Shape mismatch or empty sequences, skip auxiliary loss
                    aux_loss = None
            else:
                # Sequences too short for auxiliary loss (need at least 2 timesteps)
                aux_loss = None

        # Concatenate evolved state, target item and other features
        # According to DIEN paper: final interest state + target item embedding + other features (user profile, context)
        # Use mean pooling for other features to get fixed size representation
        if other_features.size(1) > self.embed_dim:
            # Pool multiple features to embed_dim using mean
            num_features = other_features.size(1) // self.embed_dim
            remainder = other_features.size(1) % self.embed_dim
            if remainder == 0 and num_features > 0:
                # Can evenly divide, use mean pooling
                other_features_pooled = other_features.view(
                    categorical_input.size(0), num_features, self.embed_dim
                ).mean(dim=1)
            else:
                # Use linear projection if not evenly divisible
                if not hasattr(self, 'other_features_proj'):
                    self.other_features_proj = nn.Linear(
                        other_features.size(1), self.embed_dim
                    ).to(other_features.device)
                    nn.init.xavier_normal_(self.other_features_proj.weight)
                    if self.other_features_proj.bias is not None:
                        nn.init.zeros_(self.other_features_proj.bias)
                other_features_pooled = self.other_features_proj(other_features)
        elif other_features.size(1) < self.embed_dim:
            # Pad with zeros if too small
            padding = torch.zeros(
                categorical_input.size(0),
                self.embed_dim - other_features.size(1),
                device=other_features.device
            )
            other_features_pooled = torch.cat([other_features, padding], dim=1)
        else:
            other_features_pooled = other_features
        
        # Concatenate: evolved_state (gru_hidden_dim) + target_item_embed (embed_dim) + other_features (embed_dim)
        concat_features = torch.cat([evolved_state, target_item_embed, other_features_pooled], dim=1)

        # MLP layers
        for layer in self.mlp:
            concat_features = layer(concat_features)

        # Output layer
        output = self.sigmoid(self.output_layer(concat_features))

        # Always return tuple for consistency (training expects 2 values)
        if aux_loss is not None:
            return output, aux_loss
        else:
            # Return zero aux_loss if not computed (e.g., sequences too short)
            zero_aux_loss = torch.tensor(0.0, device=output.device, requires_grad=False)
            return output, zero_aux_loss


class DIEN_base(BaseRecommender):
    """Deep Interest Evolution Network (DIEN) base implementation.

    DIEN model captures user's temporal interest dynamics through GRU with attentional
    updates and auxiliary loss for better training.

    Architecture:
         ┌──────────────┐              ┌─────────────┐
    User │ Behavior Seq │──┬───────────│   Interest  │
    ────▶│ Embedding    │  │  ┌───────▶│  Evolution  │──┐
         └──────────────┘  │  │        └─────────────┘  │  ┌───────────┐     ┌─────────┐
                           │  │                         │  │ Prediction │    │         │
    Item ┌──────────────┐  │  │                         ├─▶│   Layer    │───▶│ Output  │
    ────▶│  Embedding   │──┘  │                         │  │  (MLP)     │    │         │
         └──────────────┘     │  ┌─────────────┐        │  └───────────┘     └─────────┘
                              │  │  Interest   │        │
                              └─▶│ Extraction  │────────┘
                                 │   (GRU)     │
                                 └─────────────┘

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, seed: int = 42):
        """
        Initialize DIEN model.

        Args:
            name: Model name.
            config: Configuration dictionary.
            seed: Random seed.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        super().__init__(name=name)

        # Set the random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Store configuration
        self.config = config or {}
        self.embed_dim = self.config.get("embed_dim", 64)
        self.mlp_dims = self.config.get("mlp_dims", [128, 64])
        self.attention_dims = self.config.get("attention_dims", [64])
        self.gru_hidden_dim = self.config.get("gru_hidden_dim", 64)
        self.aux_loss_weight = self.config.get("aux_loss_weight", 0.2)
        self.dropout = self.config.get("dropout", 0.2)
        self.batch_size = self.config.get("batch_size", 256)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.num_epochs = self.config.get("num_epochs", 10)
        self.weight_decay = self.config.get("weight_decay", 1e-6)
        self.max_seq_length = self.config.get("max_seq_length", 50)
        self.device = torch.device(self.config.get("device", "cpu"))
        self.seed = seed
        self.verbose = self.config.get("verbose", True)

        # Initialize logger
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(name)

        # Initialize maps and feature lists
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.feature_encoders = {}
        self.numerical_means = {}
        self.numerical_stds = {}
        self.field_dims = []

        # Initialize training variables
        self.user_sequences = {}
        self.loss_history = []

        # Initialize hooks for inspecting model internals
        self.hooks = {}

        # Set fitted flag
        self.is_fitted = False

    def fit(self, interactions: List[Tuple]) -> "DIEN_base":
        """
        Fit DIEN model to interactions.

        Args:
            interactions: List of (user, item, features, label) tuples.

        Returns:
            Fitted model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.logger.info("Fitting DIEN model...")

        # Extract user-item pairs and features
        self._extract_features(interactions)

        # Build model
        self._build_model()

        # Train model
        self._train(interactions)

        # Set fitted flag
        self.is_fitted = True

        self.logger.info("Model fitting completed")

        return self

    def _extract_features(self, interactions: List[Tuple]) -> None:
        """
        Extract features from interactions.

        Args:
            interactions: List of (user, item, features, label) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Extract users and items
        users = set()
        items = set()

        for user, item, _, _ in interactions:
            users.add(user)
            items.add(item)

        # Create user and item maps if not already created
        if not self.user_map:
            self.user_map = {
                user: i + 1 for i, user in enumerate(sorted(users))
            }  # Start from 1, 0 is padding
            self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        else:
            # Add new users
            for user in users:
                if user not in self.user_map:
                    self.user_map[user] = len(self.user_map) + 1  # +1 for padding
                    self.reverse_user_map[len(self.user_map)] = user

        if not self.item_map:
            self.item_map = {
                item: i + 1 for i, item in enumerate(sorted(items))
            }  # Start from 1, 0 is padding
            self.reverse_item_map = {i: item for item, i in self.item_map.items()}
        else:
            # Add new items
            for item in items:
                if item not in self.item_map:
                    self.item_map[item] = len(self.item_map) + 1  # +1 for padding
                    self.reverse_item_map[len(self.item_map)] = item

        # Extract features
        all_features = set()
        for _, _, features, _ in interactions:
            all_features.update(features.keys())

        # Categorize features
        for feature in all_features:
            if feature not in self.feature_names:
                self.feature_names.append(feature)

                # Check if feature is categorical or numerical
                is_numerical = True
                for _, _, features, _ in interactions:
                    if feature in features:
                        try:
                            float(features[feature])
                        except (ValueError, TypeError):
                            is_numerical = False
                            break

                if is_numerical:
                    self.numerical_features.append(feature)
                else:
                    self.categorical_features.append(feature)

        # Create feature encoders for categorical features
        for feature in self.categorical_features:
            if feature not in self.feature_encoders:
                self.feature_encoders[feature] = {}

            values = set()
            for _, _, features, _ in interactions:
                if feature in features:
                    values.add(str(features[feature]))

            for value in values:
                if value not in self.feature_encoders[feature]:
                    self.feature_encoders[feature][value] = (
                        len(self.feature_encoders[feature]) + 1
                    )  # +1 for unknown

        # Compute statistics for numerical features
        for feature in self.numerical_features:
            values = []
            for _, _, features, _ in interactions:
                if feature in features:
                    try:
                        values.append(float(features[feature]))
                    except (ValueError, TypeError):
                        pass

            if values:
                self.numerical_means[feature] = np.mean(values)
                self.numerical_stds[feature] = np.std(values) or 1.0  # Avoid division by zero

        # Create user sequences from interactions
        self._create_user_sequences(interactions)

    def _create_user_sequences(self, interactions: List[Tuple]) -> None:
        """
        Create user sequences for sequential modeling.

        Args:
            interactions: List of (user, item, features, label) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Sort interactions by user and timestamp (if available)
        sorted_interactions = sorted(
            interactions, key=lambda x: (x[0], x[2].get("timestamp", 0) if x[2] else 0)
        )

        # Create user sequences
        for user, item, _, _ in sorted_interactions:
            user_idx = self.user_map[user]
            item_idx = self.item_map[item]

            if user_idx not in self.user_sequences:
                self.user_sequences[user] = []

            self.user_sequences[user].append(item_idx)

        # Truncate sequences to max_seq_length
        for user in self.user_sequences:
            if len(self.user_sequences[user]) > self.max_seq_length:
                self.user_sequences[user] = self.user_sequences[user][-self.max_seq_length :]

    def _encode_features(self, features: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """
        Encode features for model input.

        Args:
            features: Features dict.

        Returns:
            Encoded features dict.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        encoded_features = {}

        # Encode categorical features
        for feature in self.categorical_features:
            if feature in features:
                value = str(features[feature])
                # Get feature encoding or unknown (0)
                encoded_features[feature] = self.feature_encoders.get(feature, {}).get(value, 0)
            else:
                # Use 0 for missing values
                encoded_features[feature] = 0

        # Encode numerical features
        for feature in self.numerical_features:
            if feature in features:
                # Normalize feature to [0, 1]
                try:
                    value = float(features[feature])
                    if feature in self.numerical_means and feature in self.numerical_stds:
                        value = (value - self.numerical_means[feature]) / self.numerical_stds[
                            feature
                        ]
                    encoded_features[feature] = value
                except (ValueError, TypeError):
                    # If not a numerical value, use 0 as default
                    encoded_features[feature] = 0.0
            else:
                # Use 0 for missing values
                encoded_features[feature] = 0.0

        return encoded_features

    def _build_model(self) -> None:
        """
        Build DIEN model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Calculate field dimensions
        self.field_dims = [len(self.user_map) + 1, len(self.item_map) + 1]  # +1 for padding

        # Add dimensions for each categorical feature
        for feature in self.categorical_features:
            self.field_dims.append(
                len(self.feature_encoders.get(feature, {})) + 1
            )  # +1 for unknown

        # Create model
        self.model = DIENModel(
            field_dims=self.field_dims,
            embed_dim=self.embed_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
            attention_dims=self.attention_dims,
            gru_hidden_dim=self.gru_hidden_dim,
            aux_loss_weight=self.aux_loss_weight,
        )

        # Move model to device
        self.model.to(self.device)

        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _train(self, interactions: List[Tuple], num_epochs: Optional[int] = None) -> None:
        """
        Train DIEN model on interactions.

        Args:
            interactions: List of (user, item, features, label) tuples.
            num_epochs: Number of epochs to train (defaults to self.num_epochs).

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if num_epochs is None:
            num_epochs = self.num_epochs

        # Prepare training data
        train_loader = self._prepare_data_loader(interactions)

        # Training loop
        self.model.train()
        epoch_iterator = range(num_epochs)
        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator, desc="Training")

        for epoch in epoch_iterator:
            total_loss = 0
            batch_count = 0

            for batch in train_loader:
                categorical_input, seq_input, seq_lengths, numerical_input, labels = batch

                # Forward pass
                self.optimizer.zero_grad()
                outputs, aux_loss = self.model(
                    categorical_input, seq_input, seq_lengths, numerical_input
                )

                # Compute loss
                bce_loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
                loss = bce_loss

                # Add auxiliary loss if available
                if aux_loss is not None:
                    loss += self.aux_loss_weight * aux_loss

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Update stats
                total_loss += loss.item()
                batch_count += 1

            # Log epoch loss
            avg_loss = total_loss / batch_count
            self.loss_history.append(avg_loss)

            if self.verbose:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def _prepare_data_loader(self, interactions: List[Tuple]) -> torch.utils.data.DataLoader:
        """
        Prepare data loader for training.

        Args:
            interactions: List of (user, item, features, label) tuples.

        Returns:
            PyTorch DataLoader.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Prepare data
        categorical_data = []
        seq_data = []
        seq_lengths = []
        numerical_data = []
        labels = []

        for user, item, features, label in interactions:
            # Get user and item indices
            user_idx = self.user_map[user]
            item_idx = self.item_map[item]

            # Encode features
            encoded_features = self._encode_features(features or {})

            # Prepare categorical input
            categorical_input = [user_idx, item_idx]
            for feature in self.categorical_features:
                categorical_input.append(encoded_features.get(feature, 0))

            # Prepare numerical input
            numerical_input = [
                encoded_features.get(feature, 0.0) for feature in self.numerical_features
            ]

            # Prepare sequence input (user's previous items)
            seq = self.user_sequences.get(user, [])[:-1]  # Exclude current item
            if len(seq) == 0:
                seq = [0]  # Use padding if no history

            seq_length = len(seq)

            # Pad sequence
            if seq_length > self.max_seq_length:
                seq = seq[-self.max_seq_length :]
                seq_length = self.max_seq_length
            elif seq_length < self.max_seq_length:
                seq = seq + [0] * (self.max_seq_length - seq_length)

            # Add to data lists
            categorical_data.append(categorical_input)
            seq_data.append(seq)
            seq_lengths.append(seq_length)
            numerical_data.append(numerical_input)
            labels.append(label)

        # Convert to tensors
        categorical_tensor = torch.tensor(categorical_data, dtype=torch.long, device=self.device)
        seq_tensor = torch.tensor(seq_data, dtype=torch.long, device=self.device)
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long, device=self.device)
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.float, device=self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(
            categorical_tensor, seq_tensor, seq_lengths_tensor, numerical_tensor, labels_tensor
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def predict(self, user: Any, item: Any, features: Dict[str, Any] = None) -> float:
        """
        Predict the probability of interaction between user and item.

        Args:
            user: User ID.
            item: Item ID.
            features: Features dict (optional).

        Returns:
            Probability of interaction.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user and item exist
        if user not in self.user_map:
            raise ValueError(f"User {user} not found in training data")

        if item not in self.item_map:
            raise ValueError(f"Item {item} not found in training data")

        # Set model to evaluation mode
        self.model.eval()

        # Get indices
        user_idx = self.user_map[user]
        item_idx = self.item_map[item]

        # Create feature tensor
        categorical_input = [user_idx, item_idx]
        encoded_features = self._encode_features(features or {})

        for feature in self.categorical_features:
            categorical_input.append(encoded_features.get(feature, 0))

        categorical_tensor = torch.tensor([categorical_input], dtype=torch.long, device=self.device)

        # Get numerical features
        numerical_input = [
            encoded_features.get(feature, 0.0) for feature in self.numerical_features
        ]
        numerical_tensor = torch.tensor([numerical_input], dtype=torch.float, device=self.device)

        # Get user sequence
        seq = self.user_sequences.get(user, [])

        # Remove target item from sequence if present
        if item_idx in seq:
            seq = [idx for idx in seq if idx != item_idx]

        seq_length = len(seq)

        # Pad sequence
        if seq_length > self.max_seq_length:
            seq = seq[-self.max_seq_length :]
            seq_length = self.max_seq_length
        elif seq_length < self.max_seq_length:
            seq = seq + [0] * (self.max_seq_length - seq_length)

        seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        seq_length_tensor = torch.tensor([seq_length], dtype=torch.long, device=self.device)

        # Forward pass
        with torch.no_grad():
            output, _ = self.model(categorical_tensor, seq_tensor, seq_length_tensor, numerical_tensor)
            prediction = output.item()

        return prediction

    def recommend(
        self, user: Any, top_n: int = 10, exclude_seen: bool = True, features: Dict[str, Any] = None
    ) -> List[Tuple[Any, float]]:
        """
        Recommend items for a user.

        Args:
            user: User ID.
            top_n: Number of recommendations to return.
            exclude_seen: Whether to exclude already seen items.
            features: Features dict (optional).

        Returns:
            List of (item, score) tuples.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        # Check if user exists
        if user not in self.user_map:
            raise ValueError(f"User {user} not found in training data")

        # Get user's seen items
        seen_items = set()
        if exclude_seen:
            for i, seq in enumerate(self.user_sequences.get(user, [])):
                if seq != 0:  # Skip padding
                    seen_items.add(self.reverse_item_map[seq])

        # Set model to evaluation mode
        self.model.eval()

        # Score all items
        scores = []
        for item, item_idx in self.item_map.items():
            # Skip if this item is already seen
            if exclude_seen and item in seen_items:
                continue

            # Get prediction
            score = self.predict(user, item, features)
            scores.append((item, score))

        # Sort by score and take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def register_hook(self, layer_name: str, hook_fn) -> torch.utils.hooks.RemovableHandle:
        """
        Register a hook for a layer.

        Args:
            layer_name: Layer name (e.g., 'model.embedding').
            hook_fn: Hook function.

        Returns:
            Hook handle.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        try:
            layer = self.model
            for name in layer_name.split("."):
                layer = getattr(layer, name)

            handle = layer.register_forward_hook(hook_fn)
            self.hooks[layer_name] = handle
            return handle
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Error registering hook for {layer_name}: {e}")
            raise ValueError(f"Invalid layer name: {layer_name}")

    def remove_hook(self, handle: torch.utils.hooks.RemovableHandle) -> None:
        """
        Remove a hook.

        Args:
            handle: Hook handle.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if handle is not None:
            handle.remove()
            for layer_name, h in list(self.hooks.items()):
                if h == handle:
                    del self.hooks[layer_name]

    def get_activation(self, layer_name: str) -> torch.Tensor:
        """
        Get activations for a layer.

        Args:
            layer_name: Layer name (e.g., 'model.embedding').

        Returns:
            Layer activations.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        activations = {}

        def hook_fn(module, input, output):
            activations[layer_name] = output

        # Register hook
        handle = self.register_hook(layer_name, hook_fn)

        try:
            # Create dummy input
            batch_size = 1
            categorical_input = torch.zeros(
                batch_size, len(self.field_dims), dtype=torch.long, device=self.device
            )
            seq_input = torch.zeros(
                batch_size, self.max_seq_length, dtype=torch.long, device=self.device
            )
            seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            numerical_input = torch.zeros(
                batch_size, len(self.numerical_features), dtype=torch.float, device=self.device
            )

            # Forward pass
            with torch.no_grad():
                self.model(categorical_input, seq_input, seq_lengths, numerical_input)
        finally:
            # Remove hook
            self.remove_hook(handle)

        return activations.get(layer_name, None)

    def get_user_embeddings(self) -> Dict[Any, np.ndarray]:
        """
        Get user embeddings from the model.
        
        Returns:
            Dictionary mapping user IDs to their embedding vectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        user_embeddings = {}
        with torch.no_grad():
            for user_id in self.user_map.keys():
                user_idx = self.user_map[user_id]
                # Get user embedding from the first embedding layer (user field)
                user_embed = self.model.embedding[0](torch.tensor([user_idx], device=self.device))
                # Convert to numpy via list to avoid NumPy 2.x compatibility issues
                user_emb = user_embed.cpu().detach().tolist()
                user_embeddings[user_id] = np.array(user_emb).flatten()
        
        return user_embeddings

    def get_item_embeddings(self) -> Dict[Any, np.ndarray]:
        """
        Get item embeddings from the model.
        
        Returns:
            Dictionary mapping item IDs to their embedding vectors.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        item_embeddings = {}
        with torch.no_grad():
            for item_id in self.item_map.keys():
                item_idx = self.item_map[item_id]
                # Get item embedding from the second embedding layer (item field)
                item_embed = self.model.embedding[1](torch.tensor([item_idx], device=self.device))
                # Convert to numpy via list to avoid NumPy 2.x compatibility issues
                item_emb = item_embed.cpu().detach().tolist()
                item_embeddings[item_id] = np.array(item_emb).flatten()
        
        return item_embeddings

    def save(self, filepath: str) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save the model to.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            self.logger.error("Model is not fitted yet. Call fit() first.")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Save model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "user_map": self.user_map,
                "item_map": self.item_map,
                "reverse_user_map": self.reverse_user_map,
                "reverse_item_map": self.reverse_item_map,
                "feature_names": self.feature_names,
                "categorical_features": self.categorical_features,
                "numerical_features": self.numerical_features,
                "feature_encoders": self.feature_encoders,
                "numerical_means": self.numerical_means,
                "numerical_stds": self.numerical_stds,
                "field_dims": self.field_dims,
                "loss_history": self.loss_history,
                "user_sequences": self.user_sequences,
            },
            filepath,
        )

        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "DIEN_base":
        """
        Load model from file.

        Args:
            filepath: Path to load the model from.
            device: Device to load the model on.

        Returns:
            Loaded model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location="cpu")

        # Create new instance with the saved config
        config = checkpoint["config"]
        instance = cls(name=config["name"], config=config, seed=config["seed"])

        # Load instance variables
        instance.user_map = checkpoint["user_map"]
        instance.item_map = checkpoint["item_map"]
        instance.reverse_user_map = checkpoint["reverse_user_map"]
        instance.reverse_item_map = checkpoint["reverse_item_map"]
        instance.feature_names = checkpoint["feature_names"]
        instance.categorical_features = checkpoint["categorical_features"]
        instance.numerical_features = checkpoint["numerical_features"]
        instance.feature_encoders = checkpoint["feature_encoders"]
        instance.numerical_means = checkpoint["numerical_means"]
        instance.numerical_stds = checkpoint["numerical_stds"]
        instance.field_dims = checkpoint["field_dims"]
        instance.loss_history = checkpoint["loss_history"]
        instance.user_sequences = checkpoint["user_sequences"]

        # Build model
        instance._build_model()

        # Load model state
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Set fitted flag
        instance.is_fitted = True

        instance.logger.info(f"Model loaded from {filepath}")

        return instance

    def set_device(self, device: str) -> None:
        """
        Set the device to run the model on.

        Args:
            device: Device to run the model on.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.device = torch.device(device)
        if hasattr(self, "model") and self.model is not None:
            self.model.to(self.device)

    def update_incremental(
        self, new_interactions: List[Tuple], new_users: List = None, new_items: List = None
    ) -> "DIEN_base":
        """
        Update the model with new data.

        Args:
            new_interactions: List of new interactions.
            new_users: List of new users.
            new_items: List of new items.

        Returns:
            Updated model.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        if not self.is_fitted:
            self.logger.warning("Model not fitted yet. Fitting with new data.")
            return self.fit(new_interactions)

        # Update user and item maps
        if new_users:
            for user_id in new_users:
                if user_id not in self.user_map:
                    self.user_map[user_id] = len(self.user_map) + 1  # +1 for padding
                    self.reverse_user_map[len(self.user_map)] = user_id

        if new_items:
            for item_id in new_items:
                if item_id not in self.item_map:
                    self.item_map[item_id] = len(self.item_map) + 1  # +1 for padding
                    self.reverse_item_map[len(self.item_map)] = item_id

        # Process new interactions
        if new_interactions:
            # Update user sequences
            self._extract_features(new_interactions)

            # Save old model state
            old_state_dict = None

            if hasattr(self, "model") and self.model is not None:
                old_state_dict = self.model.state_dict()

            # Build new model
            self._build_model()

            # Load old weights where possible
            if old_state_dict is not None:
                with torch.no_grad():
                    # Load params with matching shapes
                    for name, param in self.model.named_parameters():
                        if name in old_state_dict and param.shape == old_state_dict[name].shape:
                            param.copy_(old_state_dict[name])

            # Fine-tune with new data
            self._train(new_interactions, num_epochs=min(5, self.num_epochs))

        return self

    def forward(self, categorical_input, seq_input, seq_lengths, numerical_input):
        """Forward pass for the DIEN model.

        Args:
            categorical_input: Categorical input tensor
            seq_input: Sequence input tensor
            seq_lengths: Lengths of sequences
            numerical_input: Numerical input tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Outputs and auxiliary loss

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Get embeddings for categorical features and sequence
        categorical_emb = self.embedding(categorical_input)
        seq_emb = self.seq_embedding(seq_input)

        # Interest extraction using GRU
        # Make sure seq_lengths are valid (at least 1)
        seq_lengths = torch.clamp(seq_lengths, min=1)

        # Pack sequence for GRU
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            seq_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through GRU
        gru_output, _ = self.interest_extractor(packed_seq)
        unpacked_gru, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)

        # Ensure we have at least two sequence elements for auxiliary loss
        batch_size = seq_emb.size(0)
        seq_len = unpacked_gru.size(1)

        # Compute auxiliary loss (for training)
        if self.training:
            # If sequence length is less than 2, we need to handle this special case
            if seq_len < 2:
                # Create a dummy sequence with at least 2 elements
                dummy_seq = torch.zeros(batch_size, 2, self.gru_hidden_dim, device=seq_emb.device)
                if seq_len == 1:
                    dummy_seq[:, 0, :] = unpacked_gru[:, 0, :]
                unpacked_gru = dummy_seq
                seq_len = 2

            # Get hidden states for all but the last timestep
            h_states = unpacked_gru[:, :-1, :]

            # Get hidden states for all but the first timestep
            click_seq = unpacked_gru[:, 1:, :]

            # Normalize vectors for cosine similarity
            pred_norm = F.normalize(h_states, p=2, dim=2)
            target_norm = F.normalize(click_seq, p=2, dim=2)

            # Compute auxiliary loss
            aux_similarity = torch.sum(pred_norm * target_norm, dim=2)
            aux_loss = torch.mean(aux_similarity)
        else:
            # In evaluation mode, we don't need auxiliary loss
            aux_loss = torch.tensor(0.0, device=seq_emb.device)

        # Interest evolution using attention and GRU
        # Get the last hidden state from each sequence
        last_hidden = []
        for i, length in enumerate(seq_lengths):
            if length > 0:
                last_hidden.append(unpacked_gru[i, length - 1, :])
            else:
                # If sequence length is 0, use zeros
                last_hidden.append(torch.zeros(self.gru_hidden_dim, device=seq_emb.device))

        last_hidden = torch.stack(last_hidden)

        # Apply attention mechanism
        attention_output = self.attention_layer(last_hidden, unpacked_gru)

        # Concatenate all features
        concat_features = torch.cat(
            [categorical_emb.view(batch_size, -1), attention_output, numerical_input], dim=1
        )

        # Pass through MLP layers
        for layer in self.mlp_layers:
            concat_features = layer(concat_features)

        # Final prediction
        outputs = self.output_layer(concat_features)
        outputs = torch.sigmoid(outputs)

        if self.training:
            return outputs, aux_loss
        else:
            return outputs
