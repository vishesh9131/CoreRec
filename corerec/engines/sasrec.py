import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import pickle
import math
import json
import logging
from pathlib import Path

# Project imports (assumed present)
from corerec.api.base_recommender import BaseRecommender
from corerec.utils.validation import (
    validate_fit_inputs,
    validate_user_id,
    validate_top_k,
    validate_model_fitted,
)


class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward network for SASRec.
    """
    def __init__(self, hidden_units, dropout_rate, activation="gelu"):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units * 4, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation in ["swish", "silu"]:
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv2 = nn.Conv1d(hidden_units * 4, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, hidden_units]
        x = inputs.transpose(-1, -2)  # -> [batch_size, hidden_units, seq_len]
        x = self.conv1(x)
        x = self.activation(self.dropout1(x))
        x = self.conv2(x)
        x = self.dropout2(x)
        x = x.transpose(-1, -2)  # -> [batch_size, seq_len, hidden_units]
        x = x + inputs  # residual
        return x


class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommendation model (SASRec).
    """
    def __init__(
        self,
        n_items: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50,
        position_encoding: str = "learned",
        attention_type: str = "causal",
        activation: str = "gelu",
        item_embedding_init: Optional[np.ndarray] = None,
    ):
        super(SASRecModel, self).__init__()

        self.n_items = n_items
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        self.position_encoding = position_encoding
        self.attention_type = attention_type
        self.activation = activation

        # item embedding (+1 for padding index 0)
        self.item_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)

        if item_embedding_init is not None:
            assert item_embedding_init.shape == (n_items + 1, hidden_units), \
                f"Expected shape {(n_items + 1, hidden_units)}, got {item_embedding_init.shape}"
            self.item_emb.weight.data.copy_(torch.from_numpy(item_embedding_init))

        # position encoding
        if position_encoding == "learned":
            self.pos_emb = nn.Embedding(max_seq_length, hidden_units)
            self.register_buffer("pos_enc", torch.zeros(1))  # dummy to keep attribute
        elif position_encoding == "sinusoidal":
            pos_enc = self._get_sinusoidal_encoding(max_seq_length, hidden_units)
            self.register_buffer("pos_enc", pos_enc)  # [1, max_seq_length, hidden_units]
            self.pos_emb = None
        else:
            raise ValueError(f"Unsupported position encoding: {position_encoding}")

        self.dropout = nn.Dropout(p=dropout_rate)

        # transformer blocks: attention and feedforward
        self.attention_layers = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        self.attention_layer_norms = nn.ModuleList()
        self.feed_forward_layer_norms = nn.ModuleList()

        for _ in range(num_blocks):
            attn_layer = nn.MultiheadAttention(
                embed_dim=hidden_units, 
                num_heads=num_heads, 
                dropout=dropout_rate, 
                batch_first=False,
                bias=True
            )
            # Initialize attention weights more carefully to prevent NaN
            if hasattr(attn_layer, 'in_proj_weight') and attn_layer.in_proj_weight is not None:
                torch.nn.init.xavier_uniform_(attn_layer.in_proj_weight, gain=0.1)
            if hasattr(attn_layer, 'out_proj') and hasattr(attn_layer.out_proj, 'weight'):
                torch.nn.init.xavier_uniform_(attn_layer.out_proj.weight, gain=0.1)
            self.attention_layers.append(attn_layer)
            
            self.feed_forwards.append(
                PointWiseFeedForward(hidden_units, dropout_rate, activation)
            )
            self.attention_layer_norms.append(nn.LayerNorm(hidden_units))
            self.feed_forward_layer_norms.append(nn.LayerNorm(hidden_units))

        self.layer_norm_final = nn.LayerNorm(hidden_units)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Use smaller initialization to prevent numerical instability
            module.weight.data.normal_(0.0, 0.01)
            # Clamp to prevent extreme values
            module.weight.data.clamp_(-0.1, 0.1)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, nn.Conv1d):
            # Initialize Conv1d layers with smaller weights
            module.weight.data.normal_(0.0, 0.01)
            module.weight.data.clamp_(-0.1, 0.1)
            if module.bias is not None:
                module.bias.data.zero_()

    def _get_attention_mask(self, seq_len, device):
        if self.attention_type == "causal":
            # causal mask: upper triangular
            # Use float mask with large negative value instead of bool to prevent NaN
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * -1e9, diagonal=1)
            return mask
        return None

    def _get_sinusoidal_encoding(self, max_seq_len, hidden_units):
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_units, 2).float() * -(math.log(10000.0) / hidden_units))
        pe = torch.zeros(max_seq_len, hidden_units)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_seq_len, hidden_units]

    def forward(self, input_seqs: torch.LongTensor, padding_mask: Optional[torch.BoolTensor] = None, return_attention_weights: bool = False):
        """
        input_seqs: LongTensor of shape [batch_size, seq_len]
        padding_mask: BoolTensor of shape [batch_size, seq_len] where True indicates padding positions
        """
        batch_size, seq_length = input_seqs.size()

        # Validate input - ensure all indices are in valid range [0, n_items]
        if (input_seqs < 0).any() or (input_seqs > self.n_items).any():
            # Clamp invalid indices to valid range
            input_seqs = torch.clamp(input_seqs, 0, self.n_items)

        seq_emb = self.item_emb(input_seqs)  # [batch_size, seq_len, hidden_units]
        
        # Check for NaN in embeddings
        if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
            # Replace NaN/Inf with zeros
            seq_emb = torch.where(torch.isnan(seq_emb) | torch.isinf(seq_emb), torch.zeros_like(seq_emb), seq_emb)

        # add positional encoding
        if self.position_encoding == "learned":
            positions = torch.arange(seq_length, device=input_seqs.device)
            pos_emb = self.pos_emb(positions).unsqueeze(0)  # [1, seq_len, hidden_units]
            seq_emb = seq_emb + pos_emb
        else:
            seq_emb = seq_emb + self.pos_enc[:, :seq_length, :].to(seq_emb.device)

        seq_emb = self.dropout(seq_emb)

        # MultiheadAttention expects [seq_len, batch_size, hidden]
        x = seq_emb.transpose(0, 1)
        
        # Convert padding_mask format for MultiheadAttention
        # MultiheadAttention with batch_first=False expects key_padding_mask: [batch, seq_len] where True = ignore
        # But we need to transpose it since x is [seq_len, batch, hidden]
        if padding_mask is not None:
            padding_mask = padding_mask.bool()  # True for padding
            # Keep as [batch, seq_len] - MultiheadAttention will handle it correctly
            # Note: key_padding_mask is bool, attn_mask is float - this is correct for PyTorch

        attention_mask = self._get_attention_mask(seq_length, input_seqs.device)  # [seq_len, seq_len] or None

        all_attention_weights = [] if return_attention_weights else None

        for i in range(self.num_blocks):
            # pre-attention layer norm (pre-norm)
            residual = x
            x = self.attention_layer_norms[i](x.transpose(0, 1)).transpose(0, 1)  # normalize over hidden

            # Clamp to prevent extreme values
            x = torch.clamp(x, min=-10.0, max=10.0)

            # Clamp inputs before attention to prevent extreme values
            x_clamped = torch.clamp(x, min=-5.0, max=5.0)
            
            try:
                if return_attention_weights:
                    x_attn, attn_w = self.attention_layers[i](
                        query=x_clamped, key=x_clamped, value=x_clamped,
                        key_padding_mask=padding_mask,
                        attn_mask=attention_mask,
                        need_weights=True
                    )
                    all_attention_weights.append(attn_w)
                else:
                    x_attn, _ = self.attention_layers[i](
                        query=x_clamped, key=x_clamped, value=x_clamped,
                        key_padding_mask=padding_mask,
                        attn_mask=attention_mask,
                        need_weights=False
                    )
            except Exception as e:
                # If attention fails, use residual connection only
                x_attn = x_clamped
            
            # Check for NaN in attention output and replace with zeros
            if torch.isnan(x_attn).any() or torch.isinf(x_attn).any():
                # Replace NaN/Inf with zeros
                x_attn = torch.where(torch.isnan(x_attn) | torch.isinf(x_attn), torch.zeros_like(x_attn), x_attn)

            x = x_attn + residual
            # Clamp after residual
            x = torch.clamp(x, min=-10.0, max=10.0)

            # feed-forward block
            residual = x
            x = self.feed_forward_layer_norms[i](x.transpose(0, 1)).transpose(0, 1)
            x_ff = x.transpose(0, 1)  # -> [batch_size, seq_len, hidden]
            x_ff = self.feed_forwards[i](x_ff)
            
            # Check for NaN in feedforward output
            if torch.isnan(x_ff).any() or torch.isinf(x_ff).any():
                x_ff = torch.where(torch.isnan(x_ff) | torch.isinf(x_ff), torch.zeros_like(x_ff), x_ff)
            
            x = x_ff.transpose(0, 1)
            x = x + residual
            # Clamp after residual
            x = torch.clamp(x, min=-10.0, max=10.0)

        # final normalization
        x = self.layer_norm_final(x.transpose(0, 1)).transpose(0, 1)
        seq_emb = x.transpose(0, 1)  # [batch_size, seq_len, hidden_units]

        if return_attention_weights:
            return seq_emb, all_attention_weights
        return seq_emb


class SASRec(BaseRecommender):
    """
    SASRec wrapper that ties the SASRecModel to training, evaluation and I/O helpers.
    """
    def __init__(
        self,
        name: str = "SASRec",
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50,
        position_encoding: str = "learned",
        attention_type: str = "causal",
        activation: str = "gelu",
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        l2_reg: float = 1e-6,
        batch_size: int = 128,
        num_epochs: int = 10,
        neg_samples: int = 1,
        loss_type: str = "bce",  # 'bce' | 'bpr' | 'ce'
        early_stopping_patience: int = 3,
        save_checkpoints: bool = False,
        checkpoint_dir: str = "./checkpoints",
        export_embeddings: bool = False,
        item_popularity_bias: bool = False,
        user_cooling: bool = False,
        log_interval: int = 100,
        verbose: bool = True,
    ):
        super().__init__()
        self.name = name
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        self.position_encoding = position_encoding
        self.attention_type = attention_type
        self.activation = activation

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.neg_samples = neg_samples
        self.loss_type = loss_type.lower()
        self.early_stopping_patience = early_stopping_patience
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.export_embeddings = export_embeddings
        self.item_popularity_bias = item_popularity_bias
        self.user_cooling = user_cooling
        self.log_interval = log_interval
        self.verbose = verbose

        self.logger = self._create_logger()
        # mappings and state
        self.item_to_index: Dict[Any, int] = {}
        self.index_to_item: Dict[int, Any] = {}
        self.user_sequences: Dict[Any, List[int]] = {}
        self.item_popularity: Optional[np.ndarray] = None
        self.user_cooling_weights: Dict[Any, float] = {}
        self.best_model_state = None
        self.training_history: List[Dict[str, Any]] = []
        self.is_fitted = False

        # placeholder for model and optimizer
        self.model: Optional[SASRecModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def _create_logger(self):
        logger = logging.getLogger(f"{self.name}_{id(self)}")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[torch.device] = None):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # If loading an instance of SASRec, return it; otherwise raise
        if isinstance(obj, SASRec):
            if device is not None and hasattr(obj, 'model') and obj.model is not None:
                obj.device = device
                obj.model.to(device)
            return obj
        raise ValueError("Loaded object is not a SASRec instance")

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"{self.name} model saved to {path}")

    def get_item_embeddings(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.item_emb.weight.data.cpu().numpy()

    def export_item_embeddings(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = f"{self.name}_embeddings.pkl"
        embeddings = self.get_item_embeddings()
        export_data = {
            "embeddings": embeddings,
            "index_to_item": self.index_to_item,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        self.logger.info(f"Item embeddings exported to {filepath}")

    def predict(self, last_emb: torch.Tensor, item_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Compute scores for items given last_emb [batch_size, hidden].
        If item_indices is None -> return scores over all items [batch_size, n_items+1]
        Otherwise return scores for those indices [batch_size, len(item_indices)]
        """
        assert self.model is not None, "Model not initialized"
        item_weights = self.model.item_emb.weight  # [n_items+1, hidden]
        # scores = last_emb @ item_weights.T
        scores = torch.matmul(last_emb, item_weights.t())  # [batch, n_items+1]
        if item_indices is None:
            return scores
        else:
            return scores[:, item_indices]

    def fit(
        self,
        arg1: Union[List[Any], np.ndarray, Any],
        arg2: Optional[Union[List[Any], np.ndarray]] = None,
        arg3: Optional[Union[List[Any], np.ndarray]] = None,
        validation_data: Optional[Dict[Any, Tuple[List[int], List[int]]]] = None,
        item_embedding_init: Optional[np.ndarray] = None,
        user_item_timestamps: Optional[Dict[Any, List[Tuple[Any, Any]]]] = None,
        **kwargs
    ):
        """
        Fit SASRec model.

        Supports two calling conventions:
        1. fit(interaction_matrix, user_ids, item_ids, ...)  # Legacy
        2. fit(user_ids, item_ids, interaction_matrix, ...)  # Standard

        user_ids: list of user identifiers (len = n_users)
        item_ids: list of item identifiers (len = n_items)
        interaction_matrix: 2D binary or counts matrix shape [n_users, n_items]
        validation_data: optional dict {user_id: (input_seq, ground_truth_list)}
        """
        # Handle both calling conventions by checking first argument type
        import scipy.sparse as sp
        
        # Check if first arg is a matrix (numpy array or sparse matrix)
        is_matrix = (isinstance(arg1, (np.ndarray, sp.spmatrix)) or 
                    (hasattr(arg1, 'toarray') and hasattr(arg1, 'shape')))
        
        if is_matrix and arg2 is not None and isinstance(arg2, list) and arg3 is not None and isinstance(arg3, list):
            # Legacy: fit(interaction_matrix, user_ids, item_ids)
            interaction_matrix = arg1
            user_ids = arg2
            item_ids = arg3
        elif isinstance(arg1, list) and arg2 is not None and isinstance(arg2, list):
            # Standard: fit(user_ids, item_ids, interaction_matrix)
            user_ids = arg1
            item_ids = arg2
            interaction_matrix = arg3
            if interaction_matrix is None:
                raise ValueError("interaction_matrix must be provided as third argument")
        else:
            raise ValueError("Invalid arguments. Use either fit(interaction_matrix, user_ids, item_ids) or fit(user_ids, item_ids, interaction_matrix)")
        
        # Custom validation for matrix format
        if not isinstance(user_ids, list) or len(user_ids) == 0:
            raise ValueError("user_ids must be a non-empty list")
        if not isinstance(item_ids, list) or len(item_ids) == 0:
            raise ValueError("item_ids must be a non-empty list")
        
        # Handle sparse matrices (scipy.sparse)
        if hasattr(interaction_matrix, 'toarray'):
            interaction_matrix = interaction_matrix.toarray()
        elif not isinstance(interaction_matrix, np.ndarray):
            interaction_matrix = np.array(interaction_matrix)
        
        if interaction_matrix.ndim != 2:
            raise ValueError("interaction_matrix must be 2D")
        if interaction_matrix.shape[0] != len(user_ids):
            raise ValueError(f"interaction_matrix shape[0] ({interaction_matrix.shape[0]}) must match len(user_ids) ({len(user_ids)})")
        if interaction_matrix.shape[1] != len(item_ids):
            raise ValueError(f"interaction_matrix shape[1] ({interaction_matrix.shape[1]}) must match len(item_ids) ({len(item_ids)})")

        # build mappings
        self.item_to_index = {}
        self.index_to_item = {}
        for idx, item_id in enumerate(item_ids):
            self.item_to_index[item_id] = idx + 1  # reserve 0 for padding
            self.index_to_item[idx + 1] = item_id

        n_items = len(item_ids)
        # item popularity
        if self.item_popularity_bias:
            # sum across users
            self.item_popularity = np.asarray(interaction_matrix.sum(axis=0)).flatten() + 1.0
            self.item_popularity = np.log(self.item_popularity)

        # build user sequences
        self.user_sequences = {}
        user_interaction_counts = []
        
        # If timestamps are provided, use them to build ordered sequences
        if user_item_timestamps is not None:
            for user_id in user_ids:
                if user_id in user_item_timestamps:
                    # Get ordered items from timestamps
                    ordered_items = [item for item, _ in user_item_timestamps[user_id]]
                    # Map to internal indices
                    items = [self.item_to_index.get(item, 0) for item in ordered_items if item in self.item_to_index]
                    # Filter out padding (0)
                    items = [item for item in items if item > 0]
                    if len(items) > 0:
                        self.user_sequences[user_id] = items
                        user_interaction_counts.append(len(items))
        else:
            # Fallback: build sequences from interaction matrix (no temporal order)
            for u_idx, user_id in enumerate(user_ids):
                # For 1D array, nonzero() returns (indices,), so use [0]
                user_interactions = interaction_matrix[u_idx].nonzero()[0]
                if len(user_interactions) > 0:
                    items = [self.item_to_index[item_ids[i]] for i in user_interactions]
                    self.user_sequences[user_id] = items
                    user_interaction_counts.append(len(items))

        # user cooling
        if self.user_cooling:
            max_count = max(user_interaction_counts) if user_interaction_counts else 1
            for user_id, seq in self.user_sequences.items():
                self.user_cooling_weights[user_id] = 1.0 / math.sqrt(len(seq) / max_count) if len(seq) > 0 else 1.0

        # build model
        self.model = SASRecModel(
            n_items=n_items,
            hidden_units=self.hidden_units,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            max_seq_length=self.max_seq_length,
            position_encoding=self.position_encoding,
            attention_type=self.attention_type,
            activation=self.activation,
            item_embedding_init=item_embedding_init
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        
        # Verify model initialization - check for NaN in initial parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.logger.error(f"NaN/Inf detected in {name} after initialization. Reinitializing...")
                # Reinitialize this parameter
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param, gain=0.1)
                    else:
                        torch.nn.init.normal_(param, 0.0, 0.01)
                param.data.clamp_(-0.1, 0.1)
        
        # Test forward pass with dummy input to catch issues early
        try:
            dummy_seq = torch.zeros(1, self.max_seq_length, dtype=torch.long, device=self.device)
            dummy_mask = torch.ones(1, self.max_seq_length, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                test_output = self.model(dummy_seq, dummy_mask)
                if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    self.logger.error("Model produces NaN/Inf on dummy input. This indicates initialization issues.")
        except Exception as e:
            self.logger.warning(f"Model forward test failed: {e}. Continuing anyway.")
        
        # Verify model initialization - check for NaN in initial parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                self.logger.error(f"NaN/Inf detected in {name} after initialization. Reinitializing...")
                # Reinitialize this parameter
                if 'weight' in name:
                    if len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.normal_(param, 0.0, 0.01)
                param.data.clamp_(-0.1, 0.1)
        
        # Test forward pass with dummy input to catch issues early
        try:
            dummy_seq = torch.zeros(1, self.max_seq_length, dtype=torch.long, device=self.device)
            dummy_mask = torch.ones(1, self.max_seq_length, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                test_output = self.model(dummy_seq, dummy_mask)
                if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                    self.logger.error("Model produces NaN/Inf on dummy input. This indicates initialization issues.")
        except Exception as e:
            self.logger.warning(f"Model forward test failed: {e}. Continuing anyway.")

        # prepare training sequences
        train_sequences = []
        train_targets = []
        train_users = []

        for user_id, seq in self.user_sequences.items():
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                target = seq[i]
                if len(input_seq) > self.max_seq_length:
                    input_seq = input_seq[-self.max_seq_length:]
                else:
                    input_seq = [0] * (self.max_seq_length - len(input_seq)) + input_seq
                train_sequences.append(input_seq)
                train_targets.append(target)
                train_users.append(user_id)

        n_train = len(train_sequences)
        self.logger.info(f"Created {n_train} training instances")

        best_loss = float('inf')
        patience_counter = 0
        best_metric = 0.0 if validation_data else None
        self.training_history = []
        self.best_model_state = None

        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            if n_train == 0:
                self.logger.warning("No training instances. Skipping training.")
                break

            indices = np.arange(n_train)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            processed = 0

            for i in range(0, n_train, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n_train)]
                batch_sequences = [train_sequences[idx] for idx in batch_indices]
                batch_targets = [train_targets[idx] for idx in batch_indices]
                batch_users = [train_users[idx] for idx in batch_indices]
                batch_size = len(batch_indices)

                # negatives: sample neg_samples negatives per positive
                batch_negatives = []
                for _ in range(self.neg_samples):
                    negatives = []
                    for user_id, target in zip(batch_users, batch_targets):
                        user_seq = self.user_sequences.get(user_id, [])
                        # sample until not in seq
                        while True:
                            neg = np.random.randint(1, n_items + 1)
                            if neg not in user_seq:
                                break
                        negatives.append(neg)
                    batch_negatives.append(negatives)

                batch_sequences_t = torch.LongTensor(batch_sequences).to(self.device)
                batch_targets_t = torch.LongTensor(batch_targets).to(self.device)
                batch_negatives_t = [torch.LongTensor(negs).to(self.device) for negs in batch_negatives]

                # Validate input sequences - ensure all indices are valid (0 to n_items)
                if (batch_sequences_t < 0).any() or (batch_sequences_t > n_items).any():
                    self.logger.warning(f"Invalid sequence indices at epoch {epoch+1}, batch {i//self.batch_size}. Clamping values.")
                    batch_sequences_t = torch.clamp(batch_sequences_t, 0, n_items)
                
                if (batch_targets_t < 1).any() or (batch_targets_t > n_items).any():
                    self.logger.warning(f"Invalid target indices at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                    continue

                padding_mask = (batch_sequences_t == 0)  # [batch, seq_len]
                # MultiheadAttention expects key_padding_mask where True means ignore
                # Our padding_mask is already correct (True = padding = ignore)

                self.optimizer.zero_grad()
                
                # Check inputs before forward pass
                if torch.isnan(batch_sequences_t).any() or (batch_sequences_t < 0).any() or (batch_sequences_t > n_items).any():
                    self.logger.warning(f"Invalid input sequences at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                    continue
                
                try:
                    with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision to avoid NaN issues
                        seq_emb = self.model(batch_sequences_t, padding_mask)  # [batch, seq_len, hidden]
                except Exception as e:
                    self.logger.warning(f"Error in model forward pass at epoch {epoch+1}, batch {i//self.batch_size}: {e}. Skipping batch.")
                    continue
                
                # Check for NaN in model output
                if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
                    # Log which part of the model produced NaN
                    self.logger.warning(f"NaN/Inf in model output at epoch {epoch+1}, batch {i//self.batch_size}. "
                                      f"Input range: [{batch_sequences_t.min().item()}, {batch_sequences_t.max().item()}]. "
                                      f"Output stats: min={seq_emb.min().item():.4f}, max={seq_emb.max().item():.4f}, "
                                      f"mean={seq_emb.mean().item():.4f}, std={seq_emb.std().item():.4f}. Skipping batch.")
                    continue
                
                last_emb = seq_emb[:, -1, :]  # [batch, hidden]
                
                # Check for NaN in last embedding
                if torch.isnan(last_emb).any() or torch.isinf(last_emb).any():
                    self.logger.warning(f"NaN/Inf in last embedding at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                    continue

                # Check model parameters for NaN before computing loss
                has_nan_params = False
                for param in self.model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan_params = True
                        break
                
                if has_nan_params:
                    self.logger.warning(f"NaN/Inf in model parameters at epoch {epoch+1}, batch {i//self.batch_size}. Reinitializing model.")
                    # Reinitialize model
                    self.model = SASRecModel(
                        n_items=n_items,
                        hidden_units=self.hidden_units,
                        num_blocks=self.num_blocks,
                        num_heads=self.num_heads,
                        dropout_rate=self.dropout_rate,
                        max_seq_length=self.max_seq_length,
                        position_encoding=self.position_encoding,
                        attention_type=self.attention_type,
                        activation=self.activation,
                        item_embedding_init=None
                    ).to(self.device)
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.l2_reg
                    )
                    continue

                if self.loss_type == 'bpr':
                    # Get full scores for all items
                    full_scores = self.predict(last_emb, item_indices=None)  # [batch, n_items+1]
                    
                    # Check for NaN in scores
                    if torch.isnan(full_scores).any() or torch.isinf(full_scores).any():
                        self.logger.warning(f"NaN/Inf in prediction scores at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                        continue
                    
                    # Extract positive scores
                    pos_idx = batch_targets_t.unsqueeze(1)  # [batch,1]
                    pos_scores = full_scores.gather(1, pos_idx).squeeze(1)  # [batch]
                    loss = 0.0
                    for negs in batch_negatives_t:
                        # negs is [batch] - one negative per example
                        # Use gather to extract scores for each negative item
                        neg_idx = negs.unsqueeze(1)  # [batch, 1]
                        neg_scores = full_scores.gather(1, neg_idx).squeeze(1)  # [batch]
                        # Clamp difference to prevent numerical issues
                        diff = pos_scores - neg_scores
                        diff = torch.clamp(diff, min=-50, max=50)  # Prevent extreme values
                        loss += -torch.log(torch.sigmoid(diff) + 1e-8).mean()
                    loss = loss / self.neg_samples

                elif self.loss_type == 'ce':
                    logits = self.predict(last_emb)  # [batch, n_items+1]
                    
                    # Check for NaN in logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        self.logger.warning(f"NaN/Inf in logits at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                        continue
                    
                    # cross entropy expects [batch, C] and targets in 0..C-1
                    # our padding 0 is present; target is in 1..n_items
                    loss = F.cross_entropy(logits, batch_targets_t)

                else:  # 'bce' default
                    # Get full scores for all items
                    full_scores = self.predict(last_emb, item_indices=None)  # [batch, n_items+1]
                    
                    # Check for NaN in scores
                    if torch.isnan(full_scores).any() or torch.isinf(full_scores).any():
                        self.logger.warning(f"NaN/Inf in prediction scores at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                        continue
                    
                    # Extract positive scores
                    pos_idx = batch_targets_t.unsqueeze(1)  # [batch,1]
                    pos_scores = full_scores.gather(1, pos_idx).squeeze(1)  # [batch]
                    
                    # Clamp scores to prevent extreme values
                    pos_scores = torch.clamp(pos_scores, min=-50, max=50)
                    
                    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                    
                    neg_loss = 0.0
                    for negs in batch_negatives_t:
                        # negs is [batch] - one negative per example
                        neg_idx = negs.unsqueeze(1)  # [batch, 1]
                        neg_scores = full_scores.gather(1, neg_idx).squeeze(1)  # [batch]
                        
                        # Clamp scores to prevent extreme values
                        neg_scores = torch.clamp(neg_scores, min=-50, max=50)
                        
                        neg_loss += F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
                    loss = (pos_loss + neg_loss) / (1 + self.neg_samples)

                # user cooling weights
                if self.user_cooling:
                    cooling_weights = torch.tensor(
                        [self.user_cooling_weights.get(u, 1.0) for u in batch_users],
                        dtype=loss.dtype,
                        device=self.device
                    )
                    loss = (loss * cooling_weights).mean()

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN/Inf loss detected at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                    self.optimizer.zero_grad()  # Clear gradients
                    continue

                loss.backward()
                # Clip gradients to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad or torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"NaN/Inf gradients detected at epoch {epoch+1}, batch {i//self.batch_size}. Skipping batch.")
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                # Check for NaN parameters after update
                has_nan_param = False
                for param in self.model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan_param = True
                        break
                
                if has_nan_param:
                    self.logger.error(f"NaN/Inf parameters detected after update at epoch {epoch+1}, batch {i//self.batch_size}. Training may be unstable.")
                    # Try to recover by reloading best model state if available
                    if self.best_model_state is not None:
                        self.logger.info("Attempting to recover by reloading best model state.")
                        self.model.load_state_dict(self.best_model_state)
                        self.optimizer.zero_grad()
                    continue

                epoch_loss += loss.item() * batch_size
                processed += batch_size

                if (i // self.batch_size) % self.log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} Batch {i//self.batch_size} Loss: {loss.item():.4f}")

            # Check if we processed any batches
            if processed == 0:
                self.logger.error(f"No valid batches processed in epoch {epoch+1}. All batches had NaN/Inf. Stopping training.")
                if self.best_model_state is not None:
                    self.logger.info("Reloading best model state.")
                    self.model.load_state_dict(self.best_model_state)
                break
            
            avg_loss = epoch_loss / processed
            
            # Save best model state if loss improved and is valid
            if not (np.isnan(avg_loss) or np.isinf(avg_loss)):
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                self.logger.warning(f"Epoch {epoch+1} ended with invalid loss: {avg_loss}. Not updating best model.")
                patience_counter += 1

            metrics = {}
            if validation_data:
                metrics = self.evaluate(validation_data)
                current_metric = np.mean([metrics[m] for m in metrics]) if len(metrics) > 0 else 0.0
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Validation metric(avg): {current_metric:.4f}")

                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping after epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping after epoch {epoch+1}")
                        break

            history_entry = {"epoch": epoch + 1, "loss": avg_loss}
            history_entry.update(metrics)
            self.training_history.append(history_entry)

            if self.save_checkpoints:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.name}_epoch_{epoch+1}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)

        # restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        if self.export_embeddings:
            self.export_item_embeddings()

        self.is_fitted = True
        return self

    def recommend(self, user_id: Any, top_n: int = 10, exclude_seen: bool = True) -> List[Any]:
        """
        Recommend top-n items for a user.
        """
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_sequences if hasattr(self, 'user_sequences') else {})

        if user_id not in self.user_sequences:
            self.logger.warning(f"Unknown user: {user_id}")
            return []

        seq = list(self.user_sequences[user_id])  # original sequence of internal indices
        if len(seq) > self.max_seq_length:
            seq_in = seq[-self.max_seq_length:]
        else:
            seq_in = [0] * (self.max_seq_length - len(seq)) + seq

        input_seq = torch.LongTensor([seq_in]).to(self.device)
        padding_mask = (input_seq == 0)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_seq, padding_mask)  # [1, seq_len, hidden]
            last_idx = torch.sum(input_seq > 0, dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
            last_emb = logits[0, last_idx[0], :].unsqueeze(0)  # [1, hidden]
            scores = self.predict(last_emb)  # [1, n_items+1]
            scores = scores.squeeze(0).cpu().numpy()

        scores[0] = -np.inf  # disallow padding
        if exclude_seen:
            for item_idx in seq:
                if 0 <= item_idx < len(scores):
                    scores[item_idx] = -np.inf
        if self.item_popularity_bias and self.item_popularity is not None:
            scores[1:] = scores[1:] - self.item_popularity

        top_indices = np.argsort(scores)[::-1][:top_n]
        recommendations = [self.index_to_item.get(int(idx), None) for idx in top_indices]
        return recommendations

    def evaluate(self, eval_data: Dict[Any, Tuple[List[int], List[int]]], metrics: Optional[List[str]] = None, cutoffs: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate model on eval_data: dict {user_id: (input_seq, ground_truth_list)}
        metrics: list e.g. ["hit", "ndcg", "precision", "recall"]
        cutoffs: e.g. [5, 10]
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        if metrics is None:
            metrics = ["hit", "ndcg", "precision", "recall"]
        if cutoffs is None:
            cutoffs = [5, 10]

        metric_values = {f"{m}@{k}": 0.0 for m in metrics for k in cutoffs}
        n_users = 0

        for user_id, (input_seq, ground_truth) in eval_data.items():
            if user_id not in self.user_sequences:
                continue
            recs = self.recommend(user_id, top_n=max(cutoffs), exclude_seen=True)
            for k in cutoffs:
                recs_at_k = recs[:k]

                if "hit" in metrics:
                    hit = int(any(item in ground_truth for item in recs_at_k))
                    metric_values[f"hit@{k}"] += hit

                if "ndcg" in metrics:
                    ndcg = 0.0
                    for i, item in enumerate(recs_at_k):
                        if item in ground_truth:
                            ndcg += 1.0 / math.log2(i + 2)
                    if len(ground_truth) > 0:
                        idcg_k = min(len(ground_truth), k)
                        idcg = sum(1.0 / math.log2(i + 2) for i in range(idcg_k))
                        ndcg = ndcg / idcg if idcg > 0 else 0.0
                    metric_values[f"ndcg@{k}"] += ndcg

                if "precision" in metrics:
                    n_rel = sum(1 for item in recs_at_k if item in ground_truth)
                    precision = n_rel / k
                    metric_values[f"precision@{k}"] += precision

                if "recall" in metrics:
                    n_rel = sum(1 for item in recs_at_k if item in ground_truth)
                    recall = n_rel / len(ground_truth) if len(ground_truth) > 0 else 0.0
                    metric_values[f"recall@{k}"] += recall

            n_users += 1

        if n_users > 0:
            for key in metric_values:
                metric_values[key] /= n_users

        return metric_values