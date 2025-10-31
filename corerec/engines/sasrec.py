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
from corerec.base_recommender import BaseCorerec

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
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.conv2 = nn.Conv1d(hidden_units * 4, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.activation(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        )))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, L)
        outputs += inputs  # residual connection
        return outputs

class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.
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
        item_embedding_init: Optional[np.ndarray] = None
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
        
        # Item embedding
        self.item_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)  # +1 for padding
        
        # Initialize with pre-trained embeddings if provided
        if item_embedding_init is not None:
            assert item_embedding_init.shape == (n_items + 1, hidden_units), \
                f"Expected shape ({n_items + 1}, {hidden_units}), got {item_embedding_init.shape}"
            self.item_emb.weight.data.copy_(torch.from_numpy(item_embedding_init))
        
        # Position encoding
        if position_encoding == "learned":
            self.pos_emb = nn.Embedding(max_seq_length, hidden_units)
        elif position_encoding == "sinusoidal":
            pos_enc = self._get_sinusoidal_encoding(max_seq_length, hidden_units)
            self.register_buffer('pos_enc', pos_enc)
            self.pos_emb = None
        else:
            raise ValueError(f"Unsupported position encoding: {position_encoding}")
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_units)
        self.layer_norm2 = nn.LayerNorm(hidden_units)
        
        # Multi-head attention blocks
        self.attention_layers = nn.ModuleList([])
        self.feed_forwards = nn.ModuleList([])
        self.attention_layer_norms = nn.ModuleList([])
        self.feed_forward_layer_norms = nn.ModuleList([])
        
        for _ in range(num_blocks):
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            )
            self.feed_forwards.append(
                PointWiseFeedForward(hidden_units, dropout_rate, activation)
            )
            self.attention_layer_norms.append(nn.LayerNorm(hidden_units))
            self.feed_forward_layer_norms.append(nn.LayerNorm(hidden_units))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _get_sinusoidal_encoding(self, max_seq_len, hidden_units):
        """Generate sinusoidal position encoding table"""
        def get_position_angle(pos, i, d_model):
            return pos / np.power(10000, 2 * (i // 2) / d_model)
            
        def get_posi_angle_vec(position):
            return [get_position_angle(position, hid_j, hidden_units) for hid_j in range(hidden_units)]
            
        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(max_seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def _get_attention_mask(self, seq_len, device):
        """Generate an attention mask based on the attention type"""
        if self.attention_type == "causal":
            # Create a causal (triangular) mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            return mask
        elif self.attention_type == "bidirectional":
            # No directional restriction
            return None
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
    
    def forward(self, input_seqs, padding_mask=None, return_attention_weights=False):
        # Get sequence length
        batch_size, seq_length = input_seqs.size()
        
        # Get item embeddings
        seq_emb = self.item_emb(input_seqs)  # [batch_size, seq_len, hidden_units]
        
        # Add position embeddings
        if self.position_encoding == "learned":
            positions = torch.arange(seq_length, dtype=torch.long, device=input_seqs.device)
            positions = positions.unsqueeze(0).expand_as(input_seqs)
            pos_emb = self.pos_emb(positions)
            seq_emb = seq_emb + pos_emb
        else:  # sinusoidal
            seq_emb = seq_emb + self.pos_enc[:, :seq_length, :]
        
        # Apply dropout
        seq_emb = self.dropout(seq_emb)
        
        # Prepare attention mask
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
        
        # Get attention mask based on attention type
        attention_mask = self._get_attention_mask(seq_length, input_seqs.device)
        
        # Transpose for attention operation: [batch_size, seq_len, hidden] -> [seq_len, batch_size, hidden]
        x = seq_emb.transpose(0, 1)
        
        # Store attention weights if needed
        all_attention_weights = [] if return_attention_weights else None
        
        # Apply transformer blocks
        for i in range(self.num_blocks):
            # Layer normalization before attention
            residual = x
            x = self.attention_layer_norms[i](x)
            
            # Self-attention
            if return_attention_weights:
                x, attn_weights = self.attention_layers[i](
                    query=x, key=x, value=x,
                    key_padding_mask=padding_mask,
                    attn_mask=attention_mask,
                    need_weights=True
                )
                all_attention_weights.append(attn_weights)
            else:
                x, _ = self.attention_layers[i](
                    query=x, key=x, value=x,
                    key_padding_mask=padding_mask,
                    attn_mask=attention_mask,
                    need_weights=False
                )
            
            # Residual connection
            x = x + residual
            
            # Layer normalization before feed-forward
            residual = x
            x = self.feed_forward_layer_norms[i](x)
            
            # Position-wise feed-forward
            # Convert back to [batch_size, seq_len, hidden]
            x_for_ff = x.transpose(0, 1)
            x_for_ff = self.feed_forwards[i](x_for_ff)
            x = x_for_ff.transpose(0, 1)
            
            # Residual connection
            x = x + residual
        
        # Final layer normalization
        x = self.attention_layer_norms[-1](x)
        
        # Convert back: [seq_len, batch_size, hidden] -> [batch_size, seq_len, hidden]
        seq_emb = x.transpose(0, 1)
        
        if return_attention_weights:
            return seq_emb, all_attention_weights
        else:
            return seq_emb
    
    def predict(self, seq_emb, item_indices=None):
        """Generate predictions by comparing with item embeddings"""
        if item_indices is None:
            # Compare with all items
            all_items = self.item_emb.weight
            return torch.matmul(seq_emb, all_items.transpose(0, 1))
        else:
            # Compare with specific items
            item_emb = self.item_emb(item_indices)
            return torch.sum(seq_emb * item_emb, dim=-1)
    
    def get_all_embeddings(self):
        """Return all item embeddings for external use"""
        return self.item_emb.weight.data.cpu().numpy()

class SASRec(BaseCorerec):
    """
    Self-Attentive Sequential Recommendation (SASRec)
    
    A sequential recommendation model that uses self-attention mechanism
    to capture long-range dependencies in user behavior sequences.
    
    Features:
    - Transformer architecture for sequence modeling
    - Flexible attention mechanisms (causal or bidirectional)
    - Position encodings (learned or sinusoidal)
    - Multi-head attention support
    - Customizable architecture (layers, heads, dimensions)
    - Various training objectives (BPR, CE, BCE)
    - Support for pre-trained item embeddings
    - Attention weights visualization
    - Model export and import
    
    Reference:
    Wang-Cheng Kang, Julian McAuley. "Self-Attentive Sequential Recommendation." (ICDM 2018)
    """
    
    def __init__(
        self,
        name: str = "SASRec",
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50,
        attention_type: str = "causal",  # Options: causal, bidirectional
        position_encoding: str = "learned",  # Options: learned, sinusoidal
        activation: str = "gelu",  # Options: relu, gelu, swish
        loss_type: str = "bpr",  # Options: bpr, ce, bce
        learning_rate: float = 0.001,
        batch_size: int = 128,
        num_epochs: int = 20,
        l2_reg: float = 0.00001,
        early_stopping_patience: int = 5,
        item_embedding_init: Optional[np.ndarray] = None,  # Pre-trained embeddings
        eval_metrics: List[str] = ["ndcg@10", "hit@10"],
        user_cooling: bool = False,  # Reduce weight for frequent users
        item_popularity_bias: bool = False,  # Correct for popularity bias
        neg_samples: int = 1,  # Number of negative samples per positive
        save_checkpoints: bool = False,
        checkpoint_dir: str = "./checkpoints",
        export_embeddings: bool = False,
        trainable: bool = True,
        verbose: bool = False,
        log_interval: int = 100,
        seed: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        
        # Model hyperparameters
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        self.attention_type = attention_type
        self.position_encoding = position_encoding
        self.activation = activation
        self.loss_type = loss_type
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience
        self.item_embedding_init = item_embedding_init
        self.eval_metrics = eval_metrics
        self.user_cooling = user_cooling
        self.item_popularity_bias = item_popularity_bias
        self.neg_samples = neg_samples
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.export_embeddings = export_embeddings
        self.log_interval = log_interval
        self.seed = seed
        self.device = device
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Will be set during fit
        self.item_to_index = {}
        self.index_to_item = {}
        self.user_sequences = {}
        self.model = None
        self.optimizer = None
        self.item_popularity = None
        self.is_fitted = False
        self.best_model_state = None
        self.training_history = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Set up a logger for the model"""
        logger = logging.getLogger(f"SASRec_{id(self)}")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def fit(self, interaction_matrix, user_ids: List[int], item_ids: List[int], validation_data=None):
        """
        Train the recommender using user-item interactions.
        
        Parameters:
        - interaction_matrix: User-item interaction matrix (scipy sparse matrix)
        - user_ids: List of user IDs
        - item_ids: List of item IDs
        - validation_data: Optional validation data for early stopping
        """
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        self.logger.info(f"Training {self.name} model with {len(user_ids)} users and {len(item_ids)} items")
        
        # Create item mapping
        for idx, item_id in enumerate(item_ids):
            self.item_to_index[item_id] = idx + 1  # Reserve 0 for padding
            self.index_to_item[idx + 1] = item_id
        
        # Calculate item popularity if needed
        if self.item_popularity_bias:
            self.item_popularity = np.asarray(interaction_matrix.sum(axis=0)).flatten() + 1  # +1 smoothing
            self.item_popularity = np.log(self.item_popularity)
        
        # Process interaction matrix to create user sequences
        user_interaction_counts = []
        for u_idx, user_id in enumerate(user_ids):
            user_interactions = interaction_matrix[u_idx].nonzero()[1]
            if len(user_interactions) > 0:
                # Map item indices to our internal indices
                items = [self.item_to_index[item_ids[i]] for i in user_interactions]
                self.user_sequences[user_id] = items
                user_interaction_counts.append(len(items))
        
        # Calculate user cooling weights if needed
        if self.user_cooling:
            self.user_cooling_weights = {}
            max_count = max(user_interaction_counts)
            for user_id, seq in self.user_sequences.items():
                # Inverse square root cooling
                self.user_cooling_weights[user_id] = 1.0 / np.sqrt(len(seq) / max_count)
        
        # Build the model
        n_items = len(item_ids)
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
            item_embedding_init=self.item_embedding_init
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        
        # Prepare training data
        train_sequences = []
        train_targets = []
        train_users = []  # Store user IDs for cooling
        
        for user_id, seq in self.user_sequences.items():
            if len(seq) < 2:  # Skip users with too few interactions
                continue
                
            # Create sequences for training
            for i in range(1, len(seq)):
                # Use all previous items as input sequence
                input_seq = seq[:i]
                target = seq[i]
                
                # If sequence is too long, truncate it
                if len(input_seq) > self.max_seq_length:
                    input_seq = input_seq[-self.max_seq_length:]
                else:
                    # Pad sequence
                    input_seq = [0] * (self.max_seq_length - len(input_seq)) + input_seq
                
                train_sequences.append(input_seq)
                train_targets.append(target)
                train_users.append(user_id)
        
        n_train = len(train_sequences)
        self.logger.info(f"Created {n_train} training instances")
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        best_metric = 0.0 if validation_data else None
        self.training_history = []
        
        # Create checkpoint directory if needed
        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.num_epochs):
            # Set model to training mode
            self.model.train()
            
            # Shuffle training data
            indices = np.arange(n_train)
            np.random.shuffle(indices)
            
            # Mini-batch training
            epoch_loss = 0
            processed = 0
            
            for i in range(0, n_train, self.batch_size):
                # Get batch indices
                batch_indices = indices[i:min(i + self.batch_size, n_train)]
                batch_size = len(batch_indices)
                
                # Get batch data
                batch_sequences = [train_sequences[idx] for idx in batch_indices]
                batch_targets = [train_targets[idx] for idx in batch_indices]
                batch_users = [train_users[idx] for idx in batch_indices]
                
                # Create negative samples
                batch_negatives = []
                for _ in range(self.neg_samples):
                    negatives = []
                    for user_id, target in zip(batch_users, batch_targets):
                        user_seq = self.user_sequences.get(user_id, [])
                        while True:
                            neg = np.random.randint(1, n_items + 1)
                            if neg not in user_seq:
                                break
                        negatives.append(neg)
                    batch_negatives.append(negatives)
                
                # Convert to tensors
                batch_sequences = torch.LongTensor(batch_sequences).to(self.device)
                batch_targets = torch.LongTensor(batch_targets).to(self.device)
                batch_negatives = [torch.LongTensor(negs).to(self.device) for negs in batch_negatives]
                
                # Create padding mask
                padding_mask = (batch_sequences == 0)
                
                # Forward pass
                self.optimizer.zero_grad()
                seq_emb = self.model(batch_sequences, padding_mask)
                
                # Get last position predictions (for next item)
                last_emb = seq_emb[:, -1, :]
                
                # Calculate loss based on loss type
                if self.loss_type == 'bpr':
                    # BPR loss
                    pos_scores = self.model.predict(last_emb, batch_targets)
                    neg_scores_list = [self.model.predict(last_emb, negs) for negs in batch_negatives]
                    
                    loss = 0
                    for neg_scores in neg_scores_list:
                        # Apply popularity correction if enabled
                        if self.item_popularity_bias:
                            pos_correction = torch.from_numpy(
                                self.item_popularity[batch_targets.cpu().numpy() - 1]
                            ).float().to(self.device)
                            neg_correction = torch.from_numpy(
                                self.item_popularity[negs.cpu().numpy() - 1]
                            ).float().to(self.device)
                            
                            pos_scores = pos_scores - pos_correction
                            neg_scores = neg_scores - neg_correction
                        
                        loss += -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
                    
                    loss /= self.neg_samples
                
                elif self.loss_type == 'ce':
                    # Cross entropy loss
                    logits = self.model.predict(last_emb)  # [batch_size, n_items+1]
                    loss = F.cross_entropy(logits, batch_targets)
                
                else:  # 'bce'
                    # Binary cross entropy loss
                    pos_scores = self.model.predict(last_emb, batch_targets)
                    neg_scores_list = [self.model.predict(last_emb, negs) for negs in batch_negatives]
                    
                    loss = 0
                    loss += F.binary_cross_entropy_with_logits(
                        pos_scores, torch.ones_like(pos_scores)
                    )
                    
                    for neg_scores in neg_scores_list:
                        loss += F.binary_cross_entropy_with_logits(
                            neg_scores, torch.zeros_like(neg_scores)
                        )
                    
                    loss /= (1 + self.neg_samples)
                
                # Apply user cooling if enabled
                if self.user_cooling:
                    cooling_weights = torch.tensor(
                        [self.user_cooling_weights.get(user_id, 1.0) for user_id in batch_users],
                        device=self.device
                    )
                    loss = (loss * cooling_weights).mean()
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                epoch_loss += loss.item() * batch_size
                processed += batch_size
                
                if i % self.log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, " + 
                                     f"Batch {i//self.batch_size}/{n_train//self.batch_size}, " + 
                                     f"Loss: {loss.item():.4f}")
            
            # Calculate average loss
            avg_loss = epoch_loss / processed if processed > 0 else float('inf')
            
            # Validation if provided
            metrics = {}
            if validation_data:
                metrics = self.evaluate(validation_data)
                current_metric = np.mean([metrics[m] for m in self.eval_metrics])
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, " +
                               f"Validation: {current_metric:.4f}")
                
                # Early stopping check
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after epoch {epoch+1}")
                        break
            else:
                self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
                # Early stopping based on training loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after epoch {epoch+1}")
                        break
            
            # Save history
            history_entry = {"epoch": epoch+1, "loss": avg_loss}
            history_entry.update(metrics)
            self.training_history.append(history_entry)
            
            # Save checkpoint if enabled
            if self.save_checkpoints:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.name}_epoch_{epoch+1}.pt")
                self.save_model(checkpoint_path)
        
        # Restore best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        
        # Export embeddings if requested
        if self.export_embeddings:
            self.export_item_embeddings()
        
        self.is_fitted = True
        return self
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Parameters:
        - user_id: User ID
        - top_n: Number of recommendations to generate
        - exclude_seen: Whether to exclude already seen items
        
        Returns:
        - List of recommended item IDs
        """
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_k if 'top_k' in locals() else 10)
        
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if user_id not in self.user_sequences:
            self.logger.warning(f"Unknown user: {user_id}")
            return []  # Return empty list for unknown users
        
        # Get user's sequence
        seq = self.user_sequences[user_id]
        
        # Create input sequence
        if len(seq) > self.max_seq_length:
            seq = seq[-self.max_seq_length:]
        else:
            seq = [0] * (self.max_seq_length - len(seq)) + seq
        
        # Create tensor
        input_seq = torch.LongTensor([seq]).to(self.device)
        padding_mask = (input_seq == 0)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate predictions
        with torch.no_grad():
            logits = self.model(input_seq, padding_mask)
            
            # Get predictions for the last position
            last_idx = torch.sum(input_seq > 0, dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
            predictions = logits[0, last_idx[0]]
            
            # Get all item scores
            scores = self.model.predict(predictions.unsqueeze(0)).squeeze(0)
            
            # Convert to numpy for easier manipulation
            scores = scores.cpu().numpy()
        
        # Set scores of padding item to -inf
        scores[0] = -np.inf
        
        # Exclude seen items if requested
        if exclude_seen:
            original_seq = self.user_sequences[user_id]
            for item_idx in original_seq:
                scores[item_idx] = -np.inf
        
        # Apply popularity bias correction if enabled
        if self.item_popularity_bias:
            scores[1:] = scores[1:] - self.item_popularity
        
        # Get top-n item indices
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Convert indices back to original item IDs
        recommendations = [self.index_to_item[idx] for idx in top_indices]
        
        return recommendations
    
    def evaluate(self, eval_data, metrics: List[str] = None, cutoffs: List[int] = [5, 10, 20]):
        """
        Evaluate the model on test data.
        
        Parameters:
        - eval_data: Dictionary with user_id -> (input_sequence, ground_truth) pairs
        - metrics: List of metrics to compute (default: self.eval_metrics)
        - cutoffs: List of cutoff values for metrics
        
        Returns:
        - Dictionary with evaluation results
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if metrics is None:
            metrics = self.eval_metrics
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        metric_values = {f"{m}@{k}": 0.0 for m in metrics for k in cutoffs}
        n_users = 0
        
        for user_id, (input_seq, ground_truth) in eval_data.items():
            if user_id not in self.user_sequences:
                continue
            
            # Generate recommendations
            recs = self.recommend(user_id, top_n=max(cutoffs), exclude_seen=True)
            
            # Compute metrics
            for k in cutoffs:
                recs_at_k = recs[:k]
                
                # Hit rate
                if "hit" in metrics:
                    hit = int(any(item in ground_truth for item in recs_at_k))
                    metric_values[f"hit@{k}"] += hit
                
                # NDCG
                if "ndcg" in metrics:
                    ndcg = 0
                    for i, item in enumerate(recs_at_k):
                        if item in ground_truth:
                            ndcg += 1 / np.log2(i + 2)
                    if len(ground_truth) > 0:
                        idcg = min(len(ground_truth), k)
                        idcg = sum(1 / np.log2(i + 2) for i in range(idcg))
                        ndcg = ndcg / idcg if idcg > 0 else 0
                    metric_values[f"ndcg@{k}"] += ndcg
                
                # Precision
                if "precision" in metrics:
                    n_relevant = sum(1 for item in recs_at_k if item in ground_truth)
                    precision = n_relevant / k
                    metric_values[f"precision@{k}"] += precision
                
                # Recall
                if "recall" in metrics:
                    n_relevant = sum(1 for item in recs_at_k if item in ground_truth)
                    recall = n_relevant / len(ground_truth) if len(ground_truth) > 0 else 0
                    metric_values[f"recall@{k}"] += recall
            
            n_users += 1
        
        # Average metrics
        if n_users > 0:
            for metric in metric_values:
                metric_values[metric] /= n_users
        
        return metric_values
    
    def get_item_embeddings(self):
        """Return the learned item embeddings"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.model.get_all_embeddings()
    
    def export_item_embeddings(self, filepath=None):
        """Export item embeddings to a file"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        embeddings = self.get_item_embeddings()
        export_data = {
            'embeddings': embeddings,
            'index_to_item': self.index_to_item
        }
        
        if filepath is None:
            filepath = f"{self.name}_embeddings.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(export_data, f)
        
        self.logger.info(f"Item embeddings exported to {filepath}")
    
    def get_attention_weights(self, user_id):
        """Get attention weights for a user sequence for visualization"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if user_id not in self.user_sequences:
            raise ValueError(f"Unknown user: {user_id}")
        
        # Get user's sequence
        seq = self.user_sequences[user_id]
        
        # Create input sequence
        if len(seq) > self.max_seq_length:
            seq = seq[-self.max_seq_length:]
        else:
            seq = [0] * (self.max_seq_length - len(seq)) + seq
        
        # Create tensor
        input_seq = torch.LongTensor([seq]).to(self.device)
        padding_mask = (input_seq == 0)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate predictions with attention weights
        with torch.no_grad():
            _, attention_weights = self.model(input_seq, padding_mask, return_attention_weights=True)
        
        # Convert to numpy for easier manipulation
        attention_weights = [w.cpu().numpy() for w in attention_weights]
        
        return attention_weights, seq
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model_config': {
                'hidden_units': self.hidden_units,
                'num_blocks': self.num_blocks,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate,
                'max_seq_length': self.max_seq_length,
                'attention_type': self.attention_type,
                'position_encoding': self.position_encoding,
                'activation': self.activation
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'l2_reg': self.l2_reg,
                'loss_type': self.loss_type,
                'user_cooling': self.user_cooling,
                'item_popularity_bias': self.item_popularity_bias
            },
            'item_to_index': self.item_to_index,
            'index_to_item': self.index_to_item,
            'user_sequences': self.user_sequences,
            'model_state_dict': self.model.state_dict(),
            'item_popularity': self.item_popularity if self.item_popularity_bias else None,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
        @classmethod
        def load_model(cls, filepath: str, device: str = None) -> 'SASRec':
            """
            Load a model from a file.
            
            Parameters
            ----------
            filepath : str
                Path to the saved model.
            device : str, optional
                Device to load the model on
                
            Returns
            -------
            SASRec
                Loaded model.
            """
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
            model_data = torch.load(filepath, map_location=device)
            
            # Create a new instance
            instance = cls(
                name=os.path.basename(filepath).split('.')[0],
                hidden_units=model_data['model_config']['hidden_units'],
                num_blocks=model_data['model_config']['num_blocks'],
                num_heads=model_data['model_config']['num_heads'],
                dropout_rate=model_data['model_config']['dropout_rate'],
                max_seq_length=model_data['model_config']['max_seq_length'],
                attention_type=model_data['model_config']['attention_type'],
                position_encoding=model_data['model_config']['position_encoding'],
                activation=model_data['model_config']['activation'],
                learning_rate=model_data['training_config']['learning_rate'],
                batch_size=model_data['training_config']['batch_size'],
                num_epochs=model_data['training_config']['num_epochs'],
                l2_reg=model_data['training_config']['l2_reg'],
                loss_type=model_data['training_config']['loss_type'],
                device=device
            )
            
            # Restore mappings and user sequences
            instance.item_to_index = model_data['item_to_index']
            instance.index_to_item = model_data['index_to_item']
            instance.user_sequences = model_data['user_sequences']
            
            # Set item popularity if available
            if 'item_popularity' in model_data and model_data['item_popularity'] is not None:
                instance.item_popularity = model_data['item_popularity']
                instance.item_popularity_bias = True
            
            # Restore training history if available
            if 'training_history' in model_data:
                instance.training_history = model_data['training_history']
            
            # Recreate the model
            n_items = len(instance.item_to_index)
            instance.model = SASRecModel(
                n_items=n_items,
                hidden_units=instance.hidden_units,
                num_blocks=instance.num_blocks,
                num_heads=instance.num_heads,
                dropout_rate=instance.dropout_rate,
                max_seq_length=instance.max_seq_length,
                attention_type=instance.attention_type,
                position_encoding=instance.position_encoding,
                activation=instance.activation
            ).to(instance.device)
            
            # Load the model weights
            instance.model.load_state_dict(model_data['model_state_dict'])
            instance.model.eval()
            instance.is_fitted = True
            
            return instance