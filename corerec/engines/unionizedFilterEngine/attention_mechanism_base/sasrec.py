import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from scipy.sparse import csr_matrix
import os
import pickle
import math

from ..base_recommender import BaseRecommender

class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward network for SASRec.
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, L)
        outputs += inputs
        return outputs

class SASRecModel(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.
    """
    def __init__(
        self, 
        n_items, 
        hidden_units=64, 
        num_blocks=2, 
        num_heads=1, 
        dropout_rate=0.1, 
        max_seq_length=50
    ):
        super(SASRecModel, self).__init__()
        
        self.n_items = n_items
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        
        # Item embedding
        self.item_emb = nn.Embedding(n_items + 1, hidden_units, padding_idx=0)  # +1 for padding
        
        # Position embedding
        self.pos_emb = nn.Embedding(max_seq_length, hidden_units)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_units)
        self.layer_norm2 = nn.LayerNorm(hidden_units)
        
        # Multi-head attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            PointWiseFeedForward(hidden_units, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Final layer normalization
        self.last_layer_norm = nn.LayerNorm(hidden_units)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_units, n_items + 1)
        
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
    
    def forward(self, input_seqs, mask=None):
        # Get sequence length
        seq_length = input_seqs.size(1)
        
        # Create position indices
        positions = torch.arange(seq_length, dtype=torch.long, device=input_seqs.device)
        positions = positions.unsqueeze(0).expand_as(input_seqs)
        
        # Get item and position embeddings
        seq_emb = self.item_emb(input_seqs)
        pos_emb = self.pos_emb(positions)
        
        # Add position embeddings
        seq_emb = seq_emb + pos_emb
        
        # Apply dropout
        seq_emb = self.dropout(seq_emb)
        
        # Create attention mask
        if mask is None:
            mask = torch.ones((input_seqs.size(0), seq_length), device=input_seqs.device)
        
        # Create causal mask for self-attention
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=input_seqs.device) * float('-inf'),
            diagonal=1
        )
        
        # Apply attention blocks
        for i in range(self.num_blocks):
            # Layer normalization
            seq_emb = self.layer_norm1(seq_emb)
            
            # Self-attention
            seq_emb_t = seq_emb.transpose(0, 1)  # (L, B, E)
            attn_output, _ = self.attention_blocks[i](
                seq_emb_t, seq_emb_t, seq_emb_t,
                attn_mask=causal_mask,
                key_padding_mask=~mask.bool() if mask is not None else None
            )
            attn_output = attn_output.transpose(0, 1)  # (B, L, E)
            
            # Residual connection
            seq_emb = seq_emb + attn_output
            
            # Layer normalization
            seq_emb = self.layer_norm2(seq_emb)
            
            # Feed-forward network
            seq_emb = self.feed_forwards[i](seq_emb)
        
        # Final layer normalization
        seq_emb = self.last_layer_norm(seq_emb)
        
        # Output layer
        logits = self.output_layer(seq_emb)
        
        return logits


class SASRec(BaseRecommender):
    """
    Self-Attentive Sequential Recommendation (SASRec)
    
    SASRec is a sequential recommendation model that uses self-attention
    to capture long-term semantics and predict the next items in a sequence.
    
    Parameters
    ----------
    hidden_units : int, optional
        Size of hidden units, by default 64
    num_blocks : int, optional
        Number of transformer blocks, by default 2
    num_heads : int, optional
        Number of attention heads, by default 1
    dropout_rate : float, optional
        Dropout rate, by default 0.1
    max_seq_length : int, optional
        Maximum sequence length, by default 50
    learning_rate : float, optional
        Learning rate, by default 0.001
    batch_size : int, optional
        Batch size, by default 128
    num_epochs : int, optional
        Number of training epochs, by default 200
    l2_reg : float, optional
        L2 regularization coefficient, by default 0.0
    device : str, optional
        Device to use for training ('cpu' or 'cuda'), by default 'cpu'
    """
    
    def __init__(
        self,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
        max_seq_length: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        num_epochs: int = 200,
        l2_reg: float = 0.0,
        device: str = 'cpu'
    ):
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.l2_reg = l2_reg
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        self.item_to_index = {}
        self.index_to_item = {}
        self.user_sequences = {}
        self.model = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int], timestamps: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs.
        item_ids : List[int]
            List of item IDs.
        timestamps : List[int]
            List of timestamps.
        """
        # Create item mapping (0 is reserved for padding)
        unique_items = sorted(set(item_ids))
        self.item_to_index = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.index_to_item = {idx: item for item, idx in self.item_to_index.items()}
        
        # Create user sequences
        user_seq_dict = {}
        for user, item, timestamp in zip(user_ids, item_ids, timestamps):
            if user not in user_seq_dict:
                user_seq_dict[user] = []
            user_seq_dict[user].append((item, timestamp))
        
        # Sort sequences by timestamp and convert items to indices
        self.user_sequences = {}
        for user, seq in user_seq_dict.items():
            sorted_seq = [self.item_to_index[item] for item, _ in sorted(seq, key=lambda x: x[1])]
            self.user_sequences[user] = sorted_seq
    
    def _generate_training_data(self) -> List[Tuple[int, List[int], int]]:
        """
        Generate training data from user sequences.
        
        Returns
        -------
        List[Tuple[int, List[int], int]]
            List of (user_id, sequence, target_item) tuples.
        """
        train_data = []
        
        for user, seq in self.user_sequences.items():
            if len(seq) < 2:  # Need at least 2 items (1 for input, 1 for target)
                continue
                
            for i in range(1, len(seq)):
                # Use items up to i-1 as input and item i as target
                input_seq = seq[:i]
                target = seq[i]
                
                # Truncate input sequence if it's longer than max_seq_length
                if len(input_seq) > self.max_seq_length:
                    input_seq = input_seq[-self.max_seq_length:]
                
                train_data.append((user, input_seq, target))
        
        return train_data
    
    def _prepare_batch(self, batch_data: List[Tuple[int, List[int], int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of data for training.
        
        Parameters
        ----------
        batch_data : List[Tuple[int, List[int], int]]
            Batch of (user_id, sequence, target_item) tuples.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Input sequences and target items.
        """
        batch_size = len(batch_data)
        
        # Get max sequence length in this batch
        max_len = max([len(seq) for _, seq, _ in batch_data])
        max_len = min(max_len, self.max_seq_length)
        
        # Initialize input and target tensors
        input_seqs = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Fill tensors
        for i, (_, seq, target) in enumerate(batch_data):
            # Truncate sequence if needed
            if len(seq) > max_len:
                seq = seq[-max_len:]
            
            # Pad sequence
            input_seqs[i, -len(seq):] = torch.tensor(seq, dtype=torch.long, device=self.device)
            targets[i] = target
        
        return input_seqs, targets
    
    def fit(self, user_ids: List[int], item_ids: List[int], timestamps: List[int]) -> None:
        """
        Fit the SASRec model.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs.
        item_ids : List[int]
            List of item IDs.
        timestamps : List[int]
            List of timestamps.
        """
        # Create mappings and user sequences
        self._create_mappings(user_ids, item_ids, timestamps)
        
        # Initialize model
        n_items = len(self.item_to_index)
        self.model = SASRecModel(
            n_items=n_items,
            hidden_units=self.hidden_units,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            max_seq_length=self.max_seq_length
        ).to(self.device)
        
        # Generate training data
        train_data = self._generate_training_data()
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        
        # Training loop
        self.model.train()
        n_train = len(train_data)
        
        for epoch in range(self.num_epochs):
            # Shuffle training data
            indices = np.random.permutation(n_train)
            
            # Initialize metrics
            epoch_loss = 0
            
            # Process batches
            for i in range(0, n_train, self.batch_size):
                # Get batch indices
                batch_indices = indices[i:min(i + self.batch_size, n_train)]
                batch_data = [train_data[idx] for idx in batch_indices]
                
                # Prepare batch
                input_seqs, targets = self._prepare_batch(batch_data)
                
                # Forward pass
                logits = self.model(input_seqs)
                
                # Get predictions for the last item in each sequence
                last_item_indices = torch.sum(input_seqs > 0, dim=1) - 1
                last_item_indices = torch.clamp(last_item_indices, min=0)
                batch_indices = torch.arange(input_seqs.size(0), device=self.device)
                last_item_logits = logits[batch_indices, last_item_indices]
                
                # Compute loss
                loss = F.cross_entropy(last_item_logits, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item() * len(batch_data)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss / n_train:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend items for a user.
        
        Parameters
        ----------
        user_id : int
            User ID.
        top_n : int, optional
            Number of recommendations to return. Default is 10.
        exclude_seen : bool, optional
            Whether to exclude items the user has already interacted with.
            Default is True.
            
        Returns
        -------
        List[int]
            List of recommended item IDs.
        """
        if user_id not in self.user_sequences:
            return []
        
        # Get user's sequence
        seq = self.user_sequences[user_id]
        
        # Use the last max_seq_length items as input
        if len(seq) > self.max_seq_length:
            seq = seq[-self.max_seq_length:]
        
        # Prepare input
        input_seq = torch.zeros((1, self.max_seq_length), dtype=torch.long, device=self.device)
        input_seq[0, -len(seq):] = torch.tensor(seq, dtype=torch.long, device=self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_seq)
            
            # Get predictions for the last item
            last_idx = torch.sum(input_seq > 0, dim=1) - 1
            last_idx = torch.clamp(last_idx, min=0)
            predictions = logits[0, last_idx[0]]
            
            # Convert to numpy for easier manipulation
            predictions = predictions.cpu().numpy()
        
        # Set scores of padding item to -inf
        predictions[0] = -np.inf
        
        # Exclude seen items if requested
        if exclude_seen:
            for item_idx in seq:
                predictions[item_idx] = -np.inf
        
        # Get top-n item indices
        top_indices = np.argsort(predictions)[::-1][:top_n]
        
        # Convert indices back to original item IDs
        recommendations = [self.index_to_item[idx] for idx in top_indices]
        
        return recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'hidden_units': self.hidden_units,
            'num_blocks': self.num_blocks,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'max_seq_length': self.max_seq_length,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'l2_reg': self.l2_reg,
            'item_to_index': self.item_to_index,
            'index_to_item': self.index_to_item,
            'user_sequences': self.user_sequences,
            'model_state_dict': self.model.state_dict()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SASRec':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
            
        Returns
        -------
        SASRec
            Loaded model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            hidden_units=model_data['hidden_units'],
            num_blocks=model_data['num_blocks'],
            num_heads=model_data['num_heads'],
            dropout_rate=model_data['dropout_rate'],
            max_seq_length=model_data['max_seq_length'],
            learning_rate=model_data['learning_rate'],
            batch_size=model_data['batch_size'],
            num_epochs=model_data['num_epochs'],
            l2_reg=model_data['l2_reg']
        )
        
        # Restore mappings and user sequences
        instance.item_to_index = model_data['item_to_index']
        instance.index_to_item = model_data['index_to_item']
        instance.user_sequences = model_data['user_sequences']
        
        # Recreate the model
        n_items = len(instance.item_to_index)
        instance.model = SASRecModel(
            n_items=n_items,
            hidden_units=instance.hidden_units,
            num_blocks=instance.num_blocks,
            num_heads=instance.num_heads,
            dropout_rate=instance.dropout_rate,
            max_seq_length=instance.max_seq_length
        ).to(instance.device)
        
        # Load the model weights
        instance.model.load_state_dict(model_data['model_state_dict'])
        instance.model.eval()
        
        return instance
