import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from scipy.sparse import csr_matrix
import os
import pickle
import json

from .base_recommender import BaseRecommender


class SLiRec(BaseRecommender):
    """
    Short-term and Long-term Preference Integrated Recommender (SLi-Rec)
    
    This model integrates both short-term and long-term user preferences for sequential recommendation.
    It uses attention mechanisms to capture user's evolving interests over time.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of item and user embeddings
    hidden_dim : int
        Dimension of hidden layers
    num_layers : int
        Number of layers in LSTM
    dropout_rate : float
        Dropout probability
    attention_size : int
        Size of attention layer
    l2_reg : float
        L2 regularization coefficient
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs for training
    sequence_length : int
        Maximum length of user interaction sequences
    device : str
        Device to run the model on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
        attention_size: int = 64,
        l2_reg: float = 1e-5,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        sequence_length: int = 50,
        device: str = 'cpu'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.attention_size = attention_size
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        
        # Mappings
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_id_map = {}
        self.reverse_item_id_map = {}
        
        # Model components will be initialized in _build_model
        self.model = None
        self.optimizer = None
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        
        self.reverse_user_id_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_id_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)
        
    def _build_model(self) -> None:
        """
        Build the SLiRec model components.
        """
        # User and item embeddings
        self.user_embeddings = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embeddings = nn.Embedding(self.n_items + 1, self.embedding_dim)  # +1 for padding
        
        # LSTM for sequence modeling (short-term)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_size),
            nn.Tanh(),
            nn.Linear(self.attention_size, 1, bias=False)
        )
        
        # Integration layer for short and long-term preferences
        self.integration_layer = nn.Sequential(
            nn.Linear(self.hidden_dim + self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Move model to device
        self.user_embeddings.to(self.device)
        self.item_embeddings.to(self.device)
        self.lstm.to(self.device)
        self.attention_layer.to(self.device)
        self.integration_layer.to(self.device)
        self.output_layer.to(self.device)
        
        # Optimizer
        model_params = list(self.user_embeddings.parameters()) + \
                      list(self.item_embeddings.parameters()) + \
                      list(self.lstm.parameters()) + \
                      list(self.attention_layer.parameters()) + \
                      list(self.integration_layer.parameters()) + \
                      list(self.output_layer.parameters())
        
        self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.l2_reg)
    
    def _attention_net(self, lstm_output: torch.Tensor, final_state: torch.Tensor) -> torch.Tensor:
        """
        Attention network to focus on relevant parts of the sequence.
        
        Parameters
        ----------
        lstm_output : torch.Tensor
            Output of LSTM layer (batch_size, seq_len, hidden_dim)
        final_state : torch.Tensor
            Final hidden state of LSTM (batch_size, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Context vector after applying attention
        """
        batch_size, seq_len, hidden_dim = lstm_output.size()
        
        # Calculate attention scores
        attention_weights = self.attention_layer(lstm_output.reshape(-1, hidden_dim))
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights to get context vector
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        context = context.squeeze(1)
        
        return context
    
    def _get_sequences(self, interaction_matrix: csr_matrix) -> Tuple[List[List[int]], List[int]]:
        """
        Extract user interaction sequences from the interaction matrix.
        
        Parameters
        ----------
        interaction_matrix : csr_matrix
            User-item interaction matrix
            
        Returns
        -------
        Tuple[List[List[int]], List[int]]
            User interaction sequences and corresponding user indices
        """
        sequences = []
        user_indices = []
        
        for user_idx in range(interaction_matrix.shape[0]):
            items = interaction_matrix[user_idx].indices
            if len(items) > 0:
                sequences.append(items.tolist())
                user_indices.append(user_idx)
        
        return sequences, user_indices
    
    def _prepare_batch(self, sequences: List[List[int]], user_indices: List[int], batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of sequences for training.
        
        Parameters
        ----------
        sequences : List[List[int]]
            List of user interaction sequences
        user_indices : List[int]
            List of user indices
        batch_indices : List[int]
            Indices of sequences to include in the batch
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Input sequences, target items, sequence lengths, and user indices
        """
        batch_sequences = [sequences[i] for i in batch_indices]
        batch_users = [user_indices[i] for i in batch_indices]
        
        # Prepare sequences for training (use all but last item as input, last item as target)
        input_sequences = []
        target_items = []
        seq_lengths = []
        
        for seq in batch_sequences:
            if len(seq) > 1:  # Need at least 2 items (1 for input, 1 for target)
                input_seq = seq[:-1][-self.sequence_length:]  # Take up to sequence_length items
                target = seq[-1]
                
                input_sequences.append(input_seq)
                target_items.append(target)
                seq_lengths.append(len(input_seq))
        
        # Pad sequences
        max_len = max(seq_lengths)
        padded_sequences = []
        
        for seq in input_sequences:
            padded_seq = seq + [self.n_items] * (max_len - len(seq))  # Use n_items as padding index
            padded_sequences.append(padded_seq)
        
        return (
            torch.LongTensor(padded_sequences).to(self.device),
            torch.LongTensor(target_items).to(self.device),
            torch.LongTensor(seq_lengths).to(self.device),
            torch.LongTensor(batch_users).to(self.device)
        )
    
    def _forward(self, user_indices: torch.Tensor, item_sequences: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SLiRec model.
        
        Parameters
        ----------
        user_indices : torch.Tensor
            Indices of users
        item_sequences : torch.Tensor
            Padded sequences of item indices
        seq_lengths : torch.Tensor
            Actual lengths of sequences
            
        Returns
        -------
        torch.Tensor
            Output item embeddings for prediction
        """
        batch_size = user_indices.size(0)
        
        # Get user embeddings (long-term preferences)
        user_emb = self.user_embeddings(user_indices)  # (batch_size, embedding_dim)
        
        # Get item embeddings from sequences
        item_emb = self.item_embeddings(item_sequences)  # (batch_size, seq_len, embedding_dim)
        
        # Apply dropout to embeddings
        item_emb = self.dropout(item_emb)
        
        # Pack padded sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            item_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)
        
        # Unpack output
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get final hidden state
        hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Apply attention to get short-term preferences
        short_term_pref = self._attention_net(lstm_output, hidden)
        
        # Integrate short-term and long-term preferences
        integrated_pref = self.integration_layer(torch.cat([short_term_pref, user_emb], dim=1))
        
        # Final output projection
        output = self.output_layer(integrated_pref)
        
        return output
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the SLiRec model.
        
        Parameters
        ----------
        interaction_matrix : csr_matrix
            User-item interaction matrix
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Build model
        self._build_model()
        
        # Extract sequences
        sequences, user_indices = self._get_sequences(interaction_matrix)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found in the interaction matrix")
        
        # Training loop
        n_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(len(sequences))
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(sequences))
                batch_indices = indices[start_idx:end_idx]
                
                # Prepare batch
                item_seq, target_items, seq_lengths, batch_users = self._prepare_batch(
                    sequences, user_indices, batch_indices
                )
                
                if len(seq_lengths) == 0:
                    continue  # Skip empty batches
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self._forward(batch_users, item_seq, seq_lengths)
                
                # Compute loss
                all_items = self.item_embeddings.weight[:-1]  # Exclude padding
                scores = torch.matmul(output, all_items.t())
                
                target_one_hot = F.one_hot(target_items, num_classes=self.n_items).float()
                loss = F.cross_entropy(scores, target_items)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate recommendations for a user.
        
        Parameters
        ----------
        user_id : int
            ID of the user
        top_n : int
            Number of recommendations to generate
        exclude_seen : bool
            Whether to exclude items the user has already interacted with
            
        Returns
        -------
        List[int]
            List of recommended item IDs
        """
        if user_id not in self.user_id_map:
            return []  # User not in training data
        
        user_idx = self.user_id_map[user_id]
        
        # Get user's interaction history
        user_items = []
        if exclude_seen and hasattr(self, 'interaction_matrix'):
            user_items = self.interaction_matrix[user_idx].indices.tolist()
        
        # Convert user index to tensor
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        
        with torch.no_grad():
            # Get user embedding (long-term preference)
            user_emb = self.user_embeddings(user_tensor)
            
            # If user has no history, use only long-term preference
            if not hasattr(self, 'interaction_matrix') or len(self.interaction_matrix[user_idx].indices) == 0:
                output = self.output_layer(
                    self.integration_layer(
                        torch.cat([torch.zeros(1, self.hidden_dim).to(self.device), user_emb], dim=1)
                    )
                )
            else:
                # Get user's sequence
                seq = self.interaction_matrix[user_idx].indices.tolist()[-self.sequence_length:]
                seq_length = len(seq)
                
                # Pad sequence if needed
                if seq_length < self.sequence_length:
                    seq = seq + [self.n_items] * (self.sequence_length - seq_length)
                
                # Convert to tensors
                item_seq = torch.LongTensor([seq]).to(self.device)
                seq_length_tensor = torch.LongTensor([seq_length]).to(self.device)
                
                # Forward pass
                output = self._forward(user_tensor, item_seq, seq_length_tensor)
            
            # Compute scores for all items
            all_items = self.item_embeddings.weight[:-1]  # Exclude padding
            scores = torch.matmul(output, all_items.t()).squeeze(0)
            
            # Convert scores to numpy for processing
            scores = scores.cpu().numpy()
            
            # Exclude seen items if required
            if exclude_seen:
                scores[user_items] = -np.inf
            
            # Get top-n items
            top_item_indices = np.argsort(-scores)[:top_n]
            
            # Convert indices back to original item IDs
            recommendations = [self.reverse_item_id_map[idx] for idx in top_item_indices]
            
            return recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        model_dict = {
            'user_embeddings': self.user_embeddings.state_dict(),
            'item_embeddings': self.item_embeddings.state_dict(),
            'lstm': self.lstm.state_dict(),
            'attention_layer': self.attention_layer.state_dict(),
            'integration_layer': self.integration_layer.state_dict(),
            'output_layer': self.output_layer.state_dict(),
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_id_map': self.reverse_user_id_map,
            'reverse_item_id_map': self.reverse_item_id_map,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'hyperparams': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'attention_size': self.attention_size,
                'l2_reg': self.l2_reg,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'sequence_length': self.sequence_length,
                'device': str(self.device)
            }
        }
        
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SLiRec':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        SLiRec
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        hyperparams = model_dict['hyperparams']
        
        # Create instance with saved hyperparameters
        instance = cls(
            embedding_dim=hyperparams['embedding_dim'],
            hidden_dim=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            dropout_rate=hyperparams['dropout_rate'],
            attention_size=hyperparams['attention_size'],
            l2_reg=hyperparams['l2_reg'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            epochs=hyperparams['epochs'],
            sequence_length=hyperparams['sequence_length'],
            device=hyperparams['device']
        )
        
        # Restore mappings
        instance.user_id_map = model_dict['user_id_map']
        instance.item_id_map = model_dict['item_id_map']
        instance.reverse_user_id_map = model_dict['reverse_user_id_map']
        instance.reverse_item_id_map = model_dict['reverse_item_id_map']
        instance.n_users = model_dict['n_users']
        instance.n_items = model_dict['n_items']
        
        # Build model
        instance._build_model()
        
        # Load state dictionaries
        instance.user_embeddings.load_state_dict(model_dict['user_embeddings'])
        instance.item_embeddings.load_state_dict(model_dict['item_embeddings'])
        instance.lstm.load_state_dict(model_dict['lstm'])
        instance.attention_layer.load_state_dict(model_dict['attention_layer'])
        instance.integration_layer.load_state_dict(model_dict['integration_layer'])
        instance.output_layer.load_state_dict(model_dict['output_layer'])
        
        return instance
