import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.sparse import csr_matrix
import os
import pickle
from ..base_recommender import BaseRecommender


class GRUNet(nn.Module):
    def __init__(
        self, 
        num_items: int, 
        embedding_dim: int = 64, 
        hidden_dim: int = 128, 
        num_layers: int = 1, 
        dropout: float = 0.2
    ):
        super(GRUNet, self).__init__()
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_dim, num_items + 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, item_seq: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # item_seq shape: (batch_size, seq_len)
        embedded = self.dropout(self.item_embeddings(item_seq))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        output, hidden = self.gru(embedded, hidden)
        # output shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        
        output = self.dropout(output)
        logits = self.out(output)
        # logits shape: (batch_size, seq_len, num_items + 1)
        
        return logits, hidden


class GRUCF(BaseRecommender):
    """
    GRU-based Collaborative Filtering recommender.
    
    This model uses a Gated Recurrent Unit (GRU) to model user-item interactions as sequences,
    capturing temporal patterns in user behavior.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 0.001,
        batch_size: int = 64,
        epochs: int = 10,
        sequence_length: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state: Optional[int] = None
    ):
        """
        Initialize the GRU-based Collaborative Filtering model.
        
        Args:
            embedding_dim: Dimension of item embeddings
            hidden_dim: Dimension of hidden state in GRU
            num_layers: Number of GRU layers
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Batch size for training
            epochs: Number of training epochs
            sequence_length: Length of sequence for each user
            device: Device to run the model on ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.device = device
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        self.user_sequences = {}
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings between original IDs and internal indices.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        # Reserve index 0 for padding
        self.item_to_idx = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        
    def _prepare_sequences(self, user_ids: List[int], item_ids: List[int], timestamps: Optional[List[int]] = None) -> None:
        """
        Prepare user-item interaction sequences for training.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            timestamps: Optional list of timestamps for sorting interactions
        """
        # Group interactions by user
        user_items = {}
        for i, (user_id, item_id) in enumerate(zip(user_ids, item_ids)):
            if user_id not in user_items:
                user_items[user_id] = []
            
            if timestamps is not None:
                user_items[user_id].append((item_id, timestamps[i]))
            else:
                user_items[user_id].append((item_id, i))  # Use index as timestamp if not provided
        
        # Sort interactions by timestamp and create sequences
        self.user_sequences = {}
        for user_id, items in user_items.items():
            # Sort by timestamp
            sorted_items = [item for item, _ in sorted(items, key=lambda x: x[1])]
            # Convert to internal indices
            item_indices = [self.item_to_idx[item] for item in sorted_items]
            self.user_sequences[self.user_to_idx[user_id]] = item_indices
    
    def _generate_training_batches(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate training batches from user sequences.
        
        Returns:
            List of (input_sequence, target_sequence) tuples
        """
        batches = []
        sequences = []
        
        # Create sliding window sequences
        for user_idx, item_sequence in self.user_sequences.items():
            if len(item_sequence) < self.sequence_length + 1:
                # Pad sequences shorter than sequence_length + 1
                padded = [0] * (self.sequence_length + 1 - len(item_sequence)) + item_sequence
                sequences.append(padded)
            else:
                # Create sliding windows for longer sequences
                for i in range(len(item_sequence) - self.sequence_length):
                    sequences.append(item_sequence[i:i + self.sequence_length + 1])
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Create batches
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i + self.batch_size]
            if not batch_sequences:
                continue
                
            # Pad sequences in batch to same length
            max_len = max(len(seq) for seq in batch_sequences)
            padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in batch_sequences]
            
            # Convert to tensors
            batch_input = torch.tensor([seq[:-1] for seq in padded_sequences], dtype=torch.long, device=self.device)
            batch_target = torch.tensor([seq[1:] for seq in padded_sequences], dtype=torch.long, device=self.device)
            
            batches.append((batch_input, batch_target))
        
        return batches
    
    def fit(self, user_ids: List[int], item_ids: List[int], timestamps: Optional[List[int]] = None) -> None:
        """
        Train the GRU-based Collaborative Filtering model.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            timestamps: Optional list of timestamps
        """
        # Create mappings and prepare sequences
        self._create_mappings(user_ids, item_ids)
        self._prepare_sequences(user_ids, item_ids, timestamps)
        
        # Initialize model
        num_items = len(self.item_to_idx)
        self.model = GRUNet(
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            batches = self._generate_training_batches()
            
            for batch_input, batch_target in batches:
                self.optimizer.zero_grad()
                
                # Forward pass
                logits, _ = self.model(batch_input)
                
                # Reshape for cross entropy loss
                batch_size, seq_len, vocab_size = logits.size()
                logits_flat = logits.view(batch_size * seq_len, vocab_size)
                targets_flat = batch_target.view(-1)
                
                # Calculate loss
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
            
            avg_loss = total_loss / len(batches) if batches else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended item IDs
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if user_id not in self.user_to_idx:
            # Return empty list for unknown users
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's interaction history
        if user_idx not in self.user_sequences or not self.user_sequences[user_idx]:
            # Return empty list for users with no interactions
            return []
        
        # Get the most recent items in the user's history
        item_sequence = self.user_sequences[user_idx][-self.sequence_length:]
        if len(item_sequence) < self.sequence_length:
            # Pad sequence if shorter than sequence_length
            item_sequence = [0] * (self.sequence_length - len(item_sequence)) + item_sequence
        
        # Convert to tensor
        sequence_tensor = torch.tensor([item_sequence], dtype=torch.long, device=self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(sequence_tensor)
            # Get predictions from the last position in the sequence
            predictions = logits[:, -1, :]  # Shape: (1, num_items + 1)
            
            # Set scores of padding item to negative infinity
            predictions[:, 0] = float('-inf')
            
            if exclude_seen:
                # Set scores of seen items to negative infinity
                seen_items = set(self.user_sequences[user_idx])
                for item_idx in seen_items:
                    if item_idx > 0:  # Skip padding
                        predictions[:, item_idx] = float('-inf')
            
            # Get top-n item indices
            _, top_indices = torch.topk(predictions[0], k=min(top_n + 10, predictions.size(1) - 1))
            top_indices = top_indices.cpu().numpy()
            
            # Convert indices back to original item IDs
            recommended_items = []
            for idx in top_indices:
                if idx in self.idx_to_item:
                    recommended_items.append(self.idx_to_item[idx])
                    if len(recommended_items) >= top_n:
                        break
            
            return recommended_items
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'user_sequences': self.user_sequences,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'sequence_length': self.sequence_length,
            'device': self.device
        }
        
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'GRUCF':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded GRUCF model
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new instance with the saved hyperparameters
        instance = cls(
            embedding_dim=model_state['embedding_dim'],
            hidden_dim=model_state['hidden_dim'],
            num_layers=model_state['num_layers'],
            dropout=model_state['dropout'],
            lr=model_state['lr'],
            batch_size=model_state['batch_size'],
            epochs=model_state['epochs'],
            sequence_length=model_state['sequence_length'],
            device=model_state['device']
        )
        
        # Restore mappings and sequences
        instance.user_to_idx = model_state['user_to_idx']
        instance.item_to_idx = model_state['item_to_idx']
        instance.idx_to_item = model_state['idx_to_item']
        instance.user_sequences = model_state['user_sequences']
        
        # Initialize and restore model
        num_items = len(instance.item_to_idx)
        instance.model = GRUNet(
            num_items=num_items,
            embedding_dim=instance.embedding_dim,
            hidden_dim=instance.hidden_dim,
            num_layers=instance.num_layers,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(model_state['model_state_dict'])
        
        # Initialize and restore optimizer
        instance.optimizer = optim.Adam(instance.model.parameters(), lr=instance.lr)
        instance.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        # Set criterion
        instance.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        return instance