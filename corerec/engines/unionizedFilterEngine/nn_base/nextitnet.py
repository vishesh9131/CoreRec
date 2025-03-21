import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import pickle
from scipy.sparse import csr_matrix
from collections import defaultdict

from ..base_recommender import BaseRecommender


class ResidualBlock(nn.Module):
    """
    Residual block with dilated convolutions for NextItNet.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        """
        Initialize a residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for the convolutions
            dilation: Dilation factor for the convolutions
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation
        )
        
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # First convolutional layer
        out = self.conv1(x)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, out_channels)
        out = self.layer_norm1(out)
        out = out.permute(0, 2, 1)  # (batch_size, out_channels, seq_len)
        out = self.relu(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, out_channels)
        out = self.layer_norm2(out)
        out = out.permute(0, 2, 1)  # (batch_size, out_channels, seq_len)
        out = self.relu(out)
        
        # Residual connection
        return out + x


class NextItNetModel(nn.Module):
    """
    NextItNet model for next item recommendation.
    """
    
    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8, 1, 2, 4, 8],
        dropout: float = 0.2
    ):
        """
        Initialize the NextItNet model.
        
        Args:
            n_items: Number of items in the dataset
            embedding_dim: Size of item embeddings
            kernel_size: Kernel size for the convolutional layers
            dilations: List of dilation factors for the residual blocks
            dropout: Dropout rate
        """
        super(NextItNetModel, self).__init__()
        
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Item embedding layer
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=kernel_size,
                dilation=dilation
            ) for dilation in dilations
        ])
        
        # Final prediction layer
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(embedding_dim, n_items + 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.final_layer.weight)
        
        if self.final_layer.bias is not None:
            nn.init.zeros_(self.final_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NextItNet model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, n_items)
        """
        # Embedding layer
        x = self.item_embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final prediction
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, embedding_dim)
        x = self.dropout(x)
        x = self.final_layer(x)  # (batch_size, seq_len, n_items)
        
        return x


class NextItNet(BaseRecommender):
    """
    NextItNet recommender for next item prediction.
    
    This implements the NextItNet model from the paper:
    "A Simple Convolutional Generative Network for Next Item Recommendation" by Yuan et al. (2019)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8, 1, 2, 4, 8],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        num_epochs: int = 20,
        sequence_length: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the NextItNet recommender.
        
        Args:
            embedding_dim: Size of item embeddings
            kernel_size: Kernel size for the convolutional layers
            dilations: List of dilation factors for the residual blocks
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            sequence_length: Length of item sequences
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sequence_length = sequence_length
        self.device = device
        
        self.model = None
        self.item_id_map = {}
        self.user_sequences = {}
    
    def _create_mappings(self, item_ids: List[int]) -> None:
        """
        Create mappings from original item IDs to internal indices.
        
        Args:
            item_ids: List of item IDs
        """
        unique_item_ids = sorted(set(item_ids))
        
        # 0 is reserved for padding
        self.item_id_map = {item_id: idx + 1 for idx, item_id in enumerate(unique_item_ids)}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        self.reverse_item_map[0] = None  # Padding item
    
    def _build_user_sequences(
        self, 
        user_ids: List[int], 
        item_ids: List[int], 
        timestamps: Optional[List[int]] = None
    ) -> None:
        """
        Build sequences of items for each user, ordered by timestamp if available.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            timestamps: List of timestamps (optional)
        """
        # Group items by user
        user_items = defaultdict(list)
        
        if timestamps is not None:
            # If timestamps are available, use them to order items
            for user_id, item_id, timestamp in zip(user_ids, item_ids, timestamps):
                user_items[user_id].append((item_id, timestamp))
            
            # Sort items by timestamp for each user
            for user_id in user_items:
                user_items[user_id].sort(key=lambda x: x[1])
                user_items[user_id] = [item_id for item_id, _ in user_items[user_id]]
        else:
            # If no timestamps, assume items are already in chronological order
            for user_id, item_id in zip(user_ids, item_ids):
                user_items[user_id].append(item_id)
        
        # Convert item IDs to internal indices
        self.user_sequences = {
            user_id: [self.item_id_map[item_id] for item_id in items]
            for user_id, items in user_items.items()
        }
    
    def _generate_training_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate training data from user sequences.
        
        Returns:
            List of (input_sequence, target_sequence) pairs
        """
        training_data = []
        
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) < self.sequence_length + 1:
                # Skip sequences that are too short
                continue
            
            # Generate sliding windows
            for i in range(len(sequence) - self.sequence_length):
                input_seq = sequence[i:i + self.sequence_length]
                target_seq = sequence[i + 1:i + self.sequence_length + 1]
                
                training_data.append((
                    torch.LongTensor(input_seq),
                    torch.LongTensor(target_seq)
                ))
        
        return training_data
    
    def fit(
        self, 
        user_ids: List[int], 
        item_ids: List[int], 
        timestamps: Optional[List[int]] = None
    ) -> None:
        """
        Train the NextItNet model.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            timestamps: List of timestamps (optional)
        """
        # Create item ID mappings
        self._create_mappings(item_ids)
        
        # Build user sequences
        self._build_user_sequences(user_ids, item_ids, timestamps)
        
        # Generate training data
        training_data = self._generate_training_data()
        
        if not training_data:
            raise ValueError("No valid training sequences found. Try reducing sequence_length.")
        
        # Initialize model
        n_items = len(self.item_id_map)
        self.model = NextItNetModel(
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        self.model.train()
        n_batches = (len(training_data) + self.batch_size - 1) // self.batch_size
        
        for epoch in range(self.num_epochs):
            # Shuffle training data
            np.random.shuffle(training_data)
            
            total_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(training_data))
                batch_data = training_data[start_idx:end_idx]
                
                # Prepare batch
                input_seqs = torch.stack([data[0] for data in batch_data]).to(self.device)
                target_seqs = torch.stack([data[1] for data in batch_data]).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_seqs)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, n_items + 1)
                targets = target_seqs.view(-1)
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Generate next item recommendations for a user.
        
        Args:
            user_id: User ID
            top_n: Number of recommendations to generate
            
        Returns:
            List of recommended item IDs
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_sequences:
            # If user is not in the training data, return empty list
            return []
        
        # Get the user's sequence
        sequence = self.user_sequences[user_id]
        
        # Use the last 'sequence_length' items as input
        if len(sequence) < self.sequence_length:
            # Pad sequence if it's shorter than sequence_length
            input_seq = [0] * (self.sequence_length - len(sequence)) + sequence
        else:
            input_seq = sequence[-self.sequence_length:]
        
        # Convert to tensor
        input_tensor = torch.LongTensor([input_seq]).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Get predictions for the last position
            predictions = outputs[0, -1].cpu().numpy()
        
        # Set score of padding item to negative infinity
        predictions[0] = float('-inf')
        
        # Set scores of items in the input sequence to negative infinity to avoid recommending seen items
        for item_idx in input_seq:
            if item_idx > 0:  # Skip padding
                predictions[item_idx] = float('-inf')
        
        # Get top-N item indices
        top_indices = np.argsort(-predictions)[:top_n]
        
        # Convert indices back to original item IDs
        recommended_items = [self.reverse_item_map[idx] for idx in top_indices]
        
        return recommended_items
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'item_id_map': self.item_id_map,
            'reverse_item_map': self.reverse_item_map,
            'user_sequences': self.user_sequences,
            'embedding_dim': self.embedding_dim,
            'kernel_size': self.kernel_size,
            'dilations': self.dilations,
            'dropout': self.dropout,
            'sequence_length': self.sequence_length
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NextItNet':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded NextItNet model
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            embedding_dim=model_state['embedding_dim'],
            kernel_size=model_state['kernel_size'],
            dilations=model_state['dilations'],
            dropout=model_state['dropout'],
            sequence_length=model_state['sequence_length']
        )
        
        # Restore mappings and user sequences
        instance.item_id_map = model_state['item_id_map']
        instance.reverse_item_map = model_state['reverse_item_map']
        instance.user_sequences = model_state['user_sequences']
        
        # Recreate the model
        n_items = len(instance.item_id_map)
        instance.model = NextItNetModel(
            n_items=n_items,
            embedding_dim=instance.embedding_dim,
            kernel_size=instance.kernel_size,
            dilations=instance.dilations,
            dropout=instance.dropout
        ).to(instance.device)
        
        # Load the model weights
        instance.model.load_state_dict(model_state['model_state_dict'])
        instance.model.eval()
        
        return instance 