import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from scipy.sparse import csr_matrix
import os
import pickle
import json
import math

from .base_recommender import BaseRecommender
from .device_manager import DeviceManager


class SUMModel(BaseRecommender):
    """
    Multi-Interest-Aware Sequential User Modeling (SUM)
    
    This model captures multiple user interests from sequential behavior using
    a capsule network architecture with dynamic routing.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of item embeddings
    num_interests : int
        Number of interest capsules to use
    interest_dim : int
        Dimension of interest capsules
    routing_iterations : int
        Number of routing iterations for capsule network
    dropout_rate : float
        Dropout probability
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
        Device to run the model on ('cpu', 'cuda', 'mps', etc.)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_interests: int = 4,
        interest_dim: int = 32,
        routing_iterations: int = 3,
        dropout_rate: float = 0.2,
        l2_reg: float = 1e-5,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        sequence_length: int = 50,
        device: str = 'auto'
    ):
        super().__init__(device=device)
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.interest_dim = interest_dim
        self.routing_iterations = routing_iterations
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        
        # Initialize device manager with MPS fallback
        self.device_manager = DeviceManager(preferred_device=device)
        self.device = self.device_manager.active_device
        
        # Enable MPS fallback for unsupported operations
        if self.device == 'mps':
            torch.backends.mps.enable_fallback_to_cpu = False
        
        self.torch_device = torch.device(self.device)
        
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
        Build the SUM model components.
        """
        # Item embeddings
        self.item_embeddings = nn.Embedding(self.n_items + 1, self.embedding_dim)
        
        # Transformation matrices for capsule network
        self.transform_matrices = nn.Parameter(
            torch.randn(self.num_interests, self.embedding_dim, self.interest_dim, 
                       device=self.torch_device)  # Initialize directly on device
        )
        
        # Move all components to device at creation
        self.attention_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim).to(self.torch_device),
            nn.Tanh().to(self.torch_device),
            nn.Linear(self.embedding_dim, 1, bias=False).to(self.torch_device)
        )
        
        self.interest_attention = nn.Sequential(
            nn.Linear(self.interest_dim, self.interest_dim).to(self.torch_device),
            nn.Tanh().to(self.torch_device),
            nn.Linear(self.interest_dim, 1, bias=False).to(self.torch_device)
        )
        
        self.output_projection = nn.Linear(self.interest_dim, self.embedding_dim).to(self.torch_device)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Move item embeddings to device
        self.item_embeddings = self.item_embeddings.to(self.torch_device)
        
        # Initialize optimizer after all parameters are on device
        model_params = (
            list(self.item_embeddings.parameters()) +
            [self.transform_matrices] +
            list(self.attention_layer.parameters()) +
            list(self.interest_attention.parameters()) +
            list(self.output_projection.parameters())
        )
        
        self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.l2_reg)
    
    def _squash(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Squashing function for capsule network.
        
        Parameters
        ----------
        vectors : torch.Tensor
            Input vectors to squash
            
        Returns
        -------
        torch.Tensor
            Squashed vectors
        """
        squared_norm = torch.sum(vectors ** 2, dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * vectors / torch.sqrt(squared_norm + 1e-8)
    
    def _dynamic_routing(self, item_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Dynamic routing between capsules.
        
        Parameters:
        -----------
        item_embeddings: torch.Tensor
            Shape (batch_size, seq_len, embedding_dim)
        mask: torch.Tensor
            Shape (batch_size, seq_len)
            
        Returns:
        --------
        torch.Tensor
            Shape (batch_size, num_interests, interest_dim)
        """
        batch_size = item_embeddings.size(0)
        seq_len = item_embeddings.size(1)
        
        # Transform item embeddings by the transformation matrices
        try:
            # Reshape for transformation
            item_embeddings_expanded = item_embeddings.unsqueeze(1).expand(
                -1, self.num_interests, -1, -1
            )  # (batch_size, num_interests, seq_len, embedding_dim)
            
            # Transform using matrices
            u_hat = torch.matmul(
                item_embeddings_expanded, 
                self.transform_matrices
            )  # (batch_size, num_interests, seq_len, interest_dim)
            
        except RuntimeError as e:
            if "MPS" in str(e):
                # Fallback implementation for MPS
                u_hat = []
                for i in range(self.num_interests):
                    transform = self.transform_matrices[i]
                    u = torch.matmul(item_embeddings, transform)
                    u_hat.append(u)
                u_hat = torch.stack(u_hat, dim=1)
        
        # Initialize routing logits
        b = torch.zeros(
            batch_size, self.num_interests, seq_len, 
            device=self.torch_device
        ).unsqueeze(-1)  # (batch_size, num_interests, seq_len, 1)
        
        # Apply mask to routing logits
        mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, seq_len, 1)
        b = b.masked_fill(~mask.bool(), -1e9)
        
        # Dynamic routing iterations
        for i in range(self.routing_iterations):
            # Compute coupling coefficients
            c = F.softmax(b, dim=1)  # (batch_size, num_interests, seq_len, 1)
            
            # Compute weighted sum
            s = (c * u_hat).sum(dim=2)  # (batch_size, num_interests, interest_dim)
            
            # Apply squashing
            v = self._squash(s)  # (batch_size, num_interests, interest_dim)
            
            if i < self.routing_iterations - 1:
                # Update routing logits
                v_expanded = v.unsqueeze(2)  # (batch_size, num_interests, 1, interest_dim)
                agreement = torch.sum(
                    u_hat * v_expanded, dim=-1, keepdim=True
                )  # (batch_size, num_interests, seq_len, 1)
                b = b + agreement
        
        return v
    
    def _sequence_attention(self, item_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to sequence of item embeddings.
        
        Parameters
        ----------
        item_embeddings : torch.Tensor
            Item embeddings from sequence (batch_size, seq_len, embedding_dim)
        mask : torch.Tensor
            Mask for valid items in sequence (batch_size, seq_len)
            
        Returns
        -------
        torch.Tensor
            Weighted sum of item embeddings (batch_size, embedding_dim)
        """
        # Calculate attention scores
        attention_scores = self.attention_layer(item_embeddings)  # (batch_size, seq_len, 1)
        
        # Mask out padding
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        attention_scores = attention_scores.masked_fill(~mask.bool(), -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Weighted sum
        weighted_sum = torch.sum(attention_weights * item_embeddings, dim=1)  # (batch_size, embedding_dim)
        
        return weighted_sum
    
    def _interest_level_attention(self, interests: torch.Tensor, target_item_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply attention over multiple interests based on target item.
        
        Parameters
        ----------
        interests : torch.Tensor
            Interest capsules (batch_size, num_interests, interest_dim)
        target_item_emb : torch.Tensor
            Target item embeddings (batch_size, embedding_dim)
            
        Returns
        -------
        torch.Tensor
            Weighted sum of interests (batch_size, interest_dim)
        """
        # Project target item to interest space
        target_projection = torch.bmm(
            target_item_emb.unsqueeze(1),  # (batch_size, 1, embedding_dim)
            self.transform_matrices.mean(dim=0).unsqueeze(0).expand(target_item_emb.size(0), -1, -1)  # (batch_size, embedding_dim, interest_dim)
        ).squeeze(1)  # (batch_size, interest_dim)
        
        # Calculate similarity between interests and target
        similarities = torch.bmm(
            interests,  # (batch_size, num_interests, interest_dim)
            target_projection.unsqueeze(-1)  # (batch_size, interest_dim, 1)
        ).squeeze(-1)  # (batch_size, num_interests)
        
        # Apply softmax
        attention_weights = F.softmax(similarities, dim=1).unsqueeze(-1)  # (batch_size, num_interests, 1)
        
        # Weighted sum
        weighted_sum = torch.sum(attention_weights * interests, dim=1)  # (batch_size, interest_dim)
        
        return weighted_sum
    
    def _forward(self, item_seq: torch.Tensor, seq_mask: torch.Tensor, target_item: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Parameters
        ----------
        item_seq : torch.Tensor
            Sequence of item indices (batch_size, seq_len)
        seq_mask : torch.Tensor
            Mask for valid items in sequence (batch_size, seq_len)
        target_item : torch.Tensor, optional
            Target item indices (batch_size)
            
        Returns
        -------
        torch.Tensor
            Output embeddings for prediction (batch_size, embedding_dim)
        """
        # Get item embeddings
        seq_embeddings = self.item_embeddings(item_seq)  # (batch_size, seq_len, embedding_dim)
        seq_embeddings = self.dropout(seq_embeddings)
        
        # Extract multiple interests using capsule network
        interests = self._dynamic_routing(seq_embeddings, seq_mask)  # (batch_size, num_interests, interest_dim)
        
        if target_item is not None:
            # Get target item embeddings
            target_emb = self.item_embeddings(target_item)  # (batch_size, embedding_dim)
            
            # Apply interest-level attention
            interest_vector = self._interest_level_attention(interests, target_emb)  # (batch_size, interest_dim)
        else:
            # During inference, use attention over interests
            interest_scores = self.interest_attention(interests.view(-1, self.interest_dim)).view(-1, self.num_interests)
            interest_weights = F.softmax(interest_scores, dim=1).unsqueeze(-1)  # (batch_size, num_interests, 1)
            interest_vector = torch.sum(interest_weights * interests, dim=1)  # (batch_size, interest_dim)
        
        # Project back to embedding space
        output = self.output_projection(interest_vector)  # (batch_size, embedding_dim)
        
        return output
    
    def fit(self, user_ids: List[int], item_ids: List[int], timestamps: List[int]) -> None:
        """
        Train the model on the provided data.
        
        Parameters
        ----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        timestamps : List[int]
            List of timestamps
        """
        print(f"Training SUM model on device: {self.device}")
        
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Build model
        self._build_model()
        
        # Create user sequences
        user_sequences = {}
        for user_id, item_id, timestamp in zip(user_ids, item_ids, timestamps):
            if user_id not in self.user_id_map:
                continue
                
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
            
            if user_idx not in user_sequences:
                user_sequences[user_idx] = []
                
            user_sequences[user_idx].append((item_idx, timestamp))
        
        # Sort sequences by timestamp
        for user_idx in user_sequences:
            user_sequences[user_idx].sort(key=lambda x: x[1])
            user_sequences[user_idx] = [item for item, _ in user_sequences[user_idx]]
        
        # Create training data
        train_data = []
        for user_idx, sequence in user_sequences.items():
            if len(sequence) < 2:  # Need at least 2 items
                continue
                
            for i in range(1, len(sequence)):
                # Use items up to i-1 as input and item i as target
                seq = sequence[:i]
                target = sequence[i]
                
                # Truncate sequence if needed
                if len(seq) > self.sequence_length:
                    seq = seq[-self.sequence_length:]
                
                train_data.append((seq, target))
        
        # Training loop
        self.model = self  # For compatibility with BaseRecommender
        
        n_train = len(train_data)
        n_batches = (n_train + self.batch_size - 1) // self.batch_size
        
        try:
            for epoch in range(self.epochs):
                # Shuffle training data
                np.random.shuffle(train_data)
                
                total_loss = 0.0
                
                for i in range(n_batches):
                    batch_data = train_data[i * self.batch_size:(i + 1) * self.batch_size]
                    
                    # Prepare batch
                    max_seq_len = max(len(seq) for seq, _ in batch_data)
                    batch_seq = torch.zeros((len(batch_data), max_seq_len), dtype=torch.long).to(self.torch_device)
                    batch_mask = torch.zeros((len(batch_data), max_seq_len), dtype=torch.float).to(self.torch_device)
                    batch_target = torch.tensor([target for _, target in batch_data], dtype=torch.long).to(self.torch_device)
                    
                    for j, (seq, _) in enumerate(batch_data):
                        batch_seq[j, -len(seq):] = torch.tensor(seq, dtype=torch.long)
                        batch_mask[j, -len(seq):] = 1.0
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    
                    output = self._forward(batch_seq, batch_mask, batch_target)
                    
                    # Compute loss with potential MPS handling
                    all_items = self.item_embeddings.weight[:-1]
                    logits = torch.matmul(output, all_items.t())
                    loss = F.cross_entropy(logits, batch_target)
                    
                    # Backward pass and optimize
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item() * len(batch_data)
                
                avg_loss = total_loss / n_train
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # Store user sequences for recommendation
            self.user_sequences = user_sequences
            
        except RuntimeError as e:
            if "MPS" in str(e):
                print(f"MPS error encountered: {str(e)}")
                print("Falling back to CPU...")
                self.device = 'cpu'
                self.torch_device = torch.device('cpu')
                self._build_model()  # Rebuild model on CPU
                self.fit(user_ids, item_ids, timestamps)  # Retry training
            else:
                raise e
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend items for a user.
        
        Parameters
        ----------
        user_id : int
            User ID
        top_n : int
            Number of recommendations to return
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
        
        if user_idx not in self.user_sequences or len(self.user_sequences[user_idx]) == 0:
            return []  # User has no sequence
        
        # Get user's sequence
        sequence = self.user_sequences[user_idx]
        
        # Truncate sequence if needed
        if len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        
        # Prepare input
        seq_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(self.torch_device)
        mask_tensor = torch.ones((1, len(sequence)), dtype=torch.float).to(self.torch_device)
        
        # Get seen items
        seen_items = set(sequence) if exclude_seen else set()
        
        with torch.no_grad():
            # Forward pass
            output = self._forward(seq_tensor, mask_tensor)
            
            # Compute scores for all items
            all_items = self.item_embeddings.weight[:-1]  # Exclude padding
            scores = torch.matmul(output, all_items.t()).squeeze(0)
            
            # Convert to numpy for processing
            scores = scores.cpu().numpy()
            
            # Set scores of seen items to -inf
            for item_idx in seen_items:
                scores[item_idx] = -np.inf
            
            # Get top-n items
            top_indices = np.argsort(-scores)[:top_n]
            
            # Convert indices back to original item IDs
            recommendations = [self.reverse_item_id_map[idx] for idx in top_indices]
            
            return recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        # Move model to CPU for saving
        item_embeddings_state = self.item_embeddings.state_dict()
        transform_matrices_data = self.transform_matrices.data.cpu()
        attention_layer_state = self.attention_layer.state_dict()
        interest_attention_state = self.interest_attention.state_dict()
        output_projection_state = self.output_projection.state_dict()
        
        model_dict = {
            'item_embeddings': item_embeddings_state,
            'transform_matrices': transform_matrices_data,
            'attention_layer': attention_layer_state,
            'interest_attention': interest_attention_state,
            'output_projection': output_projection_state,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_id_map': self.reverse_user_id_map,
            'reverse_item_id_map': self.reverse_item_id_map,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_sequences': self.user_sequences,
            'hyperparams': {
                'embedding_dim': self.embedding_dim,
                'num_interests': self.num_interests,
                'interest_dim': self.interest_dim,
                'routing_iterations': self.routing_iterations,
                'dropout_rate': self.dropout_rate,
                'l2_reg': self.l2_reg,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'sequence_length': self.sequence_length,
                'device': self.device
            }
        }
        
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'auto') -> 'SUMModel':
        """
        Load a model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        device : str
            Device to load the model on
            
        Returns
        -------
        SUMModel
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        hyperparams = model_dict['hyperparams']
        
        # Create instance with saved hyperparameters but use the specified device
        instance = cls(
            embedding_dim=hyperparams['embedding_dim'],
            num_interests=hyperparams['num_interests'],
            interest_dim=hyperparams['interest_dim'],
            routing_iterations=hyperparams['routing_iterations'],
            dropout_rate=hyperparams['dropout_rate'],
            l2_reg=hyperparams['l2_reg'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            epochs=hyperparams['epochs'],
            sequence_length=hyperparams['sequence_length'],
            device=device  # Use the provided device
        )
        
        # Restore mappings
        instance.user_id_map = model_dict['user_id_map']
        instance.item_id_map = model_dict['item_id_map']
        instance.reverse_user_id_map = model_dict['reverse_user_id_map']
        instance.reverse_item_id_map = model_dict['reverse_item_id_map']
        instance.n_users = model_dict['n_users']
        instance.n_items = model_dict['n_items']
        instance.user_sequences = model_dict['user_sequences']
        
        # Build model
        instance._build_model()
        
        # Load state dictionaries
        instance.item_embeddings.load_state_dict(model_dict['item_embeddings'])
        instance.transform_matrices.data = model_dict['transform_matrices'].to(instance.torch_device)
        instance.attention_layer.load_state_dict(model_dict['attention_layer'])
        instance.interest_attention.load_state_dict(model_dict['interest_attention'])
        instance.output_projection.load_state_dict(model_dict['output_projection'])
        
        return instance
