import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import pickle
from scipy.sparse import csr_matrix

from ..base_recommender import BaseRecommender


class NCFModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_mlp: bool = True
    ):
        """
        Neural Collaborative Filtering model.
        
        Args:
            n_users: Number of users in the dataset
            n_items: Number of items in the dataset
            embedding_dim: Size of embedding vectors
            layers: List of layer sizes for MLP component
            dropout: Dropout rate
            use_gmf: Whether to use Generalized Matrix Factorization component
            use_mlp: Whether to use Multi-Layer Perceptron component
        """
        super(NCFModel, self).__init__()
        
        # Ensure at least one component is used
        assert use_gmf or use_mlp, "At least one of GMF or MLP must be used"
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        
        # GMF component
        if self.use_gmf:
            self.user_gmf_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_gmf_embedding = nn.Embedding(n_items, embedding_dim)
        
        # MLP component
        if self.use_mlp:
            self.user_mlp_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_mlp_embedding = nn.Embedding(n_items, embedding_dim)
            
            # MLP layers
            self.mlp_layers = nn.ModuleList()
            input_size = 2 * embedding_dim
            
            for i, layer_size in enumerate(layers):
                self.mlp_layers.append(nn.Linear(input_size, layer_size))
                self.mlp_layers.append(nn.ReLU())
                self.mlp_layers.append(nn.Dropout(dropout))
                input_size = layer_size
        
        # Final prediction layer
        final_layer_input = 0
        if self.use_gmf:
            final_layer_input += embedding_dim
        if self.use_mlp:
            final_layer_input += layers[-1]
            
        self.final_layer = nn.Linear(final_layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        if self.use_gmf:
            nn.init.xavier_normal_(self.user_gmf_embedding.weight)
            nn.init.xavier_normal_(self.item_gmf_embedding.weight)
        
        if self.use_mlp:
            nn.init.xavier_normal_(self.user_mlp_embedding.weight)
            nn.init.xavier_normal_(self.item_mlp_embedding.weight)
            
            for layer in self.mlp_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    
        nn.init.xavier_normal_(self.final_layer.weight)
    
    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NCF model.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            
        Returns:
            Predicted ratings/scores
        """
        outputs = []
        
        # GMF component
        if self.use_gmf:
            user_gmf_embedding = self.user_gmf_embedding(user_indices)
            item_gmf_embedding = self.item_gmf_embedding(item_indices)
            gmf_output = user_gmf_embedding * item_gmf_embedding
            outputs.append(gmf_output)
        
        # MLP component
        if self.use_mlp:
            user_mlp_embedding = self.user_mlp_embedding(user_indices)
            item_mlp_embedding = self.item_mlp_embedding(item_indices)
            mlp_input = torch.cat([user_mlp_embedding, item_mlp_embedding], dim=-1)
            
            for layer in self.mlp_layers:
                mlp_input = layer(mlp_input)
            
            outputs.append(mlp_input)
        
        # Concatenate outputs from GMF and MLP
        if len(outputs) == 1:
            concat_output = outputs[0]
        else:
            concat_output = torch.cat(outputs, dim=-1)
        
        # Final prediction
        prediction = self.final_layer(concat_output)
        prediction = self.sigmoid(prediction)
        
        return prediction.view(-1)


class NCF(BaseRecommender):
    """
    Neural Collaborative Filtering recommender.
    
    This implements the NCF model from the paper:
    "Neural Collaborative Filtering" by He et al. (2017)
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        layers: List[int] = [64, 32, 16, 8],
        dropout: float = 0.2,
        use_gmf: bool = True,
        use_mlp: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        num_epochs: int = 20,
        num_negative_samples: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the NCF recommender.
        
        Args:
            embedding_dim: Size of embedding vectors
            layers: List of layer sizes for MLP component
            dropout: Dropout rate
            use_gmf: Whether to use Generalized Matrix Factorization component
            use_mlp: Whether to use Multi-Layer Perceptron component
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            num_negative_samples: Number of negative samples per positive sample
            device: Device to use for training ('cpu' or 'cuda')
        """
        self.embedding_dim = embedding_dim
        self.layers = layers
        self.dropout = dropout
        self.use_gmf = use_gmf
        self.use_mlp = use_mlp
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_negative_samples = num_negative_samples
        self.device = device
        
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.user_items = {}
        
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Create mappings from original IDs to internal indices.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_item_ids)}
        
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
    
    def _build_user_items_dict(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Build a dictionary of items interacted with by each user.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        self.user_items = {}
        
        for user_id, item_id in zip(user_ids, item_ids):
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
            
            if user_idx not in self.user_items:
                self.user_items[user_idx] = set()
            
            self.user_items[user_idx].add(item_idx)
    
    def _sample_negative_items(self, user_idx: int, n_items: int, n_samples: int) -> List[int]:
        """
        Sample negative items for a user.
        
        Args:
            user_idx: User index
            n_items: Total number of items
            n_samples: Number of negative samples to generate
            
        Returns:
            List of negative item indices
        """
        positive_items = self.user_items.get(user_idx, set())
        negative_items = []
        
        while len(negative_items) < n_samples:
            item_idx = np.random.randint(0, n_items)
            if item_idx not in positive_items and item_idx not in negative_items:
                negative_items.append(item_idx)
        
        return negative_items
    
    def _generate_training_data(self, user_ids: List[int], item_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training data with negative sampling.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            
        Returns:
            Tuple of (user_indices, item_indices, labels)
        """
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Convert to internal indices
        user_indices = [self.user_id_map[user_id] for user_id in user_ids]
        item_indices = [self.item_id_map[item_id] for item_id in item_ids]
        
        # Create positive samples
        train_user_indices = []
        train_item_indices = []
        train_labels = []
        
        for user_idx, item_idx in zip(user_indices, item_indices):
            # Add positive sample
            train_user_indices.append(user_idx)
            train_item_indices.append(item_idx)
            train_labels.append(1.0)
            
            # Add negative samples
            neg_items = self._sample_negative_items(user_idx, n_items, self.num_negative_samples)
            for neg_item_idx in neg_items:
                train_user_indices.append(user_idx)
                train_item_indices.append(neg_item_idx)
                train_labels.append(0.0)
        
        return (
            torch.LongTensor(train_user_indices).to(self.device),
            torch.LongTensor(train_item_indices).to(self.device),
            torch.FloatTensor(train_labels).to(self.device)
        )
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: Optional[List[float]] = None) -> None:
        """
        Train the NCF model.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs
            ratings: List of ratings (optional, not used in binary case)
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        self._build_user_items_dict(user_ids, item_ids)
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Initialize model
        self.model = NCFModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=self.embedding_dim,
            layers=self.layers,
            dropout=self.dropout,
            use_gmf=self.use_gmf,
            use_mlp=self.use_mlp
        ).to(self.device)
        
        # Generate training data
        user_indices, item_indices, labels = self._generate_training_data(user_ids, item_ids)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        n_samples = len(user_indices)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            user_indices_shuffled = user_indices[indices]
            item_indices_shuffled = item_indices[indices]
            labels_shuffled = labels[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, n_samples)
                
                batch_user_indices = user_indices_shuffled[start_idx:end_idx]
                batch_item_indices = item_indices_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.model(batch_user_indices, batch_item_indices)
                loss = criterion(predictions, batch_labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
    
    def fit_from_sparse_matrix(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the NCF model from a sparse interaction matrix.
        
        Args:
            interaction_matrix: Sparse interaction matrix
            user_ids: List of user IDs
            item_ids: List of item IDs
        """
        # Extract user-item pairs from the interaction matrix
        users, items = interaction_matrix.nonzero()
        
        # Map to original IDs
        user_ids_from_matrix = [user_ids[u] for u in users]
        item_ids_from_matrix = [item_ids[i] for i in items]
        
        # Train the model
        self.fit(user_ids_from_matrix, item_ids_from_matrix)
    
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
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_id_map:
            # If user is not in the training data, return empty list or random items
            return []
        
        user_idx = self.user_id_map[user_id]
        n_items = len(self.item_id_map)
        
        # Get items the user has already interacted with
        seen_items = self.user_items.get(user_idx, set()) if exclude_seen else set()
        
        # Generate predictions for all items
        user_tensor = torch.LongTensor([user_idx] * n_items).to(self.device)
        item_tensor = torch.LongTensor(range(n_items)).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Filter out seen items and get top-N
        candidate_items = []
        for item_idx in range(n_items):
            if item_idx not in seen_items:
                candidate_items.append((item_idx, predictions[item_idx]))
        
        # Sort by predicted score and get top-N
        recommended_item_indices = [item_idx for item_idx, _ in sorted(candidate_items, key=lambda x: x[1], reverse=True)[:top_n]]
        
        # Convert back to original item IDs
        recommended_items = [self.reverse_item_map[item_idx] for item_idx in recommended_item_indices]
        
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
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'user_items': self.user_items,
            'embedding_dim': self.embedding_dim,
            'layers': self.layers,
            'dropout': self.dropout,
            'use_gmf': self.use_gmf,
            'use_mlp': self.use_mlp,
            'n_users': len(self.user_id_map),
            'n_items': len(self.item_id_map)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NCF':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded NCF model
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            embedding_dim=model_state['embedding_dim'],
            layers=model_state['layers'],
            dropout=model_state['dropout'],
            use_gmf=model_state['use_gmf'],
            use_mlp=model_state['use_mlp']
        )
        
        # Restore mappings and user-item interactions
        instance.user_id_map = model_state['user_id_map']
        instance.item_id_map = model_state['item_id_map']
        instance.reverse_user_map = model_state['reverse_user_map']
        instance.reverse_item_map = model_state['reverse_item_map']
        instance.user_items = model_state['user_items']
        
        # Recreate the model
        n_users = model_state['n_users']
        n_items = model_state['n_items']
        
        instance.model = NCFModel(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=instance.embedding_dim,
            layers=instance.layers,
            dropout=instance.dropout,
            use_gmf=instance.use_gmf,
            use_mlp=instance.use_mlp
        ).to(instance.device)
        
        # Load the model weights
        instance.model.load_state_dict(model_state['model_state_dict'])
        instance.model.eval()
        
        return instance
