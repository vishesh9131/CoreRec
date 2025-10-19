# LightGCN
# IMPLEMENTATION IN PROGRESS
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from typing import List, Optional, Dict, Tuple, Any
from ..base_recommender import BaseRecommender
from scipy.sparse import csr_matrix

class LightGCNBase(BaseRecommender):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    Implementation based on the paper:
    "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
    by Xiangnan He et al.
    
    This is a simplified GCN model specifically designed for recommendation systems,
    removing unnecessary components like feature transformation and nonlinear activation.
    
    Parameters:
    -----------
    embedding_dim : int
        Dimension of embeddings
    n_layers : int
        Number of graph convolution layers
    learning_rate : float
        Learning rate for optimizer
    regularization : float
        L2 regularization weight
    batch_size : int
        Size of training batches
    epochs : int
        Number of training epochs
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        embedding_dim: int = 64,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        regularization: float = 1e-4,
        batch_size: int = 1024,
        epochs: int = 100,
        seed: Optional[int] = None
    ):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.model = None
        self.user_embeddings = None
        self.item_embeddings = None
        self.user_final_embeddings = None
        self.item_final_embeddings = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
    
    def _build_model(self, n_users: int, n_items: int) -> tf.keras.Model:
        """Build the LightGCN model architecture"""
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
        
        # Initialize user and item embeddings
        initializer = tf.keras.initializers.GlorotUniform()
        self.user_embeddings = tf.Variable(
            initializer([n_users, self.embedding_dim]),
            name='user_embeddings',
            dtype=tf.float32
        )
        self.item_embeddings = tf.Variable(
            initializer([n_items, self.embedding_dim]),
            name='item_embeddings',
            dtype=tf.float32
        )
        
        # Define model inputs
        user_indices = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='user_indices')
        pos_item_indices = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='pos_item_indices')
        neg_item_indices = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='neg_item_indices')
        
        # Create normalized adjacency matrix for graph convolution
        norm_adj = self._create_adj_matrix(n_users, n_items)
        
        # Perform graph convolution to get final embeddings
        self.user_final_embeddings, self.item_final_embeddings = self._light_gcn(norm_adj)
        
        # Look up embeddings for the batch
        user_emb = tf.nn.embedding_lookup(self.user_final_embeddings, user_indices)
        pos_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, pos_item_indices)
        neg_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, neg_item_indices)
        
        # Calculate BPR loss
        pos_scores = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), axis=1)
        
        # Create model
        model = tf.keras.Model(
            inputs=[user_indices, pos_item_indices, neg_item_indices],
            outputs=[pos_scores, neg_scores]
        )
        
        return model
    
    def _create_adj_matrix(self, n_users: int, n_items: int) -> tf.sparse.SparseTensor:
        """Create normalized adjacency matrix for graph convolution"""
        # Create adjacency matrix from interaction matrix
        # Shape: [n_users+n_items, n_users+n_items]
        adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        
        # User-item interactions
        for user_idx in range(n_users):
            items = self.user_item_matrix[user_idx].indices
            for item_idx in items:
                # Add edges between users and items
                adj_mat[user_idx, n_users + item_idx] = 1.0
                adj_mat[n_users + item_idx, user_idx] = 1.0
        
        # Convert to CSR format for efficient operations
        adj_mat = adj_mat.tocsr()
        
        # Calculate degree matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        # Calculate normalized adjacency matrix: D^-0.5 * A * D^-0.5
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        # Convert to TensorFlow sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col)).transpose()
        values = norm_adj.data
        shape = norm_adj.shape
        
        return tf.sparse.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=shape
        )
    
    def _light_gcn(self, norm_adj: tf.sparse.SparseTensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform LightGCN propagation to get final embeddings
        
        LightGCN simplifies GCN by:
        1. Removing feature transformation matrices
        2. Removing nonlinear activation functions
        3. Using layer combination for final embeddings
        """
        # Concatenate user and item embeddings
        all_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        
        # Storage for embeddings at each layer
        embeddings_list = [all_embeddings]
        
        # Graph convolution for n_layers
        for layer in range(self.n_layers):
            # Simple message passing: e^{l+1} = (D^{-0.5} * A * D^{-0.5}) * e^{l}
            all_embeddings = tf.sparse.sparse_dense_matmul(norm_adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Layer combination: final embeddings are the sum of embeddings at each layer
        all_embeddings = tf.add_n(embeddings_list) / (self.n_layers + 1)
        
        # Split user and item embeddings
        user_embeddings, item_embeddings = tf.split(
            all_embeddings, [self.user_embeddings.shape[0], self.item_embeddings.shape[0]], 0
        )
        
        return user_embeddings, item_embeddings
    
    def _create_bpr_loss(self, users, pos_items, neg_items):
        """Create BPR loss for LightGCN"""
        # Get embeddings
        user_emb = tf.nn.embedding_lookup(self.user_final_embeddings, users)
        pos_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, pos_items)
        neg_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, neg_items)
        
        # Calculate scores
        pos_scores = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), axis=1)
        
        # BPR loss
        loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        # L2 regularization
        regularizer = tf.nn.l2_loss(user_emb) + tf.nn.l2_loss(pos_item_emb) + tf.nn.l2_loss(neg_item_emb)
        reg_loss = self.regularization * regularizer
        
        return loss + reg_loss
    
    def _sample_negative_items(self, users: np.ndarray, n_users: int, n_items: int) -> np.ndarray:
        """Sample negative items for each user in the batch"""
        neg_items = np.random.randint(0, n_items, size=len(users))
        
        for i, user in enumerate(users):
            # Ensure negative item is not interacted with by the user
            user_items = self.user_item_matrix[user].indices
            while neg_items[i] in user_items:
                neg_items[i] = np.random.randint(0, n_items)
                
        return neg_items
    
    def _get_training_pairs(self, n_users: int, n_items: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training pairs (user, positive_item)"""
        user_indices = []
        pos_item_indices = []
        
        for user_idx in range(n_users):
            items = self.user_item_matrix[user_idx].indices
            if len(items) == 0:
                continue
                
            # Add all positive interactions
            for item_idx in items:
                user_indices.append(user_idx)
                pos_item_indices.append(item_idx)
        
        return np.array(user_indices), np.array(pos_item_indices)
    
    def fit(self, interaction_matrix: csr_matrix, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Train the LightGCN model on the given interaction data.
        
        Parameters:
        -----------
        interaction_matrix : csr_matrix
            User-item interaction matrix where non-zero entries indicate interactions
        user_ids : List[int]
            List of user IDs corresponding to rows in the interaction matrix
        item_ids : List[int]
            List of item IDs corresponding to columns in the interaction matrix
        """
        # Create mappings
        self._create_mappings(user_ids, item_ids)
        
        # Store interaction matrix for later use
        self.user_item_matrix = interaction_matrix
        
        # Get dimensions
        n_users, n_items = interaction_matrix.shape
        
        # Build the model
        self.model = self._build_model(n_users, n_items)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Get training pairs
        user_indices, pos_item_indices = self._get_training_pairs(n_users, n_items)
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle training data
            indices = np.random.permutation(len(user_indices))
            user_indices_shuffled = user_indices[indices]
            pos_item_indices_shuffled = pos_item_indices[indices]
            
            # Mini-batch training
            total_loss = 0
            n_batches = int(np.ceil(len(user_indices) / self.batch_size))
            
            for batch in range(n_batches):
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, len(user_indices))
                
                batch_users = user_indices_shuffled[start:end]
                batch_pos_items = pos_item_indices_shuffled[start:end]
                
                # Sample negative items
                batch_neg_items = self._sample_negative_items(batch_users, n_users, n_items)
                
                # Gradient tape for automatic differentiation
                with tf.GradientTape() as tape:
                    # Forward pass
                    user_emb = tf.nn.embedding_lookup(self.user_final_embeddings, batch_users)
                    pos_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, batch_pos_items)
                    neg_item_emb = tf.nn.embedding_lookup(self.item_final_embeddings, batch_neg_items)
                    
                    # Calculate scores
                    pos_scores = tf.reduce_sum(tf.multiply(user_emb, pos_item_emb), axis=1)
                    neg_scores = tf.reduce_sum(tf.multiply(user_emb, neg_item_emb), axis=1)
                    
                    # BPR loss
                    loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
                    
                    # L2 regularization
                    regularizer = tf.nn.l2_loss(user_emb) + tf.nn.l2_loss(pos_item_emb) + tf.nn.l2_loss(neg_item_emb)
                    reg_loss = self.regularization * regularizer
                    
                    # Total loss
                    total_batch_loss = loss + reg_loss
                
                # Compute gradients
                grads = tape.gradient(total_batch_loss, [self.user_embeddings, self.item_embeddings])
                
                # Apply gradients
                optimizer.apply_gradients(zip(grads, [self.user_embeddings, self.item_embeddings]))
                
                # Update total loss
                total_loss += total_batch_loss.numpy()
            
            # Print progress
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # Update final embeddings after each epoch
            norm_adj = self._create_adj_matrix(n_users, n_items)
            self.user_final_embeddings, self.item_final_embeddings = self._light_gcn(norm_adj)
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Generate top-N recommendations for a specific user.
        
        Parameters:
        -----------
        user_id : int
            ID of the user to generate recommendations for
        top_n : int
            Number of recommendations to generate
        exclude_seen : bool
            Whether to exclude items the user has already interacted with
            
        Returns:
        --------
        List[int] : List of recommended item IDs
        """
        if self.user_final_embeddings is None or self.item_final_embeddings is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Map user_id to internal index
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.user_map[user_id]
        
        # Get user embedding
        user_emb = self.user_final_embeddings[user_idx]
        
        # Calculate scores for all items
        scores = tf.matmul(
            tf.expand_dims(user_emb, 0),
            self.item_final_embeddings,
            transpose_b=True
        ).numpy().flatten()
        
        # If requested, exclude items the user has already interacted with
        if exclude_seen:
            seen_items = self.user_item_matrix[user_idx].indices
            scores[seen_items] = float('-inf')
        
        # Get top-n item indices
        top_item_indices = np.argsort(-scores)[:top_n]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.user_final_embeddings is None or self.item_final_embeddings is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Save embeddings and model data
        model_data = {
            'user_embeddings': self.user_embeddings.numpy(),
            'item_embeddings': self.item_embeddings.numpy(),
            'user_final_embeddings': self.user_final_embeddings.numpy(),
            'item_final_embeddings': self.item_final_embeddings.numpy(),
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'params': {
                'embedding_dim': self.embedding_dim,
                'n_layers': self.n_layers,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'seed': self.seed
            }
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightGCNBase':
        """Load a model from a file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            embedding_dim=model_data['params']['embedding_dim'],
            n_layers=model_data['params']['n_layers'],
            learning_rate=model_data['params']['learning_rate'],
            regularization=model_data['params']['regularization'],
            batch_size=model_data['params']['batch_size'],
            epochs=model_data['params']['epochs'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        
        # Restore embeddings
        instance.user_embeddings = tf.Variable(model_data['user_embeddings'], name='user_embeddings')
        instance.item_embeddings = tf.Variable(model_data['item_embeddings'], name='item_embeddings')
        instance.user_final_embeddings = tf.Variable(model_data['user_final_embeddings'], name='user_final_embeddings')
        instance.item_final_embeddings = tf.Variable(model_data['item_final_embeddings'], name='item_final_embeddings')
        
        return instance