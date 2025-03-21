import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class Caser(BaseRecommender):
    """
    Convolutional Sequence Embedding Recommendation (Caser) model.
    
    This model uses horizontal and vertical convolutional filters to capture sequential patterns
    in user behavior sequences for next-item recommendation.
    
    Parameters:
    -----------
    embedding_dim : int
        Dimension of item embeddings
    n_h : int
        Number of horizontal convolutional filters
    n_v : int
        Number of vertical convolutional filters
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Training batch size
    epochs : int
        Number of training epochs
    max_seq_length : int
        Maximum sequence length to consider
    dropout_rate : float
        Dropout probability
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        embedding_dim: int = 64,
        n_h: int = 16,
        n_v: int = 4,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        max_seq_length: int = 50,
        dropout_rate: float = 0.2,
        seed: Optional[int] = None
    ):
        self.embedding_dim = embedding_dim
        self.n_h = n_h
        self.n_v = n_v
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.model = None
        self.prediction_model = None
        self.user_sequences = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
    
    def _build_model(self, n_items: int) -> tf.keras.Model:
        """Build the Caser model architecture"""
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
        
        # Input layer
        sequence_input = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32', name='sequence_input')
        
        # Embedding layer
        item_embedding = tf.keras.layers.Embedding(
            input_dim=n_items,
            output_dim=self.embedding_dim,
            embeddings_initializer='glorot_normal',
            name='item_embedding'
        )(sequence_input)
        
        # Horizontal convolutional layer
        conv_h = tf.keras.layers.Conv2D(
            filters=self.n_h,
            kernel_size=(1, self.embedding_dim),
            activation='relu',
            name='conv_h'
        )(tf.expand_dims(item_embedding, -1))
        
        # Vertical convolutional layer
        conv_v = tf.keras.layers.Conv2D(
            filters=self.n_v,
            kernel_size=(self.max_seq_length, 1),
            activation='relu',
            name='conv_v'
        )(tf.expand_dims(item_embedding, -1))
        
        # Flatten and concatenate
        flatten_h = tf.keras.layers.Flatten()(conv_h)
        flatten_v = tf.keras.layers.Flatten()(conv_v)
        concat = tf.keras.layers.Concatenate()([flatten_h, flatten_v])
        
        # Dropout
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(concat)
        
        # Dense layer
        dense = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(dropout)
        
        # Output layer
        output = tf.keras.layers.Dense(n_items, activation='softmax')(dense)
        
        # Build model
        model = tf.keras.Model(sequence_input, output, name='caser')
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='sparse_categorical_crossentropy')
        
        return model
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float]) -> None:
        """
        Train the Caser model using the provided data.
        
        Parameters:
        -----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        ratings : List[float]
            List of ratings
        """
        # Create mappings
        unique_user_ids = list(set(user_ids))
        unique_item_ids = list(set(item_ids))
        self._create_mappings(unique_user_ids, unique_item_ids)
        
        # Map IDs to indices
        user_indices = [self.user_map[user_id] for user_id in user_ids]
        item_indices = [self.item_map[item_id] for item_id in item_ids]
        
        # Create user-item matrix
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items))
        
        # Build Caser model
        self.model = self._build_model(n_items)
        
        # Prepare training data
        self.user_sequences = {user_idx: [] for user_idx in range(n_users)}
        for user_idx, item_idx in zip(user_indices, item_indices):
            self.user_sequences[user_idx].append(item_idx)
        
        X, y = [], []
        for user_idx, sequence in self.user_sequences.items():
            if len(sequence) < self.max_seq_length + 1:
                continue
            for i in range(len(sequence) - self.max_seq_length):
                X.append(sequence[i:i+self.max_seq_length])
                y.append(sequence[i+self.max_seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        
        # Create prediction model
        self.prediction_model = self.model
    
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
        if self.prediction_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Map user_id to internal index
        if user_id not in self.user_map:
            raise ValueError(f"User ID {user_id} not found in training data")
            
        user_idx = self.user_map[user_id]
        
        # Get user sequence
        if user_idx not in self.user_sequences:
            raise ValueError(f"User ID {user_id} has no sequence")
            
        sequence = self.user_sequences[user_idx]
        
        # Pad sequence
        if len(sequence) < self.max_seq_length:
            padded_sequence = [0] * (self.max_seq_length - len(sequence)) + sequence
        else:
            padded_sequence = sequence[-self.max_seq_length:]
        
        # Reshape for prediction
        padded_sequence = np.array([padded_sequence])
        
        # Get predictions
        predictions = self.prediction_model.predict(padded_sequence, verbose=0)[0]
        
        # If requested, exclude items the user has already interacted with
        if exclude_seen:
            seen_items = self.user_item_matrix[user_idx].indices
            predictions[seen_items] = -np.inf
        
        # Get top-n item indices
        top_item_indices = np.argsort(-predictions)[:top_n]
        
        # Map indices back to original item IDs
        top_items = [self.reverse_item_map[idx] for idx in top_item_indices]
        
        return top_items
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
            
        # Save Keras model
        self.model.save(f"{filepath}_keras_model")
        
        # Save additional model data
        model_data = {
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
            'user_sequences': self.user_sequences,
            'params': {
                'embedding_dim': self.embedding_dim,
                'n_h': self.n_h,
                'n_v': self.n_v,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'max_seq_length': self.max_seq_length,
                'dropout_rate': self.dropout_rate,
                'seed': self.seed
            }
        }
        np.save(f"{filepath}_model_data", model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'Caser':
        """Load a model from a file"""
        # Load model data
        model_data = np.load(f"{filepath}_model_data.npy", allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            embedding_dim=model_data['params']['embedding_dim'],
            n_h=model_data['params']['n_h'],
            n_v=model_data['params']['n_v'],
            learning_rate=model_data['params']['learning_rate'],
            batch_size=model_data['params']['batch_size'],
            epochs=model_data['params']['epochs'],
            max_seq_length=model_data['params']['max_seq_length'],
            dropout_rate=model_data['params']['dropout_rate'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        instance.user_sequences = model_data['user_sequences']
        
        # Load Keras model
        instance.model = tf.keras.models.load_model(f"{filepath}_keras_model")
        instance.prediction_model = instance.model
        
        return instance 