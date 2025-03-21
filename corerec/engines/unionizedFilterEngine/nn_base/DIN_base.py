import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from typing import List, Optional, Dict, Any, Tuple
from ..base_recommender import BaseRecommender

class DINBase(BaseRecommender):
    """
    Deep Interest Network (DIN) for recommendation.
    
    Based on the paper:
    "Deep Interest Network for Click-Through Rate Prediction" by Zhou et al.
    
    This model uses an attention mechanism to adaptively learn user interests
    from historical behaviors with respect to target items.
    
    Parameters:
    -----------
    embedding_dim : int
        Dimension of item embeddings
    attention_units : List[int]
        Sizes of attention network layers
    fc_units : List[int]
        Sizes of fully connected layers
    dropout_rate : float
        Dropout probability
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Training batch size
    epochs : int
        Number of training epochs
    max_history_length : int
        Maximum number of historical interactions to consider
    seed : Optional[int]
        Random seed for reproducibility
    """
    def __init__(
        self,
        embedding_dim: int = 64,
        attention_units: List[int] = [80, 40],
        fc_units: List[int] = [200, 80],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        max_history_length: int = 50,
        seed: Optional[int] = None
    ):
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units
        self.fc_units = fc_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_history_length = max_history_length
        self.seed = seed
        
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.model = None
        self.prediction_model = None
        self.user_histories = None
    
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]) -> None:
        """Create mappings between original IDs and matrix indices"""
        self.user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_map.items()}
    
    def _attention_layer(self, queries, keys, name):
        """Build attention layer"""
        # Concatenate queries and keys
        attention_input = tf.keras.layers.concatenate([queries, keys, queries - keys, queries * keys])
        
        # Attention network
        for i, units in enumerate(self.attention_units):
            if i == 0:
                attention_output = tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    name=f'{name}_attention_layer_{i}'
                )(attention_input)
            else:
                attention_output = tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    name=f'{name}_attention_layer_{i}'
                )(attention_output)
        
        # Output layer (scalar attention weight)
        attention_output = tf.keras.layers.Dense(1, activation='sigmoid', name=f'{name}_attention_weight')(attention_output)
        
        return attention_output
    
    def _build_model(self, n_items: int) -> tf.keras.Model:
        """Build the DIN model architecture"""
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
        
        # Input layers
        item_input = tf.keras.layers.Input(shape=(1,), dtype='int32', name='item_input')
        history_input = tf.keras.layers.Input(shape=(self.max_history_length,), dtype='int32', name='history_input')
        history_length = tf.keras.layers.Input(shape=(1,), dtype='int32', name='history_length')
        
        # Add 1 to n_items for padding (item_id=0)
        # Item embedding layer
        item_embedding_layer = tf.keras.layers.Embedding(
            input_dim=n_items + 1,
            output_dim=self.embedding_dim,
            mask_zero=False,  # We'll handle masking manually
            name='item_embedding'
        )
        
        # Embed target item
        item_embedding = item_embedding_layer(item_input)
        item_embedding = tf.keras.layers.Flatten()(item_embedding)
        
        # Embed history items
        history_embedding = item_embedding_layer(history_input)
        
        # Mask for history items (0 is padding)
        history_mask = tf.cast(tf.not_equal(history_input, 0), tf.float32)
        history_mask = tf.expand_dims(history_mask, axis=-1)
        
        # Apply attention mechanism
        # Repeat target item embedding for each history item
        target_embedding = tf.keras.layers.RepeatVector(self.max_history_length)(item_embedding)
        
        # Calculate attention weights
        attention_weights = self._attention_layer(target_embedding, history_embedding, 'din')
        
        # Apply mask to attention weights
        attention_weights = attention_weights * history_mask
        
        # Normalize attention weights
        attention_sum = tf.reduce_sum(attention_weights, axis=1, keepdims=True) + 1e-10
        attention_weights = attention_weights / attention_sum
        
        # Apply attention weights to history embeddings
        weighted_history = history_embedding * attention_weights
        
        # Sum weighted history embeddings
        user_interest = tf.reduce_sum(weighted_history, axis=1)
        
        # Concatenate user interest and target item embedding
        concat_output = tf.keras.layers.concatenate([user_interest, item_embedding])
        
        # Fully connected layers
        for i, units in enumerate(self.fc_units):
            concat_output = tf.keras.layers.Dense(
                units, 
                activation='relu',
                name=f'fc_layer_{i}'
            )(concat_output)
            concat_output = tf.keras.layers.Dropout(self.dropout_rate)(concat_output)
        
        # Output layer
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(concat_output)
        
        # Create model
        model = tf.keras.Model(inputs=[item_input, history_input, history_length], outputs=output)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, user_ids: List[int], item_ids: List[int], ratings: List[float], timestamps: Optional[List[int]] = None) -> None:
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        user_ids : List[int]
            List of user IDs
        item_ids : List[int]
            List of item IDs
        ratings : List[float]
            List of ratings
        timestamps : Optional[List[int]]
            List of timestamps (if available)
        """
        # Create mappings
        unique_user_ids = sorted(set(user_ids))
        unique_item_ids = sorted(set(item_ids))
        self._create_mappings(unique_user_ids, unique_item_ids)
        
        # Map IDs to indices
        user_indices = [self.user_map[user_id] for user_id in user_ids]
        item_indices = [self.item_map[item_id] for item_id in item_ids]
        
        # Create user-item matrix
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        self.user_item_matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(n_users, n_items))
        
        # Create user histories
        self.user_histories = {}
        
        # If timestamps are provided, sort interactions by time
        if timestamps is not None:
            # Create a list of (user_idx, item_idx, timestamp) tuples
            interactions = list(zip(user_indices, item_indices, timestamps))
            
            # Sort by user and timestamp
            interactions.sort(key=lambda x: (x[0], x[1]))
            
            # Build user histories
            for user_idx, item_idx, _ in interactions:
                if user_idx not in self.user_histories:
                    self.user_histories[user_idx] = []
                
                # Add item to user history
                if item_idx + 1 not in self.user_histories[user_idx]:  # +1 to account for padding
                    self.user_histories[user_idx].append(item_idx + 1)  # +1 to account for padding
        else:
            # Without timestamps, just collect all items for each user
            for user_idx, item_idx in zip(user_indices, item_indices):
                if user_idx not in self.user_histories:
                    self.user_histories[user_idx] = []
                
                # Add item to user history
                if item_idx + 1 not in self.user_histories[user_idx]:  # +1 to account for padding
                    self.user_histories[user_idx].append(item_idx + 1)  # +1 to account for padding
        
        # Prepare training data
        X_item = []
        X_history = []
        X_history_length = []
        y = []
        
        # For each positive interaction, create a training sample
        for user_idx, item_idx in zip(user_indices, item_indices):
            # Get user history excluding current item
            history = [h for h in self.user_histories[user_idx] if h != item_idx + 1]
            
            # Skip if no history
            if not history:
                continue
            
            # Truncate or pad history
            history_length = len(history)
            if history_length > self.max_history_length:
                history = history[-self.max_history_length:]
                history_length = self.max_history_length
            else:
                history = [0] * (self.max_history_length - history_length) + history
            
            # Add positive sample
            X_item.append(item_idx + 1)  # +1 to account for padding
            X_history.append(history)
            X_history_length.append(history_length)
            y.append(1)
            
            # Add negative samples (randomly sampled items)
            for _ in range(4):  # 4 negative samples per positive
                neg_item_idx = np.random.randint(n_items)
                while self.user_item_matrix[user_idx, neg_item_idx] > 0:
                    neg_item_idx = np.random.randint(n_items)
                
                X_item.append(neg_item_idx + 1)  # +1 to account for padding
                X_history.append(history)
                X_history_length.append(history_length)
                y.append(0)
        
        # Convert to numpy arrays
        X_item = np.array(X_item).reshape(-1, 1)
        X_history = np.array(X_history)
        X_history_length = np.array(X_history_length).reshape(-1, 1)
        y = np.array(y)
        
        # Build model
        self.model = self._build_model(n_items)
        
        # Train model
        self.model.fit(
            [X_item, X_history, X_history_length], y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        
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
        
        # Get user history
        if user_idx not in self.user_histories:
            raise ValueError(f"User ID {user_id} has no history")
            
        history = self.user_histories[user_idx]
        
        # Truncate or pad history
        history_length = len(history)
        if history_length > self.max_history_length:
            history = history[-self.max_history_length:]
            history_length = self.max_history_length
        else:
            history = [0] * (self.max_history_length - history_length) + history
        
        # Prepare input for all items
        n_items = len(self.item_map)
        X_item = np.arange(1, n_items + 1).reshape(-1, 1)  # +1 to account for padding
        X_history = np.tile(history, (n_items, 1))
        X_history_length = np.full((n_items, 1), min(history_length, self.max_history_length))
        
        # Get predictions
        predictions = self.prediction_model.predict([X_item, X_history, X_history_length], batch_size=100, verbose=0).flatten()
        
        # If requested, exclude items the user has already interacted with
        if exclude_seen:
            seen_items = self.user_item_matrix[user_idx].indices
            for item_idx in seen_items:
                predictions[item_idx] = -np.inf
        
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
            'user_histories': self.user_histories,
            'params': {
                'embedding_dim': self.embedding_dim,
                'attention_units': self.attention_units,
                'fc_units': self.fc_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'max_history_length': self.max_history_length,
                'seed': self.seed
            }
        }
        np.save(f"{filepath}_model_data", model_data, allow_pickle=True)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'DINBase':
        """Load a model from a file"""
        # Load model data
        model_data = np.load(f"{filepath}_model_data.npy", allow_pickle=True).item()
        
        # Create an instance with the saved parameters
        instance = cls(
            embedding_dim=model_data['params']['embedding_dim'],
            attention_units=model_data['params']['attention_units'],
            fc_units=model_data['params']['fc_units'],
            dropout_rate=model_data['params']['dropout_rate'],
            learning_rate=model_data['params']['learning_rate'],
            batch_size=model_data['params']['batch_size'],
            epochs=model_data['params']['epochs'],
            max_history_length=model_data['params']['max_history_length'],
            seed=model_data['params']['seed']
        )
        
        # Restore instance variables
        instance.user_map = model_data['user_map']
        instance.item_map = model_data['item_map']
        instance.reverse_user_map = model_data['reverse_user_map']
        instance.reverse_item_map = model_data['reverse_item_map']
        instance.user_histories = model_data['user_histories']
        
        # Load Keras model
        instance.model = tf.keras.models.load_model(f"{filepath}_keras_model")
        instance.prediction_model = instance.model
        
        return instance 