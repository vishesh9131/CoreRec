import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from corerec.base_recommender import BaseCorerec
import logging

logger = logging.getLogger(__name__)

class MIND(BaseCorerec):
    """
    Multi-Interest Network with Dynamic routing for Recommendation (MIND)
    
    Represents a user with multiple interest vectors to better capture diverse user preferences.
    Uses a dynamic routing mechanism inspired by capsule networks to extract user interests.
    
    Reference:
    Li et al. "Multi-Interest Network with Dynamic Routing for Recommendation at Tmall" (CIKM 2019)
    """
    
    def __init__(
        self,
        name: str = "MIND",
        embedding_dim: int = 64,
        num_interests: int = 4,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2,
        routing_iterations: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        max_seq_length: int = 50,
        trainable: bool = True,
        verbose: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.routing_iterations = routing_iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.user_history = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.model = None
        
    def _build_model(self, num_items: int):
        class MultiInterestExtractor(nn.Module):
            def __init__(self, embedding_dim, num_interests, routing_iterations):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.num_interests = num_interests
                self.routing_iterations = routing_iterations
                
                # Transformation matrix for dynamic routing
                self.routing_weights = nn.Parameter(
                    torch.randn(num_interests, embedding_dim, embedding_dim) * 0.01
                )
                
            def forward(self, item_embeddings, mask=None):
                batch_size, seq_length, embedding_dim = item_embeddings.size()
                
                # Apply mask if provided
                if mask is not None:
                    item_embeddings = item_embeddings * mask.unsqueeze(-1)
                
                # Initialize routing logits
                routing_logits = torch.zeros(batch_size, seq_length, self.num_interests).to(item_embeddings.device)
                
                # Dynamic routing
                for _ in range(self.routing_iterations):
                    # Calculate routing weights
                    routing_weights = F.softmax(routing_logits, dim=-1)
                    
                    # Initialize capsules
                    capsules = torch.zeros(batch_size, self.num_interests, self.embedding_dim).to(item_embeddings.device)
                    
                    # Calculate weighted sum for each interest
                    for i in range(self.num_interests):
                        weights = routing_weights[:, :, i].unsqueeze(-1)
                        capsules[:, i] = torch.sum(weights * item_embeddings, dim=1)
                    
                    # Update routing logits
                    for i in range(self.num_interests):
                        # Transform item embeddings with routing weights
                        transformed = torch.matmul(item_embeddings, self.routing_weights[i])
                        # Calculate similarity between transformed embeddings and capsule
                        similarity = torch.sum(
                            transformed * capsules[:, i].unsqueeze(1), 
                            dim=-1
                        )
                        routing_logits[:, :, i] = routing_logits[:, :, i] + similarity
                
                # Normalize capsules
                capsule_norms = torch.norm(capsules, dim=-1, keepdim=True)
                capsules = capsules * (F.sigmoid(capsule_norms) / capsule_norms)
                
                return capsules
        
        class MINDModel(nn.Module):
            def __init__(self, num_items, embedding_dim, num_interests, hidden_dims, dropout, routing_iterations):
                super().__init__()
                self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
                self.interest_extractor = MultiInterestExtractor(embedding_dim, num_interests, routing_iterations)
                
                # Label predictor
                layers = []
                input_dim = embedding_dim
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    input_dim = hidden_dim
                layers.append(nn.Linear(input_dim, embedding_dim))
                
                self.label_predictor = nn.Sequential(*layers)
                
            def forward(self, item_sequences, target_items=None, training=True):
                # Get item embeddings
                item_embeddings = self.item_embedding(item_sequences)
                
                # Create mask for padding
                mask = (item_sequences != 0).float()
                
                # Extract multiple interests
                user_interests = self.interest_extractor(item_embeddings, mask)
                
                if training and target_items is not None:
                    # Get target item embeddings
                    target_embeddings = self.item_embedding(target_items)
                    
                    # Compute similarity between each interest and target
                    similarities = torch.bmm(
                        user_interests, 
                        target_embeddings.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Get the most related interest for each target
                    max_similarity, max_idx = torch.max(similarities, dim=1)
                    
                    # Get the corresponding user interests
                    batch_indices = torch.arange(user_interests.size(0)).to(user_interests.device)
                    selected_interests = user_interests[batch_indices, max_idx]
                    
                    # Apply label predictor
                    predicted_embeddings = self.label_predictor(selected_interests)
                    
                    # Compute scores
                    scores = torch.sum(predicted_embeddings * target_embeddings, dim=-1)
                    
                    return torch.sigmoid(scores)
                else:
                    # For recommendation, return all interests
                    return user_interests
        
        return MINDModel(num_items, self.embedding_dim, self.num_interests, self.hidden_dims, 
                         self.dropout, self.routing_iterations).to(self.device)
    
    def fit(self, user_ids: List[int], item_ids: List[int], timestamps: List[int]) -> None:
        # Create item mapping
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        unique_items = sorted(set(item_ids))
        self.item_map = {item: idx + 1 for idx, item in enumerate(unique_items)}  # +1 for padding
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        self.reverse_item_map[0] = 0  # Padding item
        
        # Create user histories
        user_item_timestamps = sorted(zip(user_ids, item_ids, timestamps), key=lambda x: (x[0], x[2]))
        current_user = None
        current_history = []
        
        for user, item, _ in user_item_timestamps:
            if user != current_user:
                if current_user is not None:
                    self.user_history[current_user] = current_history
                current_user = user
                current_history = []
            
            current_history.append(self.item_map[item])
        
        # Add the last user's history
        if current_user is not None:
            self.user_history[current_user] = current_history
        
        # Build model
        self.model = self._build_model(len(unique_items))
        
        # Create training sequences and targets
        train_sequences = []
        train_targets = []
        
        for user, history in self.user_history.items():
            for i in range(1, len(history)):
                # Use items up to position i-1 to predict item at position i
                seq = history[:i]
                
                # Truncate or pad sequence
                if len(seq) > self.max_seq_length:
                    seq = seq[-self.max_seq_length:]
                else:
                    seq = [0] * (self.max_seq_length - len(seq)) + seq
                
                train_sequences.append(seq)
                train_targets.append(history[i])
        
        # Convert to tensors
        train_sequences = torch.LongTensor(train_sequences).to(self.device)
        train_targets = torch.LongTensor(train_targets).to(self.device)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        
        # Check if we have training data
        if len(train_sequences) == 0:
            if self.verbose:
                logger.warning("Warning: No training sequences found. Skipping training.")
            return
        
        n_batches = len(train_sequences) // self.batch_size + (1 if len(train_sequences) % self.batch_size != 0 else 0)
        
        # Ensure at least 1 batch
        if n_batches == 0:
            n_batches = 1
        
        for epoch in range(self.epochs):
            total_loss = 0
            
            # Shuffle data
            indices = torch.randperm(len(train_sequences))
            shuffled_sequences = train_sequences[indices]
            shuffled_targets = train_targets[indices]
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_sequences))
                
                # Skip if batch is empty
                if start_idx >= end_idx:
                    continue
                
                batch_sequences = shuffled_sequences[start_idx:end_idx]
                batch_targets = shuffled_targets[start_idx:end_idx]
                
                # Create positive and negative examples
                pos_targets = batch_targets
                neg_targets = torch.randint(1, len(self.item_map) + 1, size=pos_targets.size()).to(self.device)
                
                # Forward pass for positive examples
                pos_scores = self.model(batch_sequences, pos_targets)
                pos_labels = torch.ones_like(pos_scores)
                
                # Forward pass for negative examples
                neg_scores = self.model(batch_sequences, neg_targets)
                neg_labels = torch.zeros_like(neg_scores)
                
                # Combine positive and negative examples
                scores = torch.cat([pos_scores, neg_scores])
                labels = torch.cat([pos_labels, neg_labels])
                
                # Compute loss
                loss = criterion(scores, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Safe division to avoid ZeroDivisionError
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[int]:
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_k if 'top_k' in locals() else 10)
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if user_id not in self.user_history:
            return []
        
        # Get user history
        history = self.user_history[user_id]
        
        # Truncate or pad sequence
        if len(history) > self.max_seq_length:
            history = history[-self.max_seq_length:]
        else:
            history = [0] * (self.max_seq_length - len(history)) + history
        
        # Convert to tensor
        history_tensor = torch.LongTensor([history]).to(self.device)
        
        # Get user interests
        self.model.eval()
        with torch.no_grad():
            user_interests = self.model(history_tensor, training=False)
        
        # Get all item embeddings
        item_embeddings = self.model.item_embedding.weight[1:]  # Skip padding
        
        # Compute scores for each interest
        scores = torch.zeros(len(self.item_map))
        
        for interest in user_interests[0]:
            # Apply label predictor
            predicted_embedding = self.model.label_predictor(interest)
            
            # Compute similarity with all items
            similarity = torch.matmul(predicted_embedding, item_embeddings.T)
            
            # Update scores (take maximum across interests)
            scores = torch.maximum(scores, similarity.cpu())
        
        # Exclude seen items if requested
        if exclude_seen:
            for item_idx in history:
                if item_idx > 0:  # Skip padding
                    scores[item_idx - 1] = float('-inf')  # -1 to account for padding index
        
        # Get top-N item indices
        top_indices = torch.argsort(scores, descending=True)[:top_n].numpy()
        
        # Convert indices back to original item IDs (add 1 to account for padding, then map)
        recommended_items = [self.reverse_item_map[idx + 1] for idx in top_indices]
        
        return recommended_items