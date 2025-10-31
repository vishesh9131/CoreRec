import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from corerec.base_recommender import BaseCorerec

class DynamicRoutingCapsule(nn.Module):
    """Dynamic routing mechanism between capsules"""
    def __init__(self, input_dim, num_capsules, capsule_dim, routing_iterations=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iterations = routing_iterations
        self.W = nn.Parameter(torch.randn(num_capsules, input_dim, capsule_dim) * 0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Expand x for transformation: [batch, seq_len, input_dim] -> [batch, seq_len, num_capsules, input_dim]
        x = x.unsqueeze(2).expand(-1, -1, self.num_capsules, -1)
        
        # Prepare W for batched matmul: [num_capsules, input_dim, capsule_dim] -> [1, 1, num_capsules, input_dim, capsule_dim]
        W = self.W.unsqueeze(0).unsqueeze(0)
        
        # Compute u_hat (predicted output vectors): [batch, seq_len, num_capsules, capsule_dim]
        u_hat = torch.matmul(x, W).squeeze()
        
        # Initialize routing logits
        b = torch.zeros(batch_size, seq_len, self.num_capsules, device=x.device)
        
        # Dynamic routing
        for i in range(self.routing_iterations):
            # Calculate coupling coefficients
            c = F.softmax(b, dim=2)
            c = c.unsqueeze(3)
            
            # Weighted sum of predictions
            s = (c * u_hat).sum(dim=1)
            
            # Squash to get output vectors
            v = self.squash(s)
            
            # Update routing logits
            if i < self.routing_iterations - 1:
                # Calculate agreement
                agreement = torch.matmul(u_hat, v.unsqueeze(3)).squeeze()
                b = b + agreement
        
        return v
    
    @staticmethod
    def squash(x, dim=-1):
        """Squashing function for capsule output"""
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

class MINDRecommender(BaseCorerec):
    """
    Multi-Interest Network with Dynamic Routing for Recommendation.
    
    This model represents a user with multiple interest vectors and
    uses a dynamic routing mechanism to extract user's diverse interests
    from their interaction history.
import logging

logger = logging.getLogger(__name__)
    
    Features:
    - Multiple interest representation for users
    - Dynamic routing for interest extraction
    - Supports both item ID-based and content-based features
    - Configurable interest extraction mechanisms
    - Compatible with various item encoders
    
    Reference:
    Li et al. "Multi-Interest Network with Dynamic Routing for 
    Recommendation at Tmall" (CIKM 2019)
    """
    
    def __init__(
        self,
        name: str = "MIND",
        embedding_dim: int = 64,
        num_interests: int = 4,
        hidden_layers: List[int] = [128, 64],
        dropout: float = 0.2,
        routing_iterations: int = 3,
        interest_extractor: str = "dynamic_routing",  # Options: dynamic_routing, self_attention, kmeans
        item_encoder: Optional[nn.Module] = None,  # Custom item encoder if provided
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 20,
        loss_type: str = "bpr",  # Options: bpr, ce, hinge
        max_seq_length: int = 50,
        feature_type: str = "id",  # Options: id, text, hybrid
        optimizer_type: str = "adam",  # Options: adam, sgd, adagrad
        early_stopping_patience: int = 5,
        l2_regularization: float = 0.0001,
        trainable: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        
        # Basic parameters
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.routing_iterations = routing_iterations
        self.interest_extractor = interest_extractor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_type = loss_type
        self.max_seq_length = max_seq_length
        self.feature_type = feature_type
        self.optimizer_type = optimizer_type
        self.early_stopping_patience = early_stopping_patience
        self.l2_regularization = l2_regularization
        self.seed = seed
        self.device = device
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Will be set during fit
        self.num_items = None
        self.user_history = {}
        self.model = None
        self.item_encoder = item_encoder
        self.is_fitted = False
        self.user_embeddings = None
        
    def _build_model(self):
        """Build the MIND model architecture"""
        class MINDModule(nn.Module):
            def __init__(
                self, 
                num_items, 
                embedding_dim, 
                num_interests, 
                hidden_layers,
                dropout,
                routing_iterations,
                interest_extractor,
                feature_type,
                item_encoder=None
            ):
                super().__init__()
                self.num_items = num_items
                self.embedding_dim = embedding_dim
                self.num_interests = num_interests
                
                # Item embedding layer (if using ID-based features)
                if feature_type in ['id', 'hybrid']:
                    self.item_embedding = nn.Embedding(num_items+1, embedding_dim, padding_idx=0)
                
                # Use custom item encoder if provided
                self.item_encoder = item_encoder
                
                # Interest extraction mechanism
                if interest_extractor == 'dynamic_routing':
                    self.interest_extractor = DynamicRoutingCapsule(
                        embedding_dim, num_interests, embedding_dim, routing_iterations
                    )
                elif interest_extractor == 'self_attention':
                    self.interest_extractor = nn.MultiheadAttention(
                        embedding_dim, num_heads=num_interests, dropout=dropout
                    )
                else:
                    raise ValueError(f"Unsupported interest extractor: {interest_extractor}")
                
                # Label predictor (projection layers)
                layers = []
                input_dim = embedding_dim
                for hidden_dim in hidden_layers:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    input_dim = hidden_dim
                layers.append(nn.Linear(input_dim, embedding_dim))
                self.label_predictor = nn.Sequential(*layers)
                
            def forward(self, x, target_items=None, training=True):
                # Get item embeddings
                if self.item_encoder is not None:
                    # Use custom encoder if provided
                    item_emb = self.item_encoder(x)
                else:
                    # Otherwise use standard embedding lookup
                    item_emb = self.item_embedding(x)
                
                # Extract user's interests
                if self.interest_extractor.__class__.__name__ == 'DynamicRoutingCapsule':
                    user_interests = self.interest_extractor(item_emb)
                else:
                    # Self-attention based extraction
                    attn_output, _ = self.interest_extractor(
                        item_emb.transpose(0, 1),
                        item_emb.transpose(0, 1),
                        item_emb.transpose(0, 1)
                    )
                    user_interests = attn_output.transpose(0, 1)
                
                if training and target_items is not None:
                    # Get target item embeddings
                    if self.item_encoder is not None:
                        target_emb = self.item_encoder(target_items)
                    else:
                        target_emb = self.item_embedding(target_items)
                    
                    # Project user interests
                    projected_interests = torch.stack([
                        self.label_predictor(interest) for interest in user_interests
                    ], dim=1)
                    
                    # Compute scores based on maximum similarity
                    scores = torch.bmm(projected_interests, target_emb.unsqueeze(2)).squeeze()
                    scores, _ = scores.max(dim=1)
                    
                    return scores
                else:
                    # Just return user interests for recommendation
                    return user_interests
        
        self.model = MINDModule(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_interests=self.num_interests,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            routing_iterations=self.routing_iterations,
            interest_extractor=self.interest_extractor,
            feature_type=self.feature_type,
            item_encoder=self.item_encoder
        ).to(self.device)
        
        return self.model
    
    def fit(self, interaction_matrix, user_ids: List[int], item_ids: List[int]):
        """
        Train the model using user-item interactions.
        
        Parameters:
        - interaction_matrix: User-item interaction matrix (scipy sparse matrix)
        - user_ids: List of user IDs
        - item_ids: List of item IDs
        """
        # Validate inputs
        validate_fit_inputs(user_ids, item_ids, ratings)
        
        self.num_items = len(item_ids)
        
        # Create user history dictionary
        for u_idx, user_id in enumerate(user_ids):
            user_items = interaction_matrix[u_idx].indices.tolist()
            if user_items:  # Only add users with at least one interaction
                self.user_history[user_id] = user_items
        
        # Build the model
        self._build_model()
        
        # Prepare training data
        train_sequences = []
        train_targets = []
        
        for user_id, items in self.user_history.items():
            if len(items) < 2:  # Skip users with too few interactions
                continue
                
            # Use all but last item as input sequence
            history = items[:-1]
            target = items[-1]
            
            # Truncate or pad sequence
            if len(history) > self.max_seq_length:
                history = history[-self.max_seq_length:]
            else:
                history = [0] * (self.max_seq_length - len(history)) + history
                
            train_sequences.append(history)
            train_targets.append(target)
        
        # Convert to tensors
        train_sequences = torch.LongTensor(train_sequences).to(self.device)
        train_targets = torch.LongTensor(train_targets).to(self.device)
        
        # Setup optimizer
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.l2_regularization
            )
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_regularization
            )
        else:
            optimizer = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2_regularization
            )
            
        # Setup loss function
        if self.loss_type == 'bpr':
            criterion = lambda pos, neg: -torch.mean(torch.log(torch.sigmoid(pos - neg)))
        elif self.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:  # hinge
            criterion = lambda pos, neg: torch.mean(torch.clamp(1 - pos + neg, min=0))
            
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Process in batches
            num_batches = (len(train_sequences) + self.batch_size - 1) // self.batch_size
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_sequences))
                
                if end_idx <= start_idx:
                    continue
                    
                batch_sequences = train_sequences[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                # Generate negative samples (random items)
                batch_negatives = torch.randint(
                    1, self.num_items + 1, 
                    size=batch_targets.size(),
                    device=self.device
                )
                
                # Forward pass for positive examples
                pos_scores = self.model(batch_sequences, batch_targets)
                
                # Forward pass for negative examples
                neg_scores = self.model(batch_sequences, batch_negatives)
                
                # Compute loss
                loss = criterion(pos_scores, neg_scores)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * (end_idx - start_idx)
                
            # Calculate average loss
            avg_loss = total_loss / len(train_sequences)
            
            if self.verbose:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
                
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        return self

    def recommend(self, user_id: int, top_n: int = 10) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Parameters:
        - user_id: User ID
        - top_n: Number of recommendations to generate
        
        Returns:
        - List of recommended item IDs
        """
        # Validate inputs
        validate_model_fitted(self.is_fitted, self.name)
        validate_user_id(user_id, self.user_map if hasattr(self, 'user_map') else {})
        validate_top_k(top_k if 'top_k' in locals() else 10)
        
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        if user_id not in self.user_history:
            return []  # Cannot recommend for unknown users
            
        # Get user history
        history = self.user_history[user_id]
        
        # Prepare sequence
        if len(history) > self.max_seq_length:
            seq = history[-self.max_seq_length:]
        else:
            seq = [0] * (self.max_seq_length - len(history)) + history
            
        # Convert to tensor
        seq_tensor = torch.LongTensor([seq]).to(self.device)
        
        # Generate user interests
        self.model.eval()
        with torch.no_grad():
            user_interests = self.model(seq_tensor, training=False)
            
        # Compute scores for all items
        scores = np.zeros(self.num_items + 1)
        
        # For each interest vector
        for interest_idx in range(self.num_interests):
            interest = user_interests[0, interest_idx].cpu().numpy()
            
            # Project interest through label predictor
            interest_tensor = torch.tensor(interest, device=self.device).float()
            projected_interest = self.model.label_predictor(interest_tensor).cpu().numpy()
            
            # Compute similarity with all item embeddings
            for item_idx in range(1, self.num_items + 1):  # Skip padding item
                item_emb = self.model.item_embedding(torch.tensor([item_idx], device=self.device)).cpu().numpy()[0]
                similarity = np.dot(projected_interest, item_emb)
                
                # Update score with maximum similarity across interests
                scores[item_idx] = max(scores[item_idx], similarity)
                
        # Set score of padding item and seen items to -inf
        scores[0] = float('-inf')
        for item in history:
            scores[item] = float('-inf')
            
        # Get top-N items
        top_items = np.argsort(scores)[::-1][:top_n].tolist()
        
        return top_items
        
    def save(self, filepath):
        """Save model to disk"""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        state = {
            'config': {
                'embedding_dim': self.embedding_dim,
                'num_interests': self.num_interests,
                'hidden_layers': self.hidden_layers,
                'dropout': self.dropout,
                'routing_iterations': self.routing_iterations,
                'interest_extractor': self.interest_extractor,
                'max_seq_length': self.max_seq_length,
                'feature_type': self.feature_type
            },
            'num_items': self.num_items,
            'user_history': self.user_history,
            'model_state': self.model.state_dict()
        }
        
        torch.save(state, filepath)
        
    @classmethod
    def load(cls, filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Load model from disk"""
        state = torch.load(filepath, map_location=device)
        
        # Create model with saved configuration
        model = cls(
            embedding_dim=state['config']['embedding_dim'],
            num_interests=state['config']['num_interests'],
            hidden_layers=state['config']['hidden_layers'],
            dropout=state['config']['dropout'],
            routing_iterations=state['config']['routing_iterations'],
            interest_extractor=state['config']['interest_extractor'],
            max_seq_length=state['config']['max_seq_length'],
            feature_type=state['config']['feature_type'],
            device=device
        )
        
        # Restore state
        model.num_items = state['num_items']
        model.user_history = state['user_history']
        
        # Build and load model weights
        model._build_model()
        model.model.load_state_dict(state['model_state'])
        model.is_fitted = True
        
        return model