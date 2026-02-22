"""
Two-Tower architecture for efficient retrieval.

This is the industry standard for large-scale recsys.
Used by YouTube, Netflix, Uber, etc.

Key idea: separate towers encode users and items independently,
then match via dot product in embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from corerec.api.base_recommender import BaseRecommender
from corerec.core.towers import UserTower, ItemTower


class TwoTowerModel(nn.Module):
    """
    Dual encoder architecture.
    
    User tower and item tower project inputs into shared embedding space.
    Similarity = dot product of embeddings.
    """
    
    def __init__(self, 
                 user_input_dim: int,
                 item_input_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.1,
                 activation: str = "relu",
                 norm_type: Optional[str] = "batch",
                 use_bias: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # config for towers
        tower_cfg = {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "activation": activation,
            "norm": norm_type,
            "use_bias": use_bias
        }
        
        # user encoding tower
        self.user_tower = UserTower(
            input_dim=user_input_dim,
            output_dim=embedding_dim,
            config=tower_cfg
        )
        
        # item encoding tower
        self.item_tower = ItemTower(
            input_dim=item_input_dim,
            output_dim=embedding_dim,
            config=tower_cfg
        )
        
        # optional: temperature scaling for dot product
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Project user into embedding space."""
        emb = self.user_tower(user_features)
        # L2 normalize for cosine similarity behavior
        return F.normalize(emb, p=2, dim=-1)
    
    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Project item into embedding space."""
        emb = self.item_tower(item_features)
        return F.normalize(emb, p=2, dim=-1)
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between users and items.
        
        user_features: [batch_size, user_dim]
        item_features: [batch_size, item_dim] or [num_items, item_dim]
        
        Returns: similarity scores [batch_size] or [batch_size, num_items]
        """
        user_emb = self.encode_user(user_features)  # [batch, embed_dim]
        item_emb = self.encode_item(item_features)  # [batch or num_items, embed_dim]
        
        if user_emb.shape[0] == item_emb.shape[0]:
            # paired scoring
            scores = torch.sum(user_emb * item_emb, dim=-1)  # [batch]
        else:
            # user vs all items
            scores = torch.matmul(user_emb, item_emb.t())  # [batch, num_items]
        
        # apply temperature
        return scores / self.temperature
    
    def batch_score(self, user_emb: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """Score a user against multiple items efficiently."""
        # user_emb: [1, embed_dim]
        # item_embs: [N, embed_dim]
        scores = torch.matmul(user_emb, item_embs.t()).squeeze(0)  # [N]
        return scores / self.temperature


class TwoTower(BaseRecommender):
    """
    Two-Tower recommender with training logic.
    
    Supports various loss functions:
    - pointwise: BCE on (user, item) pairs
    - pairwise: BPR-style ranking loss
    - contrastive: InfoNCE (good for in-batch negatives)
    """
    
    def __init__(self,
                 name: str = "TwoTower",
                 user_input_dim: int = 64,
                 item_input_dim: int = 64,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.1,
                 loss_type: str = "bce",  # bce | bpr | infonce
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 num_epochs: int = 10,
                 device: Optional[torch.device] = None,
                 negative_samples: int = 4,
                 temperature: float = 0.07,  # for InfoNCE
                 verbose: bool = True):
        super().__init__()
        
        self.name = name
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.loss_type = loss_type.lower()
        self.lr = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.neg_samples = negative_samples
        self.temp = temperature
        self.verbose = verbose
        
        self.log = logging.getLogger(self.name)
        if verbose:
            self.log.setLevel(logging.INFO)
        
        # will be initialized in fit()
        self.model = None
        self.optimizer = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.item_embeddings_cache = None  # for fast retrieval
        self.is_fitted = False
    
    def fit(self, user_ids: List, item_ids: List, interactions: np.ndarray, 
            user_features: Optional[np.ndarray] = None,
            item_features: Optional[np.ndarray] = None,
            validation_split: float = 0.1):
        """
        Train the two-tower model.
        
        user_ids: list of user identifiers
        item_ids: list of item identifiers
        interactions: matrix [n_users, n_items] with ratings/clicks
        user_features: optional [n_users, user_dim] feature matrix
        item_features: optional [n_items, item_dim] feature matrix
        """
        
        self.log.info(f"Fitting {self.name} model...")
        
        # build mappings
        self.user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}
        
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        # if no features provided, use one-hot or learned embeddings
        if user_features is None:
            user_features = np.eye(n_users, dtype=np.float32)
        if item_features is None:
            item_features = np.eye(n_items, dtype=np.float32)
        
        # update input dims if needed
        self.user_input_dim = user_features.shape[1]
        self.item_input_dim = item_features.shape[1]
        
        # init model
        self.model = TwoTowerModel(
            user_input_dim=self.user_input_dim,
            item_input_dim=self.item_input_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # convert to torch
        user_feats_t = torch.from_numpy(user_features).float().to(self.device)
        item_feats_t = torch.from_numpy(item_features).float().to(self.device)
        
        # create training pairs
        train_data = self._create_training_pairs(interactions)
        
        if len(train_data) == 0:
            self.log.warning("No positive interactions found, cannot train")
            return self
        
        # training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            
            np.random.shuffle(train_data)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i+self.batch_size]
                
                # extract batch
                user_indices = [p[0] for p in batch]
                pos_indices = [p[1] for p in batch]
                
                # get features
                batch_users = user_feats_t[user_indices]
                batch_pos_items = item_feats_t[pos_indices]
                
                self.optimizer.zero_grad()
                
                if self.loss_type == "bce":
                    # positive samples
                    pos_scores = self.model(batch_users, batch_pos_items)
                    pos_loss = F.binary_cross_entropy_with_logits(
                        pos_scores, torch.ones_like(pos_scores)
                    )
                    
                    # negative samples
                    neg_indices = np.random.randint(0, n_items, size=(len(batch), self.neg_samples))
                    neg_indices_t = torch.from_numpy(neg_indices).long()
                    batch_neg_items = item_feats_t[neg_indices_t.reshape(-1)]
                    batch_neg_items = batch_neg_items.view(len(batch), self.neg_samples, -1)
                    
                    # repeat users for each negative
                    batch_users_exp = batch_users.unsqueeze(1).expand(-1, self.neg_samples, -1)
                    batch_users_exp = batch_users_exp.reshape(-1, batch_users.shape[-1])
                    batch_neg_items = batch_neg_items.reshape(-1, batch_neg_items.shape[-1])
                    
                    neg_scores = self.model(batch_users_exp, batch_neg_items)
                    neg_scores = neg_scores.view(len(batch), self.neg_samples).mean(dim=1)
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_scores, torch.zeros_like(neg_scores)
                    )
                    
                    loss = pos_loss + neg_loss
                
                elif self.loss_type == "bpr":
                    # BPR: positive should rank higher than negative
                    pos_scores = self.model(batch_users, batch_pos_items)
                    
                    # sample negatives
                    neg_indices = np.random.randint(0, n_items, size=len(batch))
                    batch_neg_items = item_feats_t[neg_indices]
                    neg_scores = self.model(batch_users, batch_neg_items)
                    
                    # BPR loss
                    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                
                elif self.loss_type == "infonce":
                    # contrastive loss with in-batch negatives
                    user_emb = self.model.encode_user(batch_users)
                    pos_emb = self.model.encode_item(batch_pos_items)
                    
                    # cosine similarity
                    logits = torch.matmul(user_emb, pos_emb.t()) / self.temp
                    
                    # target: diagonal (positive pairs)
                    labels = torch.arange(len(batch), device=self.device)
                    loss = F.cross_entropy(logits, labels)
                
                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            
            if self.verbose and (epoch + 1) % max(1, self.num_epochs // 10) == 0:
                self.log.info(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        # cache item embeddings for fast retrieval
        self.model.eval()
        with torch.no_grad():
            self.item_embeddings_cache = self.model.encode_item(item_feats_t).cpu().numpy()
        
        self.is_fitted = True
        self.log.info("Training complete")
        return self
    
    def _create_training_pairs(self, interactions: np.ndarray) -> List[Tuple[int, int]]:
        """Extract positive user-item pairs from interaction matrix."""
        pairs = []
        rows, cols = np.nonzero(interactions > 0)
        for u, i in zip(rows, cols):
            pairs.append((u, i))
        return pairs
    
    def recommend(self, user_id: Any, top_k: int = 10, exclude_seen: bool = True) -> List[Any]:
        """
        Generate recommendations for a user.
        
        Returns list of item IDs ranked by score.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if user_id not in self.user_map:
            self.log.warning(f"Unknown user: {user_id}")
            return []
        
        user_idx = self.user_map[user_id]
        
        # encode user
        # (in practice, you'd pass actual features here)
        user_feat = torch.zeros(1, self.user_input_dim, device=self.device)
        user_feat[0, user_idx] = 1.0  # one-hot if no features
        
        user_emb = self.model.encode_user(user_feat).cpu().numpy()
        
        # score all items via cached embeddings
        scores = np.dot(user_emb, self.item_embeddings_cache.T).flatten()
        
        # get top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        recommendations = [self.reverse_item_map[idx] for idx in top_indices if idx in self.reverse_item_map]
        
        return recommendations
    
    def get_user_embedding(self, user_id: Any) -> np.ndarray:
        """Get embedding vector for a user."""
        if user_id not in self.user_map:
            return None
        
        user_idx = self.user_map[user_id]
        user_feat = torch.zeros(1, self.user_input_dim, device=self.device)
        user_feat[0, user_idx] = 1.0
        
        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode_user(user_feat).cpu().numpy()
        
        return emb
    
    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings (for building vector index)."""
        return self.item_embeddings_cache

