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
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import logging

from corerec.api.base_recommender import BaseRecommender, normalize_interactions
from corerec.api.exceptions import ModelNotFittedError
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
        self._seen_by_user = {}  # user_id -> set of item indices seen during fit
        self.is_fitted = False
    
    def fit(self, user_ids: List, item_ids: List, interactions: Optional[np.ndarray] = None,
            user_features: Optional[np.ndarray] = None,
            item_features: Optional[np.ndarray] = None,
            validation_split: float = 0.1):
        """
        Train the two-tower model.

        Accepts either:
          fit(user_ids, item_ids, interactions)  # [n_users, n_items] matrix
          fit(user_ids, item_ids, ratings)       # one entry per interaction

        user_features: optional [n_users, user_dim] feature matrix
        item_features: optional [n_items, item_dim] feature matrix
        """

        self.log.info(f"Fitting {self.name} model...")

        user_ids, item_ids, interactions = normalize_interactions(
            user_ids, item_ids, interactions
        )

        # build mappings
        self.user_map = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_map = {iid: idx for idx, iid in enumerate(item_ids)}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_map.items()}

        # What each user interacted with, by item index, so recommend(exclude_seen=True)
        # has something to exclude.
        rows, cols = np.nonzero(np.asarray(interactions) > 0)
        self._seen_by_user = {}
        for r, c in zip(rows, cols):
            self._seen_by_user.setdefault(user_ids[r], set()).add(int(c))
        
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
    
    def recommend(self, user_id: Any, top_k: int = 10,
                  exclude_items: Optional[List[Any]] = None,
                  exclude_seen: bool = True, **kwargs) -> List[Any]:
        """
        Generate recommendations for a user.

        exclude_items: item IDs to keep out of the results
        exclude_seen: also drop items this user interacted with during fit

        Returns list of item IDs ranked by score.
        """
        if not self.is_fitted:
            self._check_fitted()

        if user_id not in self.user_map:
            self.log.warning(f"Unknown user: {user_id}")
            return []

        user_idx = self.user_map[user_id]

        # encode user
        # (in practice, you'd pass actual features here)
        user_feat = torch.zeros(1, self.user_input_dim, device=self.device)
        user_feat[0, min(user_idx, self.user_input_dim - 1)] = 1.0

        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.encode_user(user_feat).cpu().numpy()

        # score all items via cached embeddings
        scores = np.dot(user_emb, self.item_embeddings_cache.T).flatten()

        # Build the set of item *indices* to drop. Both exclusions were
        # previously ignored: exclude_seen was accepted and never read, and
        # exclude_items was missing entirely even though BaseRecommender
        # declares it -- so ModelServer's /recommend route raised TypeError
        # against this model.
        blocked = set()
        if exclude_items:
            blocked.update(self.item_map[i] for i in exclude_items if i in self.item_map)
        if exclude_seen:
            blocked.update(self._seen_by_user.get(user_id, ()))

        ranked = np.argsort(scores)[::-1]
        recommendations = []
        for idx in ranked:
            if len(recommendations) == top_k:
                break
            if idx in blocked or idx not in self.reverse_item_map:
                continue
            recommendations.append(self.reverse_item_map[idx])

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
    
    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """Predict affinity score for a single user-item pair."""
        if not self.is_fitted:
            self._check_fitted()

        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        user_feat = torch.zeros(1, self.user_input_dim, device=self.device)
        user_feat[0, min(user_idx, self.user_input_dim - 1)] = 1.0

        self.model.eval()
        with torch.no_grad():
            user_emb = self.model.encode_user(user_feat).cpu().numpy()

        item_emb = self.item_embeddings_cache[item_idx]
        score = float(np.dot(user_emb.flatten(), item_emb))
        return score

    def save(self, path: Union[str, Path], safe: bool = True, **kwargs) -> None:
        """Save model to disk."""
        from corerec.api.bundle_helpers import save_map_state

        path = Path(path)
        config = {
            "name": self.name,
            "user_input_dim": self.user_input_dim,
            "item_input_dim": self.item_input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "loss_type": self.loss_type,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
        }
        state = {
            "is_fitted": self.is_fitted,
            # Stored as [user_id, [item_idx, ...]] pairs rather than a dict so the
            # user IDs survive a JSON round-trip without being coerced to strings.
            # Without this, exclude_seen=True silently stops excluding after load.
            "seen_by_user": [
                [u, sorted(int(i) for i in idxs)] for u, idxs in self._seen_by_user.items()
            ],
            **save_map_state(
                user_map=self.user_map,
                item_map=self.item_map,
                reverse_item_map=self.reverse_item_map,
            ),
        }
        arrays = {}
        if self.item_embeddings_cache is not None:
            arrays["item_embeddings_cache"] = np.asarray(self.item_embeddings_cache, dtype=np.float64)

        from corerec.api.torch_bundle import save_torch_production

        if save_torch_production(self, path, config=config, state=state, arrays=arrays, safe=safe):
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        legacy = {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_item_map": self.reverse_item_map,
            "item_embeddings_cache": self.item_embeddings_cache,
            "config": config,
            "is_fitted": self.is_fitted,
            "seen_by_user": self._seen_by_user,
        }
        torch.save(legacy, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TwoTower":
        """Load model from disk."""
        from corerec.api.bundle_helpers import load_map_state
        from corerec.api.torch_bundle import load_torch_production

        def _restore(instance, config, state, arrays, bundle):
            maps = load_map_state(
                state, "user_map", "item_map", "reverse_item_map", int_key_names=("reverse_item_map",)
            )
            instance.user_map = maps["user_map"]
            instance.item_map = maps["item_map"]
            instance.reverse_item_map = maps["reverse_item_map"]
            instance.is_fitted = state.get("is_fitted", True)
            instance._seen_by_user = {
                u: {int(i) for i in idxs} for u, idxs in state.get("seen_by_user", [])
            }
            if arrays and arrays.get("item_embeddings_cache") is not None:
                instance.item_embeddings_cache = arrays["item_embeddings_cache"]

        def _build(instance, bundle):
            if bundle.get("state_dict") is not None:
                cfg = bundle["config"]
                instance.model = TwoTowerModel(
                    user_input_dim=cfg["user_input_dim"],
                    item_input_dim=cfg["item_input_dim"],
                    embedding_dim=cfg["embedding_dim"],
                    hidden_dims=cfg["hidden_dims"],
                    dropout=cfg["dropout"],
                )

        def _factory(cfg):
            return cls(
                name=cfg["name"],
                user_input_dim=cfg["user_input_dim"],
                item_input_dim=cfg["item_input_dim"],
                embedding_dim=cfg["embedding_dim"],
                hidden_dims=cfg["hidden_dims"],
                dropout=cfg["dropout"],
                loss_type=cfg["loss_type"],
                learning_rate=cfg["lr"],
                batch_size=cfg["batch_size"],
                num_epochs=cfg["num_epochs"],
            )

        loaded = load_torch_production(cls, path, build_model=_build, restore=_restore, factory=_factory)
        if loaded is not None:
            return loaded

        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        instance = cls(
            name=cfg["name"],
            user_input_dim=cfg["user_input_dim"],
            item_input_dim=cfg["item_input_dim"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
            loss_type=cfg["loss_type"],
            learning_rate=cfg["lr"],
            batch_size=cfg["batch_size"],
            num_epochs=cfg["num_epochs"],
        )
        instance.user_map = state["user_map"]
        instance.item_map = state["item_map"]
        instance.reverse_item_map = state["reverse_item_map"]
        instance.item_embeddings_cache = state["item_embeddings_cache"]
        instance.is_fitted = state["is_fitted"]
        instance._seen_by_user = state.get("seen_by_user", {})

        if state["model_state_dict"] is not None:
            instance.model = TwoTowerModel(
                user_input_dim=cfg["user_input_dim"],
                item_input_dim=cfg["item_input_dim"],
                embedding_dim=cfg["embedding_dim"],
                hidden_dims=cfg["hidden_dims"],
                dropout=cfg["dropout"],
            )
            instance.model.load_state_dict(state["model_state_dict"])
            instance.model.eval()
        return instance

    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings (for building vector index)."""
        return self.item_embeddings_cache

