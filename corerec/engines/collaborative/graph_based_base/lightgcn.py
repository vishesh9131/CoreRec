"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

Paper: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
by Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Tuple, Optional, Any
from pathlib import Path
import pickle
import logging
from scipy.sparse import csr_matrix

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


class LightGCNModel(nn.Module):
    """LightGCN inner neural network model."""

    def __init__(self, n_users: int, n_items: int, n_factors: int,
                 n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.dropout = dropout

        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.adj_matrix = None

    def set_adj_matrix(self, adj_matrix: torch.Tensor) -> None:
        self.adj_matrix = adj_matrix

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix has not been set")

        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)

        embeddings_list = [all_embeddings]
        for _ in range(self.n_layers):
            if self.dropout > 0 and self.training:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout)
            all_embeddings = torch.sparse.mm(self.adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )
        return user_embeddings, item_embeddings


class LightGCN(BaseRecommender):
    """
    LightGCN recommender with training, prediction, and I/O.

    Accepts interaction data as ``(user_ids, item_ids, ratings)`` triplets
    (same format as FAST/FASTRecommender) *or* a sparse ``interaction_matrix``.
    """

    def __init__(
        self,
        n_factors: int = 64,
        n_layers: int = 3,
        learning_rate: float = 0.001,
        regularization: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dropout: float = 0.0,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ):
        super().__init__()
        self.name = "LightGCN"
        self.n_factors = n_factors
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.n_users = None
        self.n_items = None
        self.user_embedding = None
        self.item_embedding = None
        self.model = None
        self.optimizer = None

        self.user_id_map: Dict[Any, int] = {}
        self.item_id_map: Dict[Any, int] = {}
        self.reverse_user_map: Dict[int, Any] = {}
        self.reverse_item_map: Dict[int, Any] = {}
        self.user_interactions: Dict[int, set] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_mappings(self, user_ids: List, item_ids: List) -> None:
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

    def _build_model(self) -> None:
        self.model = LightGCNModel(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization,
        )

    def _create_adjacency_matrix(self, interaction_matrix: csr_matrix) -> torch.Tensor:
        rows, cols = interaction_matrix.nonzero()
        user_indices = torch.LongTensor(rows)
        item_indices = torch.LongTensor(cols)

        edge_index = torch.stack([
            torch.cat([user_indices, item_indices + self.n_users]),
            torch.cat([item_indices + self.n_users, user_indices]),
        ])

        size = self.n_users + self.n_items
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            torch.Size([size, size]),
        ).to(self.device)

        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        norm_adj = torch.sparse.mm(torch.sparse.mm(d_mat, adj), d_mat)
        return norm_adj

    def _store_user_interactions(self, user_ids: List, item_ids: List) -> None:
        self.user_interactions = {}
        for uid, iid in zip(user_ids, item_ids):
            uidx = self.user_id_map.get(uid)
            iidx = self.item_id_map.get(iid)
            if uidx is not None and iidx is not None:
                self.user_interactions.setdefault(uidx, set()).add(iidx)

    def _sample_negative(self, user_idx: int) -> int:
        pos = self.user_interactions.get(user_idx, set())
        while True:
            neg = np.random.randint(0, self.n_items)
            if neg not in pos:
                return neg

    def _bpr_loss(self, users, pos_items, neg_items):
        user_emb, item_emb = self.model()
        u = user_emb[users]
        p = item_emb[pos_items]
        n = item_emb[neg_items]
        pos_scores = torch.sum(u * p, dim=1)
        neg_scores = torch.sum(u * n, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        l2 = self.regularization * (
            torch.norm(u) ** 2 + torch.norm(p) ** 2 + torch.norm(n) ** 2
        ) / len(users)
        return loss + l2

    # ------------------------------------------------------------------
    # Public API (BaseRecommender interface)
    # ------------------------------------------------------------------

    def fit(self, user_ids: List, item_ids: List, ratings: Optional[List] = None,
            interaction_matrix: Optional[csr_matrix] = None, **kwargs):
        """
        Train LightGCN.

        Accepts either triplet lists ``(user_ids, item_ids, ratings)`` or
        ``(user_ids, item_ids)`` with an explicit ``interaction_matrix``.
        """
        self._create_mappings(user_ids, item_ids)
        self._store_user_interactions(user_ids, item_ids)
        self._build_model()

        if interaction_matrix is None:
            u_idx = [self.user_id_map[u] for u in user_ids]
            i_idx = [self.item_id_map[i] for i in item_ids]
            vals = [float(r) for r in ratings] if ratings is not None else [1.0] * len(user_ids)
            interaction_matrix = csr_matrix(
                (vals, (u_idx, i_idx)),
                shape=(self.n_users, self.n_items),
            )

        norm_adj = self._create_adjacency_matrix(interaction_matrix)
        self.model.set_adj_matrix(norm_adj)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            users, pos_items, neg_items = [], [], []
            for uidx, items in self.user_interactions.items():
                for pidx in items:
                    users.append(uidx)
                    pos_items.append(pidx)
                    neg_items.append(self._sample_negative(uidx))

            if len(users) == 0:
                break

            indices = np.arange(len(users))
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(users), self.batch_size):
                batch = indices[start : start + self.batch_size]
                bu = torch.LongTensor([users[i] for i in batch]).to(self.device)
                bp = torch.LongTensor([pos_items[i] for i in batch]).to(self.device)
                bn = torch.LongTensor([neg_items[i] for i in batch]).to(self.device)

                self.optimizer.zero_grad()
                loss = self._bpr_loss(bu, bp, bn)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if self.verbose and (epoch + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.model.eval()
        with torch.no_grad():
            self.user_embedding, self.item_embedding = self.model()
        self.is_fitted = True
        return self

    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """Predict score for a single user-item pair."""
        if self.user_embedding is None or self.item_embedding is None:
            raise ValueError("Model not fitted")
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0.0
        uidx = self.user_id_map[user_id]
        iidx = self.item_id_map[item_id]
        score = torch.dot(
            self.user_embedding[uidx], self.item_embedding[iidx]
        ).detach().cpu().item()
        return float(score)

    def recommend(self, user_id: Any, top_k: int = 10,
                  exclude_items: Optional[List] = None, **kwargs) -> List[Any]:
        """Generate top-K recommendations for a user."""
        if self.user_embedding is None or self.item_embedding is None:
            raise ValueError("Model not fitted")
        if user_id not in self.user_id_map:
            return []

        uidx = self.user_id_map[user_id]
        with torch.no_grad():
            scores = torch.matmul(
                self.user_embedding[uidx], self.item_embedding.t()
            ).cpu().numpy()

        seen = self.user_interactions.get(uidx, set())
        for idx in seen:
            scores[idx] = -np.inf

        if exclude_items:
            for it in exclude_items:
                if it in self.item_id_map:
                    scores[self.item_id_map[it]] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.reverse_item_map[int(i)] for i in top_indices
                if int(i) in self.reverse_item_map]

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": {
                "n_factors": self.n_factors,
                "n_layers": self.n_layers,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "dropout": self.dropout,
                "early_stopping_patience": self.early_stopping_patience,
                "verbose": self.verbose,
            },
            "n_users": self.n_users,
            "n_items": self.n_items,
            "user_id_map": self.user_id_map,
            "item_id_map": self.item_id_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "user_interactions": {k: list(v) for k, v in self.user_interactions.items()},
            "user_embedding": (
                self.user_embedding.detach().cpu().numpy()
                if self.user_embedding is not None else None
            ),
            "item_embedding": (
                self.item_embedding.detach().cpu().numpy()
                if self.item_embedding is not None else None
            ),
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LightGCN":
        """Load model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        cfg = state["config"]
        instance = cls(**cfg)
        instance.n_users = state["n_users"]
        instance.n_items = state["n_items"]
        instance.user_id_map = state["user_id_map"]
        instance.item_id_map = state["item_id_map"]
        instance.reverse_user_map = state["reverse_user_map"]
        instance.reverse_item_map = state["reverse_item_map"]
        instance.user_interactions = {
            k: set(v) for k, v in state["user_interactions"].items()
        }
        instance.is_fitted = state["is_fitted"]

        if state["user_embedding"] is not None:
            instance.user_embedding = torch.tensor(
                state["user_embedding"]
            ).to(instance.device)
        if state["item_embedding"] is not None:
            instance.item_embedding = torch.tensor(
                state["item_embedding"]
            ).to(instance.device)

        return instance
