# corerec/engines/unionizedFilterEngine/rbm.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from scipy.sparse import csr_matrix
import pickle
import os

from .base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError


class RBM(BaseRecommender):
    """
    Restricted Boltzmann Machine for Collaborative Filtering.
    """

    def __init__(
        self,
        n_hidden: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 100,
        n_epochs: int = 20,
        k: int = 1,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = False,
    ) -> None:

        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.k = k
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.seed = seed
        self.verbose = verbose

        torch.manual_seed(seed)
        np.random.seed(seed)

        # parameters
        self.W = None
        self.v_bias = None
        self.h_bias = None

        self.W_momentum = None
        self.vb_momentum = None
        self.hb_momentum = None

        # mappings
        self.user_to_index = {}
        self.index_to_user = {}
        self.item_to_index = {}
        self.index_to_item = {}

        # stats
        self.train_errors = []
        self.is_fitted = False

    # ----------------------------------------------------------------------
    # mappings
    # ----------------------------------------------------------------------
    def _create_mappings(self, user_ids: List[int], item_ids: List[int]):
        unique_users = sorted(set(user_ids))
        unique_items = sorted(set(item_ids))

        self.user_to_index = {u: i for i, u in enumerate(unique_users)}
        self.index_to_user = {i: u for u, i in self.user_to_index.items()}

        self.item_to_index = {it: j for j, it in enumerate(unique_items)}
        self.index_to_item = {j: it for it, j in self.item_to_index.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

    # ----------------------------------------------------------------------
    # params init
    # ----------------------------------------------------------------------
    def _init_model_parameters(self):
        self.W = torch.randn(self.n_items, self.n_hidden, device=self.device) * 0.01
        self.v_bias = torch.zeros(self.n_items, device=self.device)
        self.h_bias = torch.zeros(self.n_hidden, device=self.device)

        self.W_momentum = torch.zeros_like(self.W)
        self.vb_momentum = torch.zeros_like(self.v_bias)
        self.hb_momentum = torch.zeros_like(self.h_bias)

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------
    def _sample_hidden(self, v):
        prob = torch.sigmoid(v @ self.W + self.h_bias)
        return prob, torch.bernoulli(prob)

    def _sample_visible(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return prob, torch.bernoulli(prob)

    # ----------------------------------------------------------------------
    # CD-k
    # ----------------------------------------------------------------------
    def _contrastive_divergence(self, v0):
        # positive phase
        h0_prob, h0_sample = self._sample_hidden(v0)

        vk = v0
        hk = h0_sample

        for _ in range(self.k):
            vk_prob, vk_sample = self._sample_visible(hk)
            hk_prob, hk_sample = self._sample_hidden(vk_sample)

            vk = vk_sample
            hk = hk_sample

        # gradients
        dW = v0.t() @ h0_prob - vk_prob.t() @ hk_prob
        dv = torch.sum(v0 - vk_prob, dim=0)
        dh = torch.sum(h0_prob - hk_prob, dim=0)

        # regularization
        dW -= self.weight_decay * self.W

        return dW, dv, dh

    # ----------------------------------------------------------------------
    # params update
    # ----------------------------------------------------------------------
    def _update_parameters(self, dW, dv, dh):
        self.W_momentum = self.momentum * self.W_momentum + self.learning_rate * dW
        self.vb_momentum = self.momentum * self.vb_momentum + self.learning_rate * dv
        self.hb_momentum = self.momentum * self.hb_momentum + self.learning_rate * dh

        self.W += self.W_momentum
        self.v_bias += self.vb_momentum
        self.h_bias += self.hb_momentum

    # ----------------------------------------------------------------------
    # reconstruction error
    # ----------------------------------------------------------------------
    def _compute_reconstruction_error(self, v0):
        h_prob, h_sample = self._sample_hidden(v0)
        v_prob, _ = self._sample_visible(h_sample)
        return torch.mean(torch.sum((v0 - v_prob) ** 2, dim=1)).item()

    # ----------------------------------------------------------------------
    # fit
    # ----------------------------------------------------------------------
    def fit(self, user_ids, item_ids, ratings, timestamps=None):
        self._create_mappings(user_ids, item_ids)
        self._init_model_parameters()

        # build matrix
        rows = [self.user_to_index[u] for u in user_ids]
        cols = [self.item_to_index[i] for i in item_ids]
        mat = csr_matrix((ratings, (rows, cols)), shape=(self.n_users, self.n_items))

        data = torch.tensor(mat.toarray(), dtype=torch.float32, device=self.device)
        n_samples = data.shape[0]

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_samples)

            epoch_err = 0.0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = data[perm[start:end]]

                dW, dv, dh = self._contrastive_divergence(batch)
                self._update_parameters(dW, dv, dh)

                epoch_err += self._compute_reconstruction_error(batch) * batch.size(0)

            epoch_err /= n_samples
            self.train_errors.append(epoch_err)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs} | Recon Error: {epoch_err:.4f}")

        self.is_fitted = True
        return self

    # ----------------------------------------------------------------------
    # predict vector for user
    # ----------------------------------------------------------------------
    def _predict_user(self, user_id: int):
        if user_id not in self.user_to_index:
            raise ValueError("User not found")

        user_vec = torch.zeros(self.n_items, device=self.device)
        h_prob, h_sample = self._sample_hidden(user_vec.unsqueeze(0))
        v_prob, _ = self._sample_visible(h_prob)

        return v_prob.squeeze(0)

    # ----------------------------------------------------------------------
    # recommend
    # ----------------------------------------------------------------------
    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True):
        if not self.is_fitted:
            raise ModelNotFittedError("Call fit() first.")

        preds = self._predict_user(user_id).cpu().numpy()

        if exclude_seen:
            for item, idx in self.item_to_index.items():
                preds[idx] = -np.inf

        top_idx = np.argsort(preds)[::-1][:top_n]
        return [self.index_to_item[i] for i in top_idx]

    # ----------------------------------------------------------------------
    # save
    # ----------------------------------------------------------------------
    def save_model(self, filepath: str):
        data = {
            "W": self.W.cpu().numpy(),
            "v_bias": self.v_bias.cpu().numpy(),
            "h_bias": self.h_bias.cpu().numpy(),
            "user_to_index": self.user_to_index,
            "item_to_index": self.item_to_index,
            "index_to_user": self.index_to_user,
            "index_to_item": self.index_to_item,
            "n_items": self.n_items,
            "n_users": self.n_users,
            "train_errors": self.train_errors,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    # ----------------------------------------------------------------------
    # load
    # ----------------------------------------------------------------------
    @classmethod
    def load_model(cls, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        inst = cls()
        inst.W = torch.tensor(data["W"], device=inst.device)
        inst.v_bias = torch.tensor(data["v_bias"], device=inst.device)
        inst.h_bias = torch.tensor(data["h_bias"], device=inst.device)

        inst.user_to_index = data["user_to_index"]
        inst.item_to_index = data["item_to_index"]
        inst.index_to_user = data["index_to_user"]
        inst.index_to_item = data["index_to_item"]
        inst.n_items = data["n_items"]
        inst.n_users = data["n_users"]
        inst.train_errors = data["train_errors"]

        inst.W_momentum = torch.zeros_like(inst.W)
        inst.vb_momentum = torch.zeros_like(inst.v_bias)
        inst.hb_momentum = torch.zeros_like(inst.h_bias)

        inst.is_fitted = True
        return inst