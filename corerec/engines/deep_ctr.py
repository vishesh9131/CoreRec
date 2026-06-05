"""Production-grade deep CTR / feature-interaction recommenders.

A shared base (:class:`_DeepCTRBase`) provides the full unified contract every
production CoreRec model has -- ``fit`` (with negative sampling so implicit
ranking actually trains), ``predict``, a vectorized ``recommend``, ``save``/``load``
with exact round-trip, a device flag, and a post-fit collapse guard. Each concrete
model is a small subclass that defines only its feature-interaction module, mapping
stacked field embeddings ``[B, F, d]`` to a logit ``[B]``.

Models implemented here: FM, AFM, NFM, DeepFM, DCN, AutoInt, xDeepFM, FiBiNet,
PNN, WideDeep, GMF, MLP. They share the user/item-id field representation (side
features can be added later); this is the same ID-based setup the existing
production DCN/DeepFM use, so they are directly comparable and uniformly tested.

Note on the elaborate feature-interaction models (PNN, xDeepFM, FiBiNet, AutoInt):
their cross-feature machinery (product layers, CIN, SENet, multi-head attention) is
designed for *many* feature fields and is largely idle with only two ID fields, so
on pure ID-based top-K they tend to trail the simpler FM/DeepFM/DCN and the
neighbourhood/MF/EASE models. They come into their own once side features are added.

Reference: Rendle 2010 (FM); Xiao 2017 (AFM); He 2017 (NFM); Guo 2017 (DeepFM);
Wang 2017 (DCN); Song 2019 (AutoInt); Lian 2018 (xDeepFM); Huang 2019 (FiBiNet);
Qu 2016 (PNN); Cheng 2016 (Wide&Deep).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from corerec.api.base_recommender import BaseRecommender

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Interaction modules: input fields [B, F, d] -> logit contribution [B]
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, in_dim, dims, dropout):
        super().__init__()
        layers = []
        for h in dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fm_second_order(fields):
    # 0.5 * ((sum_f e_f)^2 - sum_f e_f^2) summed over the embedding dim -> [B]
    s = fields.sum(dim=1)
    sq = (fields ** 2).sum(dim=1)
    return 0.5 * (s * s - sq).sum(dim=1)


class FMInteraction(nn.Module):
    def forward(self, fields):
        return _fm_second_order(fields)


class DeepInteraction(nn.Module):  # the "deep" half / Wide&Deep deep tower
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        self.mlp = _MLP(n_fields * dim, dims, dropout)

    def forward(self, fields):
        return self.mlp(fields.flatten(1))


class DeepFMInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        self.deep = DeepInteraction(n_fields, dim, dims, dropout)

    def forward(self, fields):
        return _fm_second_order(fields) + self.deep(fields)


class NFMInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        self.mlp = _MLP(dim, dims, dropout)

    def forward(self, fields):
        s = fields.sum(dim=1)
        sq = (fields ** 2).sum(dim=1)
        bi = 0.5 * (s * s - sq)  # bi-interaction pooling -> [B, d]
        return self.mlp(bi)


class AFMInteraction(nn.Module):
    def __init__(self, n_fields, dim, attn_dim=16, dropout=0.1):
        super().__init__()
        self.attn = nn.Linear(dim, attn_dim)
        self.proj = nn.Linear(attn_dim, 1, bias=False)
        self.out = nn.Linear(dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        idx = torch.triu_indices(n_fields, n_fields, offset=1)
        self.register_buffer("ti", idx[0]); self.register_buffer("tj", idx[1])

    def forward(self, fields):
        p = fields[:, self.ti] * fields[:, self.tj]            # [B, P, d]
        a = torch.softmax(self.proj(torch.relu(self.attn(p))), dim=1)  # [B, P, 1]
        out = (a * p).sum(dim=1)                               # [B, d]
        return self.out(self.drop(out)).squeeze(-1)


class DCNInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout, n_cross=3):
        super().__init__()
        idim = n_fields * dim
        self.w = nn.ParameterList([nn.Parameter(torch.randn(idim) * 0.01) for _ in range(n_cross)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(idim)) for _ in range(n_cross)])
        self.deep = _MLP(idim, dims, dropout)
        self.head = nn.Linear(idim + 1, 1)

    def forward(self, fields):
        x0 = fields.flatten(1)
        x = x0
        for w, b in zip(self.w, self.b):
            x = x0 * (x * w).sum(1, keepdim=True) + b + x
        d = self.deep(x0).unsqueeze(-1)
        return self.head(torch.cat([x, d], dim=1)).squeeze(-1)


class AutoIntInteraction(nn.Module):
    def __init__(self, n_fields, dim, heads=2, layers=2, dropout=0.1):
        super().__init__()
        self.att = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            for _ in range(layers)])
        self.out = nn.Linear(n_fields * dim, 1)

    def forward(self, fields):
        x = fields
        for a in self.att:
            h, _ = a(x, x, x)
            x = torch.relu(x + h)
        return self.out(x.flatten(1)).squeeze(-1)


class CIN(nn.Module):
    def __init__(self, n_fields, dim, sizes=(16, 16)):
        super().__init__()
        self.convs = nn.ModuleList()
        prev = n_fields
        self.sizes = sizes
        for s in sizes:
            self.convs.append(nn.Conv1d(n_fields * prev, s, 1))
            prev = s
        self.head = nn.Linear(sum(sizes), 1)

    def forward(self, fields):
        B, F, D = fields.shape
        x0 = fields
        xk = fields
        pools = []
        for conv in self.convs:
            # outer product along field dim
            o = torch.einsum("bhd,bmd->bhmd", xk, x0).reshape(B, -1, D)
            xk = torch.relu(conv(o))
            pools.append(xk.sum(dim=2))
        return self.head(torch.cat(pools, dim=1)).squeeze(-1)


class xDeepFMInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        self.cin = CIN(n_fields, dim)
        self.deep = DeepInteraction(n_fields, dim, dims, dropout)

    def forward(self, fields):
        return self.cin(fields) + self.deep(fields)


class FiBiNetInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout, reduction=2):
        super().__init__()
        r = max(1, n_fields // reduction)
        self.senet = nn.Sequential(nn.Linear(n_fields, r), nn.ReLU(), nn.Linear(r, n_fields), nn.ReLU())
        self.bilinear = nn.Linear(dim, dim, bias=False)
        idx = torch.triu_indices(n_fields, n_fields, offset=1)
        self.register_buffer("ti", idx[0]); self.register_buffer("tj", idx[1])
        self.mlp = _MLP(2 * idx.shape[1] * dim, dims, dropout)

    def _bi(self, f):
        p = self.bilinear(f[:, self.ti]) * f[:, self.tj]
        return p.flatten(1)

    def forward(self, fields):
        w = self.senet(fields.mean(dim=2))            # [B, F]
        fields_se = fields * w.unsqueeze(-1)
        out = torch.cat([self._bi(fields), self._bi(fields_se)], dim=1)
        return self.mlp(out)


class GMFInteraction(nn.Module):
    """Generalized matrix factorization: element-wise product -> linear (NeuMF-GMF)."""
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Linear(dim, 1, bias=False)

    def forward(self, fields):
        return self.out(fields[:, 0] * fields[:, 1]).squeeze(-1)


class MLPInteraction(nn.Module):
    """Concatenate fields and pass through an MLP (NeuMF-MLP)."""
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        self.mlp = _MLP(n_fields * dim, dims, dropout)

    def forward(self, fields):
        return self.mlp(fields.flatten(1))


class PNNInteraction(nn.Module):
    def __init__(self, n_fields, dim, dims, dropout):
        super().__init__()
        idx = torch.triu_indices(n_fields, n_fields, offset=1)
        self.register_buffer("ti", idx[0]); self.register_buffer("tj", idx[1])
        self.mlp = _MLP(n_fields * dim + idx.shape[1], dims, dropout)

    def forward(self, fields):
        ip = (fields[:, self.ti] * fields[:, self.tj]).sum(dim=2)   # inner products [B, P]
        return self.mlp(torch.cat([fields.flatten(1), ip], dim=1))


_INTERACTIONS = {
    "FM": lambda nf, d, dims, dr: FMInteraction(),
    "AFM": lambda nf, d, dims, dr: AFMInteraction(nf, d, dropout=dr),
    "NFM": lambda nf, d, dims, dr: NFMInteraction(nf, d, dims, dr),
    "DeepFM": lambda nf, d, dims, dr: DeepFMInteraction(nf, d, dims, dr),
    "DCN": lambda nf, d, dims, dr: DCNInteraction(nf, d, dims, dr),
    "AutoInt": lambda nf, d, dims, dr: AutoIntInteraction(nf, d, dropout=dr),
    "xDeepFM": lambda nf, d, dims, dr: xDeepFMInteraction(nf, d, dims, dr),
    "FiBiNet": lambda nf, d, dims, dr: FiBiNetInteraction(nf, d, dims, dr),
    "PNN": lambda nf, d, dims, dr: PNNInteraction(nf, d, dims, dr),
    "WideDeep": lambda nf, d, dims, dr: DeepInteraction(nf, d, dims, dr),
    "GMF": lambda nf, d, dims, dr: GMFInteraction(d),
    "MLP": lambda nf, d, dims, dr: MLPInteraction(nf, d, dims, dr),
}


class _CTRNet(nn.Module):
    def __init__(self, n_users, n_items, dim, interaction, use_sigmoid):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.interaction = interaction
        self.use_sigmoid = use_sigmoid
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.user_b.weight); nn.init.zeros_(self.item_b.weight)

    def forward(self, u, i):
        fields = torch.stack([self.user_emb(u), self.item_emb(i)], dim=1)  # [B,2,d]
        first = self.user_b(u).squeeze(-1) + self.item_b(i).squeeze(-1) + self.bias
        logit = first + self.interaction(fields)
        return torch.sigmoid(logit) if self.use_sigmoid else logit


class _DeepCTRBase(BaseRecommender):
    """Shared base for deep CTR / feature-interaction recommenders."""

    MODEL: str = "FM"  # overridden by subclasses

    def __init__(self, name: str = None, embedding_dim: int = 32,
                 mlp_dims: List[int] = None, dropout: float = 0.2,
                 learning_rate: float = 1e-3, batch_size: int = 4096,
                 epochs: int = 40, num_negatives: int = 4, task: str = "auto",
                 reg: float = 1e-5, verbose: bool = False,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 seed: int = 42, trainable: bool = True):
        super().__init__(name=name or self.MODEL, trainable=trainable, verbose=verbose)
        if task not in ("auto", "implicit", "rating"):
            raise ValueError("task must be 'auto', 'implicit' or 'rating'")
        self.embedding_dim = embedding_dim
        self.mlp_dims = mlp_dims or [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_negatives = num_negatives
        self.task = task
        self.reg = reg
        self.device = device
        self.seed = seed
        self.model = None
        self.user_map = {}
        self.item_map = {}
        self._fit_task = None

    # -- subclass hook -------------------------------------------------- #
    def _make_interaction(self):
        return _INTERACTIONS[self.MODEL](2, self.embedding_dim, self.mlp_dims, self.dropout)

    # -- contract: fit -------------------------------------------------- #
    def fit(self, user_ids, item_ids, ratings, **kwargs) -> "_DeepCTRBase":
        (user_ids, item_ids, ratings), _ = self._unpack_fit_args(
            user_ids, item_ids, ratings, supported_modes=("triplet",))
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        u = np.asarray(user_ids); it = np.asarray(item_ids); r = np.asarray(ratings, dtype=float)

        users = sorted(set(u.tolist())); items = sorted(set(it.tolist()))
        self.user_map = {x: k for k, x in enumerate(users)}
        self.item_map = {x: k for k, x in enumerate(items)}
        self.uid_map = self.user_map; self.iid_map = self.item_map
        self.num_users = len(users); self.num_items = len(items)
        self.reverse_item_map = {k: x for x, k in self.item_map.items()}
        uidx = np.fromiter((self.user_map[x] for x in u.tolist()), dtype=np.int64)
        iidx = np.fromiter((self.item_map[x] for x in it.tolist()), dtype=np.int64)

        task = self.task if self.task != "auto" else "implicit"
        self._fit_task = task
        dev = torch.device(self.device if (self.device == "cpu" or torch.cuda.is_available()) else "cpu")

        if task == "rating":
            tu, ti_, ty = uidx, iidx, r.astype(np.float32)
        else:
            seen = [set() for _ in range(self.num_users)]
            for a, b in zip(uidx.tolist(), iidx.tolist()):
                seen[a].add(b)
            rng = np.random.RandomState(self.seed)
            tu = [uidx]; ti_ = [iidx]; ty = [np.ones(len(uidx), np.float32)]
            negs = rng.randint(0, self.num_items, size=(len(uidx), self.num_negatives))
            tu.append(np.repeat(uidx, self.num_negatives))
            ti_.append(negs.reshape(-1))
            ty.append(np.zeros(len(uidx) * self.num_negatives, np.float32))
            tu = np.concatenate(tu); ti_ = np.concatenate(ti_); ty = np.concatenate(ty)

        self.model = _CTRNet(self.num_users, self.num_items, self.embedding_dim,
                             self._make_interaction(), use_sigmoid=(task != "rating")).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        crit = nn.BCELoss() if task != "rating" else nn.MSELoss()
        tu_t = torch.as_tensor(tu, device=dev); ti_t = torch.as_tensor(ti_, device=dev)
        ty_t = torch.as_tensor(ty, device=dev)
        n = tu_t.shape[0]
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0; nb = 0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                if idx.numel() < 2:
                    continue
                out = self.model(tu_t[idx], ti_t[idx])
                loss = crit(out, ty_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if self.verbose and (ep + 1) % 10 == 0:
                logger.info(f"{self.MODEL} epoch {ep+1}/{self.epochs} loss={tot/max(1,nb):.4f}")

        self.model.eval()
        with torch.no_grad():
            probe = self.model(tu_t[:min(4096, n)], ti_t[:min(4096, n)])
            if float(probe.std()) < 1e-4:
                logger.warning("%s output collapsed (std~0); check label scale/task.", self.MODEL)
        self._dev = dev
        self.is_fitted = True
        return self

    # -- contract: predict / recommend --------------------------------- #
    def predict(self, user_id, item_id, **kwargs) -> float:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0
        u = torch.tensor([self.user_map[user_id]], device=self._dev)
        i = torch.tensor([self.item_map[item_id]], device=self._dev)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(u, i).item())

    def _score_all_items(self, user_id) -> np.ndarray:
        u = torch.full((self.num_items,), self.user_map[user_id], device=self._dev, dtype=torch.long)
        i = torch.arange(self.num_items, device=self._dev)
        self.model.eval()
        with torch.no_grad():
            return self.model(u, i).detach().cpu().numpy()

    def recommend(self, user_id, top_k: int = 10, exclude_items=None, **kwargs) -> List[Any]:
        if not self.is_fitted:
            from corerec.api.exceptions import ModelNotFittedError
            raise ModelNotFittedError()
        if user_id not in self.user_map:
            return []
        exclude = set(exclude_items or [])
        scores = self._score_all_items(user_id)
        out = []
        for idx in np.argsort(-scores):
            iid = self.reverse_item_map[int(idx)]
            if iid in exclude:
                continue
            out.append(iid)
            if len(out) >= top_k:
                break
        return out

    # -- contract: save / load ----------------------------------------- #
    def save(self, path: Union[str, Path], **kwargs) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "cfg": {"name": self.name, "embedding_dim": self.embedding_dim,
                    "mlp_dims": self.mlp_dims, "dropout": self.dropout,
                    "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                    "epochs": self.epochs, "num_negatives": self.num_negatives,
                    "task": self.task, "reg": self.reg, "device": self.device,
                    "seed": self.seed},
            "model_class": self.__class__.__name__,
            "fit_task": self._fit_task,
            "user_map": self.user_map, "item_map": self.item_map,
            "num_users": self.num_users, "num_items": self.num_items,
            "state_dict": self.model.state_dict() if self.model else None,
        }, p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "_DeepCTRBase":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        inst = cls(**ckpt["cfg"])
        inst.user_map = ckpt["user_map"]; inst.item_map = ckpt["item_map"]
        inst.uid_map = inst.user_map; inst.iid_map = inst.item_map
        inst.num_users = ckpt["num_users"]; inst.num_items = ckpt["num_items"]
        inst.reverse_item_map = {k: x for x, k in inst.item_map.items()}
        inst._fit_task = ckpt.get("fit_task", "implicit")
        dev = torch.device("cpu")
        inst.model = _CTRNet(inst.num_users, inst.num_items, inst.embedding_dim,
                             inst._make_interaction(), use_sigmoid=(inst._fit_task != "rating")).to(dev)
        if ckpt["state_dict"] is not None:
            inst.model.load_state_dict(ckpt["state_dict"])
        inst.model.eval()
        inst._dev = dev; inst.device = "cpu"; inst.is_fitted = True
        return inst


# --------------------------------------------------------------------------- #
# Concrete models -- each is a one-line subclass of the shared base
# --------------------------------------------------------------------------- #
class FM(_DeepCTRBase):       MODEL = "FM"
class AFM(_DeepCTRBase):      MODEL = "AFM"
class NFM(_DeepCTRBase):      MODEL = "NFM"
class DeepFMCTR(_DeepCTRBase):MODEL = "DeepFM"
class DCNCTR(_DeepCTRBase):   MODEL = "DCN"
class AutoInt(_DeepCTRBase):  MODEL = "AutoInt"
class xDeepFM(_DeepCTRBase):  MODEL = "xDeepFM"
class FiBiNet(_DeepCTRBase):  MODEL = "FiBiNet"
class PNN(_DeepCTRBase):      MODEL = "PNN"
class WideDeep(_DeepCTRBase): MODEL = "WideDeep"
class GMF(_DeepCTRBase):      MODEL = "GMF"
class MLP(_DeepCTRBase):      MODEL = "MLP"

__all__ = ["FM", "AFM", "NFM", "DeepFMCTR", "DCNCTR", "AutoInt", "xDeepFM",
           "FiBiNet", "PNN", "WideDeep", "GMF", "MLP"]
