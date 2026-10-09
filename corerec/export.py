"""
Export a trained model to ONNX, for serving without Python or PyTorch.

    from corerec.export import to_onnx
    to_onnx(model, "model.onnx")

Every export has one output, ``scores`` [batch, n_items]: column j is the score
of ``item_ids[j]``, the same scores ``model.recommend()`` ranks. The input is

- ``user_index`` int64 [batch]           TwoTower, DCN, DeepFM
- ``history``    int64 [batch, max_len]  SASRec: item indices (1-based,
  ``item_ids[k-1]``), oldest first, left-padded with 0

The raw ids live in the file's metadata (``user_ids``, ``item_ids`` as JSON, in
index order), so the .onnx is all a server needs:

    import json, onnxruntime as ort
    sess = ort.InferenceSession("model.onnx")
    meta = sess.get_modelmeta().custom_metadata_map
    users, items = json.loads(meta["user_ids"]), json.loads(meta["item_ids"])
    scores = sess.run(None, {"user_index": np.array([users.index(u)])})[0]

Excluding already-seen items is left to the caller, as it needs the history.
Needs ``pip install onnx`` (and onnxruntime to run the result).
"""

import inspect
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn


class _AllItems(nn.Module):
    """Pointwise CTR network -> user_index [B] to scores [B, n_items]."""

    def __init__(self, net, user_feature_ids, item_feature_ids, n_fields):
        super().__init__()
        self.net, self.n_fields = net, n_fields
        # these models index users/items by feature id; keep that lookup in the
        # graph so user_index is just the position in user_ids, as for TwoTower
        self.register_buffer("users", torch.as_tensor(user_feature_ids, dtype=torch.long))
        self.register_buffer("items", torch.as_tensor(item_feature_ids, dtype=torch.long))

    def forward(self, user_index):
        # no arithmetic on the batch size: tracing would freeze it as a constant
        n = self.items.shape[0]
        u = self.users[user_index].unsqueeze(1).expand(-1, n)          # [B, n]
        i = self.items.unsqueeze(0).expand_as(u)                       # [B, n]
        pad = [torch.zeros_like(u)] * (self.n_fields - 2)
        x = torch.stack([u, i, *pad], dim=-1).reshape(-1, self.n_fields)
        return self.net(x).reshape(-1, n)


class _TwoTowerScores(nn.Module):
    def __init__(self, net, item_emb):
        super().__init__()
        self.net = net
        self.register_buffer("item_emb", torch.as_tensor(item_emb, dtype=torch.float32))

    def forward(self, user_index):
        return self.net.encode_user(user_index.unsqueeze(1)) @ self.item_emb.t()


class _SASRecScores(nn.Module):
    def __init__(self, net, popularity):
        super().__init__()
        self.net = net
        pop = torch.zeros(net.item_emb.weight.shape[0] - 1) if popularity is None else \
            torch.as_tensor(popularity, dtype=torch.float32)
        self.register_buffer("pop", pop)

    def forward(self, history):
        h = self.net(history, history == 0)[:, -1, :]
        return (h @ self.net.item_emb.weight.t())[:, 1:] - self.pop


def _ids(seq):
    return json.dumps([x.item() if isinstance(x, np.generic) else x for x in seq], default=str)


def _ordered(id_map):
    return [k for k, _ in sorted(id_map.items(), key=lambda kv: kv[1])]


class _RecommenderScores(nn.Module):
    """corerec.nn.Recommender: the user's module, scored over the whole catalogue."""

    def __init__(self, rec):
        super().__init__()
        self.net, self._rec = rec.model, rec

    def forward(self, query):
        return self._rec._score_all(query)


def _wrap(model):
    """(module, example input, input name, metadata) for a supported model."""
    from corerec.nn.recommender import Recommender

    name = type(model).__name__
    if isinstance(model, Recommender):
        meta = {"user_ids": model._users, "item_ids": model._items}
        if model.inputs == "user":
            return _RecommenderScores(model), torch.zeros(2, dtype=torch.long), "user_index", meta
        ex = torch.zeros(2, model.max_len, dtype=torch.long)
        ex[:, -1] = 1
        meta["max_seq_length"] = model.max_len
        return _RecommenderScores(model), ex, "history", meta
    if name == "TwoTower":
        if model.user_input_dim != len(model.user_map):
            raise NotImplementedError("TwoTower trained on user features can't be exported yet")
        users, items = _ordered(model.user_map), _ordered(model.item_map)
        mod = _TwoTowerScores(model.model, model.item_embeddings_cache)
        return mod, torch.zeros(2, dtype=torch.long), "user_index", {"user_ids": users, "item_ids": items}
    if name in ("DCN", "DeepFM"):
        if name == "DCN":
            fu, fi = model.user_map, model.item_map
            n_fields = model.model.input_dim // model.embedding_dim
        else:
            fu, fi = model.feature_map["user"], model.feature_map["item"]
            n_fields = len(model.field_dims)
        users, items = list(fu), list(fi)
        mod = _AllItems(model.model, [fu[u] for u in users], [fi[i] for i in items], n_fields)
        return mod, torch.zeros(2, dtype=torch.long), "user_index", {"user_ids": users,
                                                                     "item_ids": items}
    if name == "SASRec":
        items = [model.index_to_item[k] for k in range(1, len(model.index_to_item) + 1)]
        pop = model.item_popularity if model.item_popularity_bias else None
        # batch of 2: tracing attention at batch 1 bakes that size into reshapes
        ex = torch.zeros(2, model.max_seq_length, dtype=torch.long)
        ex[:, -1] = 1
        return _SASRecScores(model.model, pop), ex, "history", {
            "item_ids": items, "max_seq_length": model.max_seq_length}
    raise NotImplementedError(
        f"ONNX export supports TwoTower, DCN, DeepFM, SASRec and corerec.nn.Recommender, not {name}. Classic models "
        "(ALS, EASE, ItemKNN, ...) are a matrix lookup; serve them with ModelServer.")


def to_onnx(model: Any, path: Union[str, Path], opset: int = 17) -> Path:
    """Write ``model`` to ``path`` as a self-contained ONNX graph. Returns the path."""
    import onnx

    if not getattr(model, "is_fitted", False):
        raise ValueError("fit the model before exporting it")
    mod, example, input_name, meta = _wrap(model)
    mod = mod.cpu().eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # torch>=2.9 defaults to the dynamo exporter, which needs onnxscript and can't
    # trace DCN/SASRec yet. Stay on the TorchScript one; older torch has no flag.
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        torch.onnx.export(
            mod, (example.cpu(),), str(path), input_names=[input_name], output_names=["scores"],
            dynamic_axes={input_name: {0: "batch"}, "scores": {0: "batch"}}, opset_version=opset,
            **kw)
    # The export moved the model to CPU; put it back where it was trained.
    model.model.to(model.device)

    proto = onnx.load(str(path))
    meta["corerec_model"] = type(model).__name__
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key = k
        entry.value = v if isinstance(v, str) else _ids(v) if isinstance(v, list) else str(v)
    onnx.save(proto, str(path))
    return path
