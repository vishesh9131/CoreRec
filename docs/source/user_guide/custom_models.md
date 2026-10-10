# Custom Models in PyTorch

Research a new recommender in plain PyTorch and get training, evaluation,
serving and ONNX export from CoreRec. You write an `nn.Module`; CoreRec
handles id mapping, negative sampling, the training loop, devices (CUDA,
Apple MPS, CPU), seen-item filtering and persistence.

## The contract

Your module implements one method:

```text
forward(query, items) -> scores
  query   LongTensor [B]           user indices          (inputs="user")
          LongTensor [B, max_len]  item-index histories,  (inputs="history")
                                   oldest first, left-padded with 0
  items   LongTensor [B, K]        item indices
  scores  FloatTensor [B, K]       higher = better
```

Item indices run `1..n_items` and `0` is padding, so item tables are
`nn.Embedding(n_items + 1, d, padding_idx=0)`. User indices run
`0..n_users-1`. CoreRec builds your module as
`MyModel(n_users=..., n_items=..., max_len=..., **module_kwargs)`, passing only
the arguments your `__init__` declares.

Optional: `score_all(query) -> [B, n_items + 1]` scores the whole catalogue in
one call (column 0 is ignored). Without it, `forward` is called on every item.

## A first model

```python
import numpy as np
import pandas as pd
import torch.nn as nn
from corerec.nn import Recommender


class DotModel(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.users = nn.Embedding(n_users, dim)
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        # nn.Embedding starts at N(0, 1): dot products that large stall BPR
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)

    def forward(self, users, items):
        return (self.users(users).unsqueeze(1) * self.items(items)).sum(-1)


# toy data with structure: user u mostly picks from item cluster u % 20
rng = np.random.default_rng(0)
users = rng.integers(0, 200, 4000)
df = pd.DataFrame({"user_id": users,
                   "item_id": (users % 20) * 15 + rng.integers(0, 15, 4000)})
test = df.groupby("user_id").tail(2)          # hold out each user's last 2
train = df.drop(test.index)

rec = Recommender(DotModel, {"dim": 32}, loss="bpr", epochs=10, lr=0.01)
rec.fit(train)
print(rec.recommend(0, top_k=10))
print(rec.history_)       # training loss per epoch
```

Pass held-out data to watch a ranking metric while training and stop when it
stops improving:

```python
rec = Recommender(DotModel, {"dim": 32}, loss="bpr", epochs=50, lr=0.01)
rec.fit(train, validation=test, patience=3)
print(rec.val_history_)   # NDCG@10 on `test` after each epoch
```

Training stops after `patience` epochs without a better NDCG@10, and the
model keeps the weights from its best epoch.

`fit` takes a DataFrame (`user_id`, `item_id`, optional `rating` and
`timestamp`) or parallel lists. Rows with `rating <= 0` are dropped, and a
`timestamp` column sets event order.

## Sequential models

With `inputs="history"`, every event after a user's first becomes a training
example, predicted from the events before it:

```python
from corerec.nn import SASRecBlock, causal_mask


class TinySeq(nn.Module):
    def __init__(self, n_items, max_len, dim=32):
        super().__init__()
        self.items = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        self.block = SASRecBlock(dim, 1, 0.1)

    def forward(self, history, items):
        x = self.items(history) + self.pos.weight[: history.shape[1]]
        x = self.block(x, causal_mask(history.shape[1], history.device), None)
        return (x[:, -1].unsqueeze(1) * self.items(items)).sum(-1)


train = train.assign(timestamp=np.arange(len(train)))
seq = Recommender(TinySeq, inputs="history", loss="sampled_softmax",
                  num_negatives=50, max_len=20, epochs=5)
seq.fit(train)
```

## Building blocks

| `corerec.nn` | What it is |
|---|---|
| `SASRecBlock(d, heads, dropout)` | Causal self-attention block (Kang & McAuley 2018) |
| `HSTUBlock(d, heads, dqk, dv, dropout, max_len, time_buckets, use_time)` | HSTU block (Zhai et al., ICML 2024) |
| `causal_mask(n)` | The `[n, n]` mask both blocks take |
| `CrossLayer(dim)` | DCN cross layer |
| `FMInteraction()` | Factorization-machine pairwise term over `[B, F, d]` |
| `MLP(dims, dropout)` | Linear/ReLU/Dropout stack |
| `bpr_loss`, `bce_loss`, `sampled_softmax_loss` | Losses over `[B, 1 + K]` scores, positive in column 0 |

These are the same blocks CoreRec's `HSTU`, `SASRec` and `DCN` models use.
`corerec.nn.models` has two complete templates, `MatrixFactorization` and
`SequentialTransformer`. A custom loss is any `callable(scores) -> loss` over
the `[B, 1 + K]` matrix: `Recommender(MyModel, loss=my_loss)`.

## Everything else works

The wrapped model is an ordinary CoreRec recommender:

```python
from corerec.evaluation import Evaluator
from corerec.serving import ModelLoader, ModelServer

truth = test.groupby("user_id").item_id.apply(list).to_dict()
print(Evaluator(metrics=["ndcg@10", "recall@10"]).evaluate(rec, truth))

rec.save("dot.pt")
same = ModelLoader().load("dot.pt")     # rebuilds DotModel from its import path
server = ModelServer(same)              # server.run() serves it over HTTP
```

`corerec.export.to_onnx(rec, "dot.onnx")` exports it for ONNX Runtime (see
[ONNX Export](onnx_export.md)).

Saving records the module's import path. A module defined in a notebook or
script reloads fine in that same session; to load it in another process, put
it in an importable file or pass it explicitly:
`Recommender.load("dot.pt", module_cls=DotModel)`.
