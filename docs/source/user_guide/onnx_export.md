# ONNX Export

Export a trained model to a single `.onnx` file and serve it with
[ONNX Runtime](https://onnxruntime.ai) from any language, without Python or
PyTorch on the server.

```bash
pip install "corerec[onnx]"
```

## Supported models

| Model | Input | Output |
|---|---|---|
| `TwoTower`, `DCN`, `DeepFM` | `user_index` int64 `[batch]` | `scores` float `[batch, n_items]` |
| `SASRec` | `history` int64 `[batch, max_seq_length]` | `scores` float `[batch, n_items]` |

Column `j` of `scores` is the score of `item_ids[j]`, the same scores
`model.recommend()` ranks. Classic models (ALS, EASE, ItemKNN, SAR, ...) are a
matrix lookup rather than a network; serve those with `ModelServer`.

## Export

```python
import numpy as np
from corerec.engines import TwoTower
from corerec.export import to_onnx

rng = np.random.default_rng(0)
users = rng.integers(0, 100, 2000).tolist()
items = rng.integers(0, 300, 2000).tolist()

model = TwoTower(embedding_dim=32, epochs=5, verbose=False)
model.fit(users, items, [1.0] * len(users))
to_onnx(model, "two_tower.onnx")
```

## Serve

The raw user and item ids are stored in the file's metadata, so the `.onnx`
file is all a server needs.

```python
import json
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("two_tower.onnx")
meta = sess.get_modelmeta().custom_metadata_map
user_ids = json.loads(meta["user_ids"])
item_ids = json.loads(meta["item_ids"])

row = user_ids.index(users[0])
scores = sess.run(None, {"user_index": np.array([row], dtype=np.int64)})[0][0]
top10 = [item_ids[j] for j in np.argsort(-scores)[:10]]
```

The batch dimension is dynamic: pass several user rows at once to score them
together.

## SASRec input

`history` holds item indices, where index `k` is `item_ids[k - 1]` (0 is
padding). Put each user's history oldest first and left-pad to
`max_seq_length` (in the metadata):

```python
max_len = int(meta["max_seq_length"])
index = {item: k + 1 for k, item in enumerate(item_ids)}
seq = [index[i] for i in history][-max_len:]
x = np.zeros((1, max_len), dtype=np.int64)
x[0, -len(seq):] = seq
scores = sess.run(None, {"history": x})[0][0]
```

## Notes

- Already-seen items are not removed by the graph; drop them before taking
  the top k, as `model.recommend()` does by default.
- Ids that aren't JSON numbers or strings are stored with `str()`.
- `TwoTower` trained with user feature matrices can't be exported yet.
