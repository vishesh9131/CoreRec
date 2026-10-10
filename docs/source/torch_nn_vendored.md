# PyTorch Modules in CoreRec

The historical `corerec.torch_nn` and `corerec.torch_utils` trees are unavailable
in this release. Use public PyTorch APIs for layers and optimizers.

```python
import torch
from torch import nn

layer = nn.Linear(3, 2)
output = layer(torch.zeros(4, 3))
assert output.shape == (4, 2)
```

CoreRec exposes recommendation modules and a training wrapper through
`corerec.nn`. For example:

```python
from corerec.nn import Recommender
from corerec.nn.models import MatrixFactorization

model = Recommender(MatrixFactorization, {"dim": 8}, epochs=1,
                    batch_size=4, device="cpu")
model.fit([1, 1, 2, 2], [10, 20, 20, 30])
assert model.recommend(1, top_k=1) == [30]
```

Use [model persistence](user_guide/model_persistence.md) to save the wrapper.
Custom modules must implement its scoring contract and be supplied explicitly
when loading an artifact.
