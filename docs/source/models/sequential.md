# Sequential Models

Models that use ordered interaction history for next-item prediction.

## Production models (CI-tested)

| Model | Import | Tutorial |
|-------|--------|----------|
| **SASRec** | `from corerec.engines.sasrec import SASRec` | [SASRec](../tutorials/sasrec_tutorial.md) |
| **BERT4Rec** | `from corerec.engines.bert4rec import BERT4Rec` | [BERT4Rec](../tutorials/bert4rec_tutorial.md) |
| **MIND** | `from corerec.engines.mind import MIND` | [MIND](../tutorials/mind_tutorial.md) |

```{admonition} Not sequential
:class: note
**SAR** (Smart Adaptive Recommendations) is **item-similarity collaborative filtering**, not a sequential model. See [Matrix Factorization](matrix_factorization.md).
```

### SASRec / BERT4Rec (interaction matrix)

Sequential production models require a user×item **interaction matrix** (not raw rating triplets alone):

```python
from corerec.engines.sasrec import SASRec
import numpy as np

user_list = sorted(train_df["user_id"].unique())
item_list = sorted(train_df["movie_id"].unique())
# build train_mat[user_idx, item_idx] = 1.0 for observed interactions

model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    max_seq_length=50,
    num_epochs=10,
    batch_size=256,
    verbose=True,
)
model.fit(user_list, item_list, train_mat)
recs = model.recommend(user_id=1, top_k=10)
```

### MIND (triplet API)

MIND uses `(user_ids, item_ids, ratings)` triplets — see the [MIND tutorial](../tutorials/mind_tutorial.md).

## Sandbox models (experimental)

| Model | Import | Tutorial |
|-------|--------|----------|
| RBM | `corerec.sandbox.collaborative_full.rbm` | [RBM](../tutorials/rbm_tutorial.md) |
| RLRMC | sandbox sequential | [RLRMC](../tutorials/rlrmc_tutorial.md) |
| SLi-Rec | `corerec.sandbox.collaborative_full.sli` | [SLiRec](../tutorials/slirec_tutorial.md) |
| SUM | `corerec.sandbox.collaborative_full.sum` | [SUM](../tutorials/sum_tutorial.md) |
| NextItNet | sandbox sequential_model_base | [NextItNet](../tutorials/nextitnet_tutorial.md) |
| Caser | sandbox nn_base | [Caser](../tutorials/caser_tutorial.md) |

## When to use

- Session-based or next-item prediction
- User behavior is strongly order-dependent
- E-commerce / streaming click sequences

## See also

- [Deep learning models](deep_learning.md)
- [Tutorials](../tutorials/index.md)
