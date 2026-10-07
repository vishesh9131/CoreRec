# Sequential Models

Models that use ordered interaction history for next-item prediction.

## Production models (CI-tested)

| Model | Import | Tutorial |
|-------|--------|----------|
| **SASRec** | `from corerec.engines.sasrec import SASRec` | [SASRec](../tutorials/sasrec_tutorial.md) |
| **HSTU** (generative) | `from corerec.engines import HSTU` | [below](#hstu-generative-next-item-recommendation) |

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

## HSTU: generative next-item recommendation

`HSTU` is the encoder from Zhai et al., *Actions Speak Louder than Words:
Trillion-Parameter Sequential Transducers for Generative Recommendations*
(ICML 2024). It reads each user's history in time order and is trained like a
language model, predicting the next item at every position. Unlike SASRec it
gates attention with SiLU instead of softmax, multiplies the attention output
by a learned gate, and adds learned biases for position distance and, when
timestamps are given, for the time between two interactions.

It takes one row per interaction, like every other model, plus optional
timestamps:

```python
from corerec.engines import HSTU

model = HSTU(epochs=50)                      # paper recipe: 50-dim, 2 layers, 200-item history
model.fit(user_ids, item_ids, timestamps=timestamps)
model.recommend(user_id, top_k=10)           # history items are excluded by default
model.save("artifacts/hstu")
```

Training uses the paper's public ML-1M recipe: sampled softmax over 128 random
negatives, temperature 0.05, L2-normalised embeddings. One difference: the
negatives are shared by all positions of a sequence instead of drawn per
position. Every position still sees 128 uniform random negatives, only
correlated across positions, and it is about six times cheaper on a CPU. `HSTU(encoder="sasrec")` trains a SASRec encoder with the identical
recipe, for comparisons where only the architecture changes.

`corerec serve events.csv --model HSTU` passes the file's timestamp column to
the model automatically. The ML-1M comparison against SASRec is in
`BENCHMARKS.md` and reproduces with `Findings/bench/generative_bench.py`.

