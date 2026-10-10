# Bayesian Models and Pairwise Ranking

The historical sandbox BPR, BPR-MF, and VMF implementations are unavailable in
this release. CoreRec does not currently export a probabilistic Bayesian model.
See [removed models](../tutorials/removed_models.md) for the migration record.

BPR is also a pairwise ranking loss. You can train the supported PyTorch matrix
factorization module with this loss through `corerec.nn.Recommender`:

```python
from corerec.nn import Recommender
from corerec.nn.models import MatrixFactorization

model = Recommender(MatrixFactorization, {"dim": 8}, loss="bpr",
                    epochs=2, batch_size=4, device="cpu")
model.fit([1, 1, 2, 2, 3, 3], [10, 20, 20, 30, 10, 30])
assert model.recommend(1, top_k=1) == [30]
```

This model learns user and item embeddings with sampled negative items. It ranks
items; its scores are not posterior probabilities or uncertainty estimates.
Embedding storage grows with the number of users and items and the embedding
dimension. Increase negative samples only after measuring training cost.

## See also

- [Matrix factorization](matrix_factorization.md)
- [Model persistence](../user_guide/model_persistence.md)
