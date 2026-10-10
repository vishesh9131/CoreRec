# Model Documentation

Welcome to the CoreRec model documentation. This section provides detailed information about all available recommendation models.

(model-tiers)=
## Model Tiers

CoreRec organizes its models into two tiers:

(production-models-tested--stable)=
### Production Models (Tested & Stable)

Use `corerec.engines.MODELS` to inspect the installed model registry. These
models expose `fit`, `predict`, `recommend`, `save`, and `load`; production
contract tests run in CI on each push and pull request.

| Models | Use |
|--------|-----|
| ItemKNN, UserKNN, EASE, SLIM, SAR | Classic collaborative filtering |
| ALS, Item2Vec | Embedding-based collaborative filtering |
| DCN, DeepFM | Neural ranking |
| TwoTower | Dual-encoder retrieval |
| LightGCN | Graph-based collaborative filtering |
| SASRec, HSTU | Sequential recommendation |
| MultVAE, MultiDAE | Autoencoder-based collaborative filtering |
| TFIDFRecommender | Item text similarity |

(sandbox-models-experimental)=
### Experimental Components

Optional towers and tracking integrations live under `corerec.experimental`.
They are not substitutes for registered model engines. Historical sandbox
models and their imports are unavailable in this release; see
[removed models](../tutorials/removed_models.md).

## Model Categories

- [Deep learning](deep_learning.md)
- [Matrix factorization](matrix_factorization.md)
- [Graph-based models](graph_based.md)
- [Sequential models](sequential.md)
- [Bayesian models and pairwise ranking](bayesian.md)
- [Content-based models](content_based.md)

## Model Selection Guide

### When to Use Deep Learning Models
- Large datasets with complex patterns
- Rich feature sets available
- Need to capture non-linear relationships

### When to Use Matrix Factorization
- Sparse user-item matrices
- Need interpretable recommendations
- Fast training and inference required

### When to Use Graph-Based Models
- Rich relationship data available
- Social networks or knowledge graphs
- Need to leverage item/item or user/user connections

### When to Use Sequential Models
- Temporal patterns important
- User behavior sequences available
- Need next-item prediction

### When to Use Bayesian Models
- Uncertainty quantification needed
- Probabilistic interpretation desired
- Cold start problems

## Getting Started

1. **Start with production models** — they're tested and guaranteed to work
2. Browse the [Complete Model Index](models_index.md) for imports and tutorial links
3. Read the specific model tutorial for code examples
4. Only explore sandbox models if you need a specific algorithm not covered by production models
5. If using a sandbox model in production, validate thoroughly and contribute tests back

## Tutorials

For hands-on examples, see the [Tutorials](../tutorials/index.md) section.
