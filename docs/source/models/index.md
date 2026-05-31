# Model Documentation

Welcome to the CoreRec model documentation. This section provides detailed information about all available recommendation models.

(model-tiers)=
## Model Tiers

CoreRec organizes its models into two tiers:

(production-models-tested--stable)=
### Production Models (Tested & Stable)

These **14 models** are **fully tested**, CI-enforced, and recommended for production use.
They implement the full `BaseRecommender` interface (`fit`, `predict`, `recommend`, `save`, `load`) and are covered by automated tests on every commit.

| Model | Type | Description |
|-------|------|-------------|
| **DCN** | Deep Learning | Deep & Cross Network for explicit feature interactions |
| **DeepFM** | Deep Learning | Factorization Machines + Deep Neural Networks |
| **GNNRec** | Graph | Graph Neural Networks for recommendations |
| **MIND** | Multi-Interest | Multi-Interest Network with dynamic routing |
| **NASRec** | AutoML | Neural Architecture Search for recommendations |
| **SASRec** | Sequential | Self-Attentive Sequential Recommendation |
| **TwoTower** | Retrieval | Dual-encoder for large-scale candidate retrieval |
| **BERT4Rec** | Sequential | Bidirectional self-attention for sequence modeling |
| **SAR** | Collaborative | Smart Adaptive Recommendations (co-occurrence) |
| **NCF** | Collaborative | Neural Collaborative Filtering (MLP + GMF) |
| **FAST** | Collaborative | Fast Approximate Similarity-based filtering |
| **FASTRecommender** | Collaborative | Extended FAST with additional features |
| **LightGCN** | Graph | Simplified graph convolution for CF |
| **TFIDFRecommender** | Content-Based | TF-IDF content similarity recommendations |

```{admonition} Production Guarantee
:class: tip
All production models pass automated tests on every push/PR. If a production model test fails, the commit is blocked from merging.
```

(sandbox-models-experimental)=
### Sandbox Models (Experimental)

CoreRec also includes **~50 sandbox models** in `corerec/sandbox/`. These are experimental implementations of well-known recommendation algorithms intended for **research, learning, and prototyping**.

```{admonition} Sandbox Warning
:class: warning
Sandbox models are **not production-tested**. They may have incomplete implementations, missing methods, or untested edge cases. Use them for exploration and research — not for production deployments without thorough validation.
```

Sandbox models cover a wide range of architectures:

**Deep Learning** — AFM, AutoInt, BST, BiVAE, Caser, DeepCrossing, DeepRec, DIEN, DiFM, DIN, DLRM, ENSFM, ESCM2, ESMM, FGCNN, FFM, FiBiNet, FLEN, FM, GAN-Rec, GateNet, GRU-CF, NFM, NextItNet, Wide&Deep, YouTubeDNN, PNN, MMoE, PLE, Monolith, TDM, AutoFI

**Matrix Factorization** — ALS, SVD, A2SVD, MF-Base, FM-Base, UserBased CF, Matrix Factorization

**Graph-Based** — GeoIMC, LightGCN-Base, GNN-Base

**Sequential** — RBM, RLRMC, SLi-Rec, SUM

**Bayesian** — BPR, BPR-MF, VMF

**Content-Based** — MIND-Content

## Model Categories

### [Deep Learning Models](deep_learning.md)
Neural network-based recommendation models that learn complex patterns from user-item interactions.

**Production**: DCN, DeepFM, GNNRec, MIND, NASRec, SASRec, TwoTower, BERT4Rec
**Sandbox**: AFM, AutoInt, BST, BiVAE, Caser, and 25+ more

### [Matrix Factorization](matrix_factorization.md)
Classic collaborative filtering approaches using matrix decomposition.

**Production**: SAR, NCF, FAST, FASTRecommender
**Sandbox**: ALS, SVD, A2SVD, MF-Base, and more

### [Graph-Based Models](graph_based.md)
Models that leverage graph structure and relationships.

**Production**: GNNRec, LightGCN
**Sandbox**: GeoIMC, LightGCN-Base, GNN-Base

### [Sequential Models](sequential.md)
Time-aware models that consider user behavior sequences.

**Production**: SASRec, BERT4Rec
**Sandbox**: RBM, RLRMC, SLi-Rec, SUM

### [Bayesian Models](bayesian.md)
Probabilistic approaches to recommendation.

**Sandbox only**: BPR, BPR-MF, VMF

### [Content-Based Models](content_based.md)
Feature-based recommendations using item/user attributes.

**Production**: TFIDFRecommender
**Sandbox**: MIND-Content

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
