# Complete Model Index

Alphabetical reference for all documented CoreRec models. See [Model Tiers](index.md#model-tiers) for production vs sandbox policy.

## Production models (16)

The list is `corerec.engines.MODELS`; `corerec models` prints it.

| Model | Category | Import | Tutorial |
|-------|----------|--------|----------|
| ALS | Classic CF | `from corerec.engines import ALS` | - |
| DCN | Ranking | `from corerec.engines import DCN` | [Tutorial](../tutorials/dcn_tutorial.md) |
| DeepFM | Ranking | `from corerec.engines import DeepFM` | [Tutorial](../tutorials/deepfm_tutorial.md) |
| EASE | Classic CF | `from corerec.engines import EASE` | - |
| HSTU | Generative | `from corerec.engines import HSTU` | [Sequential models](sequential.md) |
| Item2Vec | Classic CF | `from corerec.engines import Item2Vec` | - |
| ItemKNN | Classic CF | `from corerec.engines import ItemKNN` | - |
| LightGCN | Graph | `from corerec.engines import LightGCN` | [Tutorial](../tutorials/lightgcn_tutorial.md) |
| MultiDAE | Autoencoder | `from corerec.engines import MultiDAE` | - |
| MultVAE | Autoencoder | `from corerec.engines import MultVAE` | - |
| SAR | Classic CF | `from corerec.engines import SAR` | [Tutorial](../tutorials/sar_tutorial.md) |
| SASRec | Sequential | `from corerec.engines import SASRec` | [Tutorial](../tutorials/sasrec_tutorial.md) |
| SLIM | Classic CF | `from corerec.engines import SLIM` | - |
| TFIDFRecommender | Content | `from corerec.engines import TFIDFRecommender` | [Tutorial](../tutorials/tfidf_tutorial.md) |
| TwoTower | Retrieval | `from corerec.engines import TwoTower` | [Tutorial](../tutorials/two_tower_tutorial.md) |
| UserKNN | Classic CF | `from corerec.engines import UserKNN` | - |

All production models implement: `fit()`, `predict()`, `recommend(top_k=)`, `save()`, `load()` via `BaseRecommender`.

## Sandbox models (by category)

### Deep learning (sandbox)

AFM, AutoFI, AutoInt, BST, BiVAE, Caser, DCN-Base, DeepCrossing, DeepFM-Base, DeepRec, DIEN, DiFM, DIN, DLRM, ENSFM, ESCM2, ESMM, FGCNN, FFM, FiBiNet, FLEN, FM, GAN-Rec, GateNet, GRU-CF, Monolith, MMoE, NFM, NextItNet, PLE, PNN, TDM, Wide&Deep, YouTubeDNN

→ Details: [Deep Learning Models](deep_learning.md)

### Matrix factorization (sandbox)

A2SVD, ALS, FM-Base, Matrix Factorization, MF-Base, SVD, User-Based CF

→ Details: [Matrix Factorization](matrix_factorization.md)

### Graph (sandbox)

GeoIMC, GNN-Base, LightGCN-Base

→ Details: [Graph-Based Models](graph_based.md)

### Sequential (sandbox)

RBM, RLRMC, SLi-Rec, SUM, NextItNet, Caser

→ Details: [Sequential Models](sequential.md)

### Bayesian (sandbox)

BPR, BPR-MF, VMF

→ Details: [Bayesian Models](bayesian.md)

### Content (sandbox)

MIND-Content

→ Details: [Content-Based Models](content_based.md)

## Category guides

```{toctree}
:maxdepth: 1

deep_learning
matrix_factorization
graph_based
sequential
bayesian
content_based
```

## Unified API

```python
model.fit(...)                    # see model-specific tutorial
score = model.predict(user, item)
recs = model.recommend(user, top_k=10)
model.save("artifacts/my_model")  # safe bundle default
loaded = type(model).load("artifacts/my_model")
```

Persistence: {doc}`../user_guide/safe_bundle_persistence`.

## Tutorials

Hands-on walkthroughs: [Tutorial Index](../tutorials/index.md)

Pipeline & serving: [Pipeline Tutorial](../tutorials/pipeline_tutorial.md) · [Serving API](../api/serving.md)
