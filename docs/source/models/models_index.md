# Complete Model Index

Alphabetical reference for all documented CoreRec models. See [Model Tiers](index.md#model-tiers) for production vs sandbox policy.

## Production models (14)

| Model | Category | Import | Tutorial |
|-------|----------|--------|----------|
| BERT4Rec | Sequential | `from corerec.engines.bert4rec import BERT4Rec` | [Tutorial](../tutorials/bert4rec_tutorial.md) |
| DCN | Deep Learning | `from corerec.engines.dcn import DCN` | [Tutorial](../tutorials/dcn_tutorial.md) |
| DeepFM | Deep Learning | `from corerec.engines.deepfm import DeepFM` | [Tutorial](../tutorials/deepfm_tutorial.md) |
| FAST | Collaborative | `from corerec.engines.collaborative import FAST` | [Tutorial](../tutorials/fast_tutorial.md) |
| FASTRecommender | Collaborative | `from corerec.engines.collaborative import FASTRecommender` | [Tutorial](../tutorials/fast_recommender_tutorial.md) |
| GNNRec | Graph | `from corerec.engines.gnnrec import GNNRec` | [Tutorial](../tutorials/gnnrec_tutorial.md) |
| LightGCN | Graph | `from corerec.engines.collaborative import LightGCN` | [Tutorial](../tutorials/lightgcn_tutorial.md) |
| MIND | Sequential / Multi-interest | `from corerec.engines.mind import MIND` | [Tutorial](../tutorials/mind_tutorial.md) |
| NASRec | Deep Learning | `from corerec.engines.nasrec import NASRec` | [Tutorial](../tutorials/nasrec_tutorial.md) |
| NCF | Collaborative | `from corerec.engines.collaborative import NCF` | [Tutorial](../tutorials/ncf_tutorial.md) |
| SAR | Collaborative | `from corerec.engines.collaborative import SAR` | [Tutorial](../tutorials/sar_tutorial.md) |
| SASRec | Sequential | `from corerec.engines.sasrec import SASRec` | [Tutorial](../tutorials/sasrec_tutorial.md) |
| TFIDFRecommender | Content | `from corerec.engines.content_based import TFIDFRecommender` | [Tutorial](../tutorials/tfidf_tutorial.md) |
| TwoTower | Deep Learning / Retrieval | `from corerec.engines.two_tower import TwoTower` | [Tutorial](../tutorials/two_tower_tutorial.md) |

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
