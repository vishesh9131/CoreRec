# CoreRec: Deep Learning Recommendation Systems Framework

Welcome to CoreRec's comprehensive documentation! CoreRec is a production-ready, professional framework for building state-of-the-art recommendation systems using deep learning and traditional collaborative filtering techniques.

```{toctree}
---
maxdepth: 2
caption: Getting Started
---
installation
quickstart
concepts
```

```{toctree}
---
maxdepth: 2
caption: Model Documentation
---
models/index
models/deep_learning
models/matrix_factorization
models/graph_based
models/sequential
models/bayesian
models/models_index
```

```{toctree}
---
maxdepth: 2
caption: Tutorials & Learning
---
tutorials/index
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---
api/base_recommender
api/exceptions
api/mixins
api/engines
```

```{toctree}
---
maxdepth: 2
caption: Examples
---
examples/basic_usage
examples/advanced_usage
examples/production_deployment
```

```{toctree}
---
maxdepth: 1
caption: Development
---
contributing
changelog
license
```

## Quick Example

```python
from corerec.engines.dcn import DCN
import cr_learn

# Load dataset
data = cr_learn.load_dataset('movielens-100k')

# Initialize model
model = DCN(
    embedding_dim=64,
    epochs=20,
    verbose=True
)

# Train
model.fit(
    user_ids=data.user_ids,
    item_ids=data.item_ids,
    ratings=data.ratings
)

# Predict
score = model.predict(user_id=1, item_id=100)

# Recommend
recommendations = model.recommend(user_id=1, top_k=10)

# Save model
model.save('dcn_model.pkl')
```

## Features

- ✅ **57+ Production-Ready Models**
- ✅ **Unified API** across all models
- ✅ **Professional Exception Handling**
- ✅ **Complete Persistence** (save/load)
- ✅ **Type Hints** throughout
- ✅ **Comprehensive Documentation**
- ✅ **Tutorial Examples** with `cr_learn`

## Model Categories

### Deep Learning Models (29 models)
Neural network-based recommendation models including DCN, DeepFM, DIEN, DIN, DLRM, and more.

### Matrix Factorization (9 models)
Classic collaborative filtering with SVD, ALS, and advanced variants.

### Graph-Based Models (6 models)
Graph neural networks for recommendations: GNN, LightGCN, and more.

### Sequential Models (6 models)
Time-aware recommendations: SASRec, MIND, RBM, SAR.

### Bayesian Models (3 models)
Probabilistic approaches: BPR, BPRMF, VMF.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
