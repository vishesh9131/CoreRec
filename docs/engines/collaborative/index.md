# Unionized Filter Engine

The Unionized Filter Engine provides comprehensive collaborative filtering algorithms that learn from user-item interaction patterns.

## Overview

Collaborative filtering is based on the idea that users who agreed in the past will agree in the future. In CoreRec 0.7 the collaborative models all live in `corerec.engines` and share one API: `fit(user_ids, item_ids, ratings)`, `recommend(user_id, top_k)`, `predict`, `save`/`load`.

```python
import corerec.engines as engines

print(engines.list_models("classic"))      # ['ALS', 'SAR', 'ItemKNN', 'UserKNN', 'EASE', 'SLIM', 'Item2Vec']
print(engines.list_models("graph"))        # ['LightGCN']
print(engines.list_models("autoencoder"))  # ['MultVAE', 'MultiDAE']
```

(`corerec.engines.unionized` is an alias of `corerec.engines.collaborative`, kept for old imports.)

## Algorithm Categories

### 1. Matrix Factorization

Decompose the user-item interaction matrix into latent factors.

**Available Algorithms:**

- **ALS** (Alternating Least Squares, implicit feedback)
- **Item2Vec** (skip-gram embeddings over interaction sequences)

[**→ Matrix Factorization Documentation**](matrix-factorization.md)

### 2. Neural Network Based

Deep learning approaches to collaborative filtering.

**Available Algorithms:**

- **DeepFM** (Deep Factorization Machines)
- **DCN** (Deep & Cross Network)
- **TwoTower** (retrieval)

NCF, AutoInt, AFM, DIN, DIEN, NFM, PNN, Wide&Deep, xDeepFM and FiBiNet were
experimental sandbox models and were removed in 0.7.0. To try one of those
architectures, write it as a PyTorch module and wrap it in
`corerec.nn.Recommender`.

### 3. Graph-Based

Leverage graph structure in recommendation.

**Available Algorithms:**

- **LightGCN** (Light Graph Convolutional Network)

NGCF and GNNRec were removed in 0.7.0; LightGCN is the graph model that ships.

[**→ Graph-Based Documentation**](graph-based.md)

### 4. Attention Mechanisms

Attention-based collaborative filtering.

**Available Algorithms:**

- **SASRec** (Self-Attentive Sequential Recommendation)
- **HSTU** (generative next-item transducer)

BERT4Rec was removed in 0.7.0.

### 5. Bayesian Methods

Probabilistic approaches to recommendation.

**Available Algorithms:**

- **MultVAE** (multinomial variational autoencoder)

Bayesian MF and the probabilistic graphical models were removed in 0.7.0.

### 6. Sequential Models

Time-aware and sequence-aware recommendations.

**Available Algorithms:**

- **SASRec**, **HSTU** (see Attention Mechanisms)
- **SAR** with time decay (`timedecay_formula=True`)

The LSTM/GRU recommenders and Caser were removed in 0.7.0.

### 7. Variational Encoders

Generative models for recommendations.

**Available Algorithms:**

- **MultVAE** (Multinomial VAE)
- **MultiDAE** (Denoising autoencoder)

[**→ Variational Encoders Documentation**](variational-encoders.md)

## Quick Start

All examples below use this toy data:

```python
import numpy as np

rng = np.random.default_rng(0)
user_ids = rng.integers(0, 200, 5000).tolist()
item_ids = rng.integers(0, 500, 5000).tolist()
ratings = rng.integers(1, 6, 5000).astype(float).tolist()
timestamps = np.sort(rng.integers(1_700_000_000, 1_710_000_000, 5000)).tolist()
```

### Example: Matrix Factorization

```python
from corerec.engines import ALS

# Initialize ALS model
model = ALS(
    factors=50,
    iterations=20,
    reg=10.0,
    alpha=1.0
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=user_ids[0], top_k=10)
print(f"Top 10 recommendations: {recommendations}")

# Preference score (implicit feedback: not a 1-5 rating)
score = model.predict(user_id=user_ids[0], item_id=item_ids[0])
print(f"Score: {score:.3f}")
```

### Example: Neighbourhood Models

ItemKNN, UserKNN and EASE are strong baselines and train in seconds:

```python
from corerec.engines import EASE, ItemKNN

knn = ItemKNN(top_k_neighbors=100, shrink=10.0)
knn.fit(user_ids, item_ids, ratings)

ease = EASE(reg=250.0)
ease.fit(user_ids, item_ids, ratings)

recommendations = ease.recommend(user_id=user_ids[0], top_k=10)
```

### Example: Graph-Based (LightGCN)

```python
from corerec.engines import LightGCN

# Initialize LightGCN model
model = LightGCN(
    n_factors=64,
    n_layers=3,
    epochs=20,
    learning_rate=0.001
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=user_ids[0], top_k=10)
```

## Special Features

### Fast Recommender

`FastRecommender` was removed in 0.7.0. For quick prototyping use `EASE` or
`ItemKNN` (seconds to train, no tuning), or `corerec train` from the command
line.

### SAR (Smart Adaptive Recommendations)

Microsoft's SAR algorithm for item-to-item similarity:

```python
from corerec.engines import SAR

model = SAR(
    similarity_type='jaccard',
    time_decay_coefficient=30,
    timedecay_formula=True
)

model.fit(user_ids, item_ids, ratings, timestamps=timestamps)
recs = model.recommend(user_id=user_ids[0], top_k=10)
```

SAR also takes a DataFrame: `model.fit(df)` with columns `userID`, `itemID`,
`rating` (and `timestamp`), or the names set by `col_user`, `col_item`, ...

### RBM (Restricted Boltzmann Machine)

Removed in 0.7.0. `MultiDAE` is the closest model that ships.

### RLRMC (Riemannian Low-Rank Matrix Completion)

Removed in 0.7.0. Use `ALS`.

### GeoMLC (Geometric Matrix Learning and Completion)

Removed in 0.7.0. Use `ALS`.

## Factory Pattern

Create models from configuration by name; every registered model takes its
hyperparameters as keyword arguments:

```python
import corerec.engines as engines

config = {
    'method': 'ALS',
    'params': {
        'factors': 50,
        'iterations': 20,
        'reg': 10.0
    }
}

model = getattr(engines, config['method'])(**config['params'])
model.fit(user_ids, item_ids, ratings)
```

## When to Use Unionized Filter Engine

**Use when:**
- You have user-item interaction data (ratings, clicks, purchases)
- You want to find patterns in user behavior
- You need collaborative filtering
- Cold start is not a major issue
- You have sufficient interaction history

**Avoid when:**
- You only have item features (use Content Filter)
- You have severe cold start problems
- You need explainable recommendations
- Your data is purely content-based

## Performance Tips

1. **Choose the right algorithm:**
   - Small data: ALS, ItemKNN, EASE
   - Medium data: LightGCN, MultVAE
   - Large data: ALS, ItemKNN (blocked neighbour search), TwoTower
   - Sequential: SASRec, HSTU
   - Sparse data: MultVAE, SLIM

2. **Optimize hyperparameters:**
   ```python
   from corerec.engines import ALS
   from corerec.evaluation import evaluate

   train = list(zip(user_ids[:4000], item_ids[:4000]))
   test = list(zip(user_ids[4000:], item_ids[4000:]))

   best = None
   for factors in (20, 50, 100):
       for reg in (1.0, 10.0):
           m = ALS(factors=factors, reg=reg).fit(user_ids[:4000], item_ids[:4000])
           ndcg = evaluate(m, test, train_interactions=train, k=10)["NDCG@10"]
           if best is None or ndcg > best[0]:
               best = (ndcg, factors, reg)
   print(best)
   ```

3. **Use GPU for large models:**
   ```python
   from corerec.engines import LightGCN

   gcn = LightGCN(device='auto')  # CUDA or Apple MPS when available, else CPU
   ```

4. **Batch predictions:**
   ```python
   # More efficient than individual predictions
   recs = ease.batch_recommend(user_ids[:10], top_k=10)  # {user: [items]}
   ```

## Algorithm Comparison

| Algorithm | Training Speed | Scalability | Best For |
|-----------|---------------|-------------|----------|
| ALS | fast | high | Implicit feedback, general purpose |
| ItemKNN / EASE | fast | high (EASE inverts an items x items matrix) | Strong baselines |
| SLIM | medium | medium | Sparse data |
| SAR | fast | high | Item-to-item similarity, time decay |
| LightGCN | medium | medium | Graph structure |
| MultVAE / MultiDAE | medium | medium | Sparse implicit feedback |
| SASRec / HSTU | slow | medium | Sequential data |

Measured numbers on MovieLens are in `BENCHMARKS.md`.

## See Also

- [Deep Learning Models](../deep-learning/index.md) - For large-scale deep learning
- [Examples](../../examples/index.md) - Usage examples
- [API Reference](../../api/index.md) - Detailed API documentation
