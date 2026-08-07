# Unionized Filter Engine

The Unionized Filter Engine provides comprehensive collaborative filtering algorithms that learn from user-item interaction patterns.

## Overview

Collaborative filtering is based on the idea that users who agreed in the past will agree in the future. The Unionized Filter Engine implements over 50 state-of-the-art collaborative filtering algorithms organized into seven categories.

## Algorithm Categories

### 1. Matrix Factorization

Decompose the user-item interaction matrix into latent factors.

**Available Algorithms:**

- **SVD** (Singular Value Decomposition)
- **ALS** (Alternating Least Squares)
- **NMF** (Non-negative Matrix Factorization)
- **PMF** (Probabilistic Matrix Factorization)
- **WNMF** (Weighted NMF)
- **SVD++** (Extended SVD)
- **RSVD** (Regularized SVD)

[**→ Matrix Factorization Documentation**](matrix-factorization.md)

### 2. Neural Network Based

Deep learning approaches to collaborative filtering.

**Available Algorithms:**

- **NCF** (Neural Collaborative Filtering)
- **DeepFM** (Deep Factorization Machines)
- **AutoInt** (Automatic Feature Interaction)
- **DCN** (Deep & Cross Network)
- **AFM** (Attentional Factorization Machines)
- **DIN** (Deep Interest Network)
- **DIEN** (Deep Interest Evolution Network)
- **NFM** (Neural Factorization Machines)
- **Wide & Deep**
- **PNN** (Product-based Neural Networks)

[**→ Neural Network Documentation**](neural-network.md)

### 3. Graph-Based

Leverage graph structure in recommendation.

**Available Algorithms:**

- **LightGCN** (Light Graph Convolutional Network)
- **DeepWalk** (Random walk embeddings)
- **GNN** (Graph Neural Networks)
- **GeoimC** (Geometric Matrix Completion)
- **Edge-Aware Filtering**
- **Multi-Relational GNN**
- **Heterogeneous Network Embedding**

[**→ Graph-Based Documentation**](graph-based.md)

### 4. Attention Mechanisms

Attention-based collaborative filtering.

**Available Algorithms:**

- **SASRec** (Self-Attentive Sequential Recommendation)
- **Transformer-based Recommenders**
- **A2SVD** (Attentive Collaborative Filtering)
- **Attention-based Sequential Models**

[**→ Attention Mechanisms Documentation**](attention-mechanisms.md)

### 5. Bayesian Methods

Probabilistic approaches to recommendation.

**Available Algorithms:**

- **BPR** (Bayesian Personalized Ranking)
- **BPRMF** (BPR Matrix Factorization)
- **Bayesian MF** (Bayesian Matrix Factorization)
- **VMF** (von Mises-Fisher)
- **Multinomial VAE**
- **Probabilistic Graphical Models**

[**→ Bayesian Methods Documentation**](bayesian-methods.md)

### 6. Sequential Models

Time-aware and sequence-aware recommendations.

**Available Algorithms:**

- **LSTM-based Recommenders**
- **GRU-based Recommenders**
- **Caser** (Convolutional Sequence Embedding)
- **NextItNet** (Next Item Network)
- **DIEN** (Deep Interest Evolution)

[**→ Sequential Models Documentation**](sequential-models.md)

### 7. Variational Encoders

Generative models for recommendations.

**Available Algorithms:**

- **VAE** (Variational Autoencoder)
- **CVAE** (Conditional VAE)
- **Beta-VAE**
- **Mult-VAE** (Multinomial VAE)
- **RecVAE**

[**→ Variational Encoders Documentation**](variational-encoders.md)

## Quick Start

### Example: Matrix Factorization

```python
from corerec.engines.unionizedFilterEngine.mf_base.SVD_base import SVD

# Initialize SVD model
model = SVD(
    n_factors=50,
    n_epochs=20,
    learning_rate=0.01,
    regularization=0.02
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=123, top_k=10)
print(f"Top 10 recommendations: {recommendations}")

# Predict rating
score = model.predict(user_id=123, item_id=456)
print(f"Predicted rating: {score:.2f}")
```

### Example: Neural Collaborative Filtering

```python
from corerec.engines.unionizedFilterEngine.nn_base.NCF_base import NCF

# Initialize NCF model
model = NCF(
    embedding_dim=64,
    layers=[128, 64, 32, 16],
    dropout=0.2,
    epochs=20,
    batch_size=256
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=123, top_k=10)
```

### Example: Graph-Based (LightGCN)

```python
from corerec.engines.unionizedFilterEngine.graph_based_base.lightgcn import LightGCN

# Initialize LightGCN model
model = LightGCN(
    embedding_dim=64,
    num_layers=3,
    epochs=100,
    learning_rate=0.001
)

# Train model
model.fit(user_ids, item_ids, ratings)

# Get recommendations
recommendations = model.recommend(user_id=123, top_k=10)
```

## Special Features

### Fast Recommender

CoreRec provides a FastAI-style fast recommender for quick prototyping:

```python
from corerec.engines.unionizedFilterEngine.fast import FastRecommender

model = FastRecommender(
    n_factors=50,
    n_epochs=20,
    learning_rate=0.01
)

model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=123, top_k=10)
```

### SAR (Smart Adaptive Recommendations)

Microsoft's SAR algorithm for item-to-item similarity:

```python
from corerec.engines.unionizedFilterEngine.sar import SAR

model = SAR(
    similarity_type='jaccard',
    time_decay_coefficient=30,
    timedecay_formula=True
)

model.fit(user_ids, item_ids, ratings, timestamps)
recs = model.recommend(user_id=123, top_k=10)
```

### RBM (Restricted Boltzmann Machine)

Energy-based collaborative filtering:

```python
from corerec.engines.unionizedFilterEngine.rbm import RBM

model = RBM(
    n_hidden=100,
    n_epochs=30,
    batch_size=10
)

model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=123, top_k=10)
```

### RLRMC (Riemannian Low-Rank Matrix Completion)

Geometric approach to matrix completion:

```python
from corerec.engines.unionizedFilterEngine.rlrmc import RLRMC

model = RLRMC(
    rank=20,
    max_iter=100,
    tol=1e-4
)

model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=123, top_k=10)
```

### GeoMLC (Geometric Matrix Learning and Completion)

```python
from corerec.engines.unionizedFilterEngine.geomlc import GeoMLC

model = GeoMLC(
    embedding_dim=50,
    num_epochs=50
)

model.fit(user_ids, item_ids, ratings)
recs = model.recommend(user_id=123, top_k=10)
```

## Factory Pattern

Use the factory to create models from configuration:

```python
from corerec.engines.unionizedFilterEngine.cr_unionizedFactory import UnionizedRecommenderFactory

config = {
    'method': 'matrix_factorization',
    'params': {
        'n_factors': 50,
        'n_epochs': 20,
        'learning_rate': 0.01
    }
}

model = UnionizedRecommenderFactory.get_recommender(config)
model.fit(user_ids, item_ids, ratings)
```

## When to Use Unionized Filter Engine

✅ **Use when:**
- You have user-item interaction data (ratings, clicks, purchases)
- You want to find patterns in user behavior
- You need collaborative filtering
- Cold start is not a major issue
- You have sufficient interaction history

❌ **Avoid when:**
- You only have item features (use Content Filter)
- You have severe cold start problems
- You need explainable recommendations
- Your data is purely content-based

## Performance Tips

1. **Choose the right algorithm:**
   - Small data: SVD, ALS
   - Medium data: NCF, DeepFM
   - Large data: LightGCN, GNN
   - Sequential: SASRec, LSTM
   - Sparse data: VAE, RBM

2. **Optimize hyperparameters:**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_factors': [20, 50, 100],
       'learning_rate': [0.001, 0.01, 0.1],
       'regularization': [0.01, 0.02, 0.05]
   }
   
   # Use grid search to find best params
   ```

3. **Use GPU for large models:**
   ```python
   model = NCF(device='cuda')
   ```

4. **Batch predictions:**
   ```python
   # More efficient than individual predictions
   recs = model.batch_recommend(user_ids, top_k=10)
   ```

## Algorithm Comparison

| Algorithm | Training Speed | Accuracy | Scalability | Best For |
|-----------|---------------|----------|-------------|----------|
| SVD | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | General purpose |
| ALS | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Implicit feedback |
| NCF | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Deep learning |
| LightGCN | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Graph structure |
| SASRec | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Sequential data |
| BPR | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Implicit feedback |
| VAE | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Sparse data |

## See Also

- [Content Filter Engine](../content-filter/index.md) - For feature-based recommendations
- [Deep Learning Models](../deep-learning/index.md) - For large-scale deep learning
- [Examples](../../examples/index.md) - Usage examples
- [API Reference](../../api/index.md) - Detailed API documentation


