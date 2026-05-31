# NCF Tutorial: Neural Collaborative Filtering

## Introduction

**NCF** replaces traditional matrix factorization's inner product with a neural network to learn complex user-item interactions.

**Paper**: He et al. 2017 - Neural Collaborative Filtering

## How NCF Works

### Architecture

**Neural Generalization of Matrix Factorization:**

1. **Embedding Layer**: Separate embeddings for GMF and MLP paths
2. **GMF Path** (Generalized Matrix Factorization):
   - Element-wise product of user and item embeddings
   - Mimics MF but with learnable weights
3. **MLP Path**: Multi-layer neural network
   - Concatenates user and item embeddings
   - Learns non-linear interactions
4. **NeuMF Layer**: Combines GMF and MLP outputs

**Key Innovation**: Replaces dot product with flexible neural architecture

**Architecture:**
```
User → [GMF Embedding] → Element-wise ×                                          → Concat → Dense → Prediction  
Item → [GMF Embedding] → Element-wise × /

User → [MLP Embedding] →                           Concat → MLP Layers → /
Item → [MLP Embedding] → /
```

### Mathematical Foundation

**GMF Component:**
```
φ_GMF(u,i) = a_out(h^T (p_u ⊙ q_i))
```
where ⊙ is element-wise product

**MLP Component:**
```
z_1 = [p_u; q_i]
z_{l+1} = σ(W_l^T z_l + b_l)
φ_MLP(u,i) = a_out(h^T z_L)
```

**NeuMF (Combined):**
```
ŷ_ui = σ(h^T [φ_GMF(u,i); φ_MLP(u,that)])
```
where ; denotes concatenation

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.collaborative import NCF
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# NCF expects user_id, item_id, rating columns
ncf_train = train_df.rename(columns={'movie_id': 'item_id'}).copy()
ncf_test = test_df.rename(columns={'movie_id': 'item_id'}).copy()

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize Model

```python
model = NCF(
    model_type="NeuMF",
    gmf_embedding_dim=32,
    mlp_embedding_dim=32,
    mlp_hidden_layers=(64, 32, 16),
    num_epochs=5,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)
```

### Step 3: Train

```python
model.fit(ncf_train)

print("Training complete!")
```

### Step 4: Predict

```python
sample_user = ncf_train['user_id'].iloc[0]
sample_item = ncf_train['item_id'].iloc[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")
```

### Step 5: Recommend

```python
recommendations = model.recommend(user_id=sample_user, top_n=10)

print(f"Top-10 recommendations for User {sample_user}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

test_users = ncf_test['user_id'].values[:100]
test_items = ncf_test['item_id'].values[:100]
test_ratings = ncf_test['rating'].values[:100]
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Sample Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('ncf_model.pkl')

loaded = NCF.load('ncf_model.pkl')
recs = loaded.recommend(sample_user, top_n=5)
print(f"Loaded model recommendations: {recs}")
```

## Key Takeaways

### When to Use NCF

✅ **Excellent For:**
- Implicit feedback data (clicks, views, purchases)
- Collaborative filtering with non-linear patterns
- When MF is too simple but you want interpretability
- Medium-scale datasets (100K-10M interactions)
- Rating prediction and top-N recommendation

❌ **Not Ideal For:**
- Very sparse data (MF works better)
- Need to incorporate features → use Deep models
- Sequential patterns → use RNN/Transformer
- Very large scale (>50M interactions) → use simpler MF for speed

### Best Practices

1. **Embedding Dimension**: 8-64 typically (smaller than image/NLP)
2. **MLP Layers**: [64, 32, 16, 8] works well (decreasing)
3. **Pre-training**: Pre-train GMF and MLP separately, then combine
4. ** Negative Sampling**: 4-10 negatives per positive
5. **Regularization**: L2 on embeddings (1e-6 to 1e-4)
6. **Activation**: ReLU or LeakyReLU in MLP
7. **Loss**: Binary cross-entropy for implicit feedback
8. **Learning Rate**: 0.001 often sufficient

## Further Reading

- Paper: He et al. 2017 - Neural Collaborative Filtering
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
