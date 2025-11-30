# AutoInt Tutorial: Automatic Feature Interaction Learning via Self-Attention

## Introduction

**AutoInt** uses multi-head self-attention to automatically learn high-order feature interactions without manual feature engineering.

**Paper**: Song et al. 2019 - AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

## How AutoInt Works

### Architecture

**Self-Attention for Feature Interactions:**

1. **Embedding Layer**: Convert sparse features to dense embeddings
2. **Interacting Layer**: Multi-head self-attention on feature embeddings
   - Each head learns different interaction patterns
   - Residual connections preserve original features
3. **Stacking**: Multiple attention layers for higher-order interactions
4. **Output Layer**: Combine all layers for prediction

**Key Innovation**: Treats features as a sequence, applies self-attention

**Architecture:**
```
Features → Embed → [Self-Attention + Residual]×L → Concat → Output
```

### Mathematical Foundation

**Multi-Head Self-Attention:**
```
head_h = Attention(E·W_h^Q, E·W_h^K, E·W_h^V)
where E = [e_1, e_2, ..., e_m]  # feature embeddings
```

**Attention Mechanism:**
```
Attention(Q,K,V) = softmax((Q·K^T)/√d) · V
```

**Multi-Head Output:**
```
MultiHead(E) = [head_1; head_2; ...; head_H] × W^O
```

**Residual Connection:**
```
E' = ReLU(MultiHead(E) + E)
```

**Final Prediction:**
```
ŷ = σ(w^T · [E^(0); E^(1); ...; E^(L)])
```
where E^(l) is output of layer l

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.autoint import AutoInt
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset with CORRECT API
data = ml_1m.load()  # Returns dict with 'ratings', 'users', 'movies'
ratings_df = data['ratings']  # DataFrame with user_id, movie_id, rating, timestamp

print(f"Loaded {len(ratings_df)} ratings")

# Split data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Extract arrays for model
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_ratings = train_df['rating'].values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = test_df['rating'].values
```

### Step 2: Initialize Model

```python
model = AutoInt(
    name="AutoInt_Model",
    embedding_dim=32,
    attention_dim=32,
    num_heads=2,
    num_layers=3,
    dropout=0.1,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)

print(f"Initialized {model.name}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_users,
    item_ids=train_items,
    ratings=train_ratings
)

print("Training complete!")
```

### Step 4: Predict

```python
# Single prediction
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Batch predictions
test_predictions = model.batch_predict(list(zip(test_users[:100], test_items[:100])))
```

### Step 5: Recommend

```python
# Get top-10 recommendations for user
user_id = 1
recommendations = model.recommend(
    user_id=user_id,
    top_k=10
)

print(f"Top-10 recommendations for User {user_id}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Predict all test ratings
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
# Save model
model.save('autoint_model.pkl')

# Load model
loaded = AutoInt.load('autoint_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use AutoInt

✅ **Excellent For:**
- Feature-rich CTR prediction
- When manual feature engineering is expensive
- Discovering unknown feature interactions
- Sparse categorical features
- Display advertising, app recommendations

❌ **Not For:**
- Simple datasets (overkill)
- No features available (use CF)
- Sequential patterns (use RNN)
- Very large feature spaces (memory intensive)

### Best Practices

1. **Attention Heads**: 2-4 heads sufficient
2. **Layers**: 2-3 interacting layers
3. **Embedding Dim**: 16-64 per feature
4. **Attention Dim**: Same as embedding
5. **Residual**: Always use residual connections
6. **Dropout**: 0.1-0.2 on attention weights
7. **Learning Rate**: 0.001 with warmup
8. **Batch Size**: 256-1024

## Further Reading

- Paper: Song et al. 2019 - AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
