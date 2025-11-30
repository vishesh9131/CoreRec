# WideDeep Tutorial: Wide & Deep Learning

## Introduction

**WideDeep** jointly trains wide linear models for memorization and deep neural networks for generalization.

**Paper**: Cheng et al. 2016 - Wide & Deep Learning for Recommender Systems (Google)

## How WideDeep Works

### Architecture

**Hybrid Architecture:**

1. **Wide Component**: Generalized Linear Model (GLM)
   - Memorizes co-occurrence of features
   - Uses cross-product transformations
2. **Deep Component**: Feed-Forward Neural Network
   - Generalizes to unseen feature combinations
   - Uses low-dimensional embeddings
3. **Joint Training**: Weighted sum of both components

**Key Innovation**: Combining benefits of memorization (Wide) and generalization (Deep)

**Architecture:**
```
Wide Part (Linear)      Deep Part (DNN)
      ↓                       ↓
    Output <----- Sigmoid(Sum)
```

### Mathematical Foundation

**Prediction:**
```
P(y=1|x) = σ(w_wide^T [x, φ(x)] + w_deep^T a^(lf) + b)
```
where:
- φ(x): Cross-product transformations (Wide)
- a^(lf): Output of last deep layer
- σ: Sigmoid function

**Joint Optimization:**
Backpropagate gradients to both parts simultaneously using FTRL (Wide) and AdaGrad (Deep).

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.widedeep import WideDeep
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
model = WideDeep(
    name="WideDeep_Model",
    wide_features=['genre', 'os'],
    deep_features=['age', 'install_history'],
    hidden_units=[1024, 512, 256],
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
model.save('widedeep_model.pkl')

# Load model
loaded = WideDeep.load('widedeep_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use WideDeep

✅ **Perfect For:**
- App stores (Google Play)
- Recommending new & old items
- Large-scale regression/classification
- Mixed feature types (sparse + dense)
- Production systems needing low latency

❌ **Not For:**
- Pure collaborative filtering (no features)
- Sequential patterns (use RNN/Transformer)
- Small datasets (overfitting risk)

### Best Practices

1. **Wide Features**: Cross-product of important categorical features
2. **Deep Features**: Embeddings for sparse, raw values for dense
3. **Optimizers**: FTRL for Wide (sparsity), Adam/AdaGrad for Deep
4. **Hidden Layers**: [1024, 512, 256] typical
5. **Embedding Dim**: 32-128
6. **Batch Size**: Large (thousands)

## Further Reading

- Paper: Cheng et al. 2016 - Wide & Deep Learning for Recommender Systems (Google)
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
