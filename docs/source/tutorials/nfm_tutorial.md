# NFM Tutorial: Neural Factorization Machine

## Introduction

**NFM** seamlessly combines the linearity of FM with the non-linearity of neural networks using a Bi-Interaction pooling layer.

**Paper**: He et al. 2017 - Neural Factorization Machines for Sparse Predictive Analytics

## How NFM Works

### Architecture

**Bi-Interaction Layer:**

1. **Embedding**: Sparse features to dense vectors
2. **Bi-Interaction**: Element-wise product of embedding pairs (pooling)
3. **Hidden Layers**: MLP to learn high-order non-linear interactions
4. **Prediction**: Linear part + Neural part

**Key Innovation**: Bi-Interaction layer captures 2nd-order interactions before deep layers

**Architecture:**
```
Embeddings -> Bi-Interaction Pooling -> MLP -> Prediction
```

### Mathematical Foundation

**Bi-Interaction Pooling:**
```
f_BI(V_x) = Σ_i Σ_{j>i} x_i v_i ⊙ x_j v_j
          = 0.5 * [(Σ x_i v_i)^2 - Σ (x_i v_i)^2]
```
Linear time complexity O(k)!

**Model Prediction:**
```
ŷ = w_0 + Σ w_i x_i + h^T σ_L(...σ_1(f_BI(V_x))...)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.nfm import NFM
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
model = NFM(
    name="NFM_Model",
    embedding_dim=64,
    hidden_units=[128, 64],
    dropout=0.2,
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
model.save('nfm_model.pkl')

# Load model
loaded = NFM.load('nfm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use NFM

✅ **Perfect For:**
- Sparse predictive analytics
- CTR prediction
- Capturing high-order feature interactions
- When FM is not expressive enough
- General replacement for FM/DeepFM

❌ **Not For:**
- Dense numerical data only
- Sequential data
- Image/Audio inputs

### Best Practices

1. **Dropout**: Essential after Bi-Interaction layer
2. **Batch Norm**: Helpful in MLP layers
3. **Activation**: ReLU or SeLU
4. **Optimizer**: Adam or Adagrad
5. **Embedding Dim**: 32-64

## Further Reading

- Paper: He et al. 2017 - Neural Factorization Machines for Sparse Predictive Analytics
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
