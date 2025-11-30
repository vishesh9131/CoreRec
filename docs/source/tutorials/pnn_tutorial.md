# PNN Tutorial: Product-based Neural Network

## Introduction

**PNN** uses a product layer to capture interactive patterns between inter-field categories before feeding into fully connected layers.

**Paper**: Qu et al. 2016 - Product-based Neural Networks for User Response Prediction

## How PNN Works

### Architecture

**Product Layer:**

1. **Embedding Layer**: Feature embeddings
2. **Product Layer**:
   - Linear part (z): Concatenation
   - Quadratic part (p): Inner or Outer products of embeddings
3. **Hidden Layers**: MLP on top of product layer

**Key Innovation**: Explicitly modeling feature interactions via products (IPNN/OPNN)

**Architecture:**
```
Embeddings -> [Linear z, Product p] -> MLP -> Prediction
```

### Mathematical Foundation

**Inner Product (IPNN):**
```
p_ij = <v_i, v_j>
```

**Outer Product (OPNN):**
```
p_ij = v_i v_j^T
```

**Prediction:**
```
l_1 = ReLU(W_z z + W_p p + b_1)
ŷ = σ(W_out l_n + b_out)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.pnn import PNN
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
model = PNN(
    name="PNN_Model",
    embedding_dim=64,
    use_inner_product=True,
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
model.save('pnn_model.pkl')

# Load model
loaded = PNN.load('pnn_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use PNN

✅ **Perfect For:**
- CTR prediction
- Multi-field categorical data
- When interactions are crucial
- Ad click prediction

❌ **Not For:**
- High-dimensional sparse data without fields
- Sequence modeling

### Best Practices

1. **Inner vs Outer**: IPNN usually faster and sufficient
2. **Kernel Trick**: Use factorization to speed up product layer
3. **Hidden Units**: [256, 128, 64]
4. **Regularization**: L2 or Dropout needed

## Further Reading

- Paper: Qu et al. 2016 - Product-based Neural Networks for User Response Prediction
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
