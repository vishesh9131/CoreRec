# DCN Tutorial: Deep & Cross Network

## Introduction

**DCN** combines explicit feature crossing with deep learning to model both low and high-order feature interactions efficiently.

**Paper**: Wang et al. 2017 - Deep & Cross Network for Ad Click Predictions

## How DCN Works

### Architecture

**Two Parallel Networks:**

1. **Cross Network**: Learns explicit bounded-degree feature interactions
   - Applies element-wise multiplication at each layer
   - Models feature crossing efficiently: `x_{l+1} = x_0 · x_l^T · w_l + b_l + x_l`
   - Each layer increases polynomial degree by 1

2. **Deep Network**: Learns implicit high-order interactions
   - Standard fully-connected neural network
   - Captures complex non-linear patterns
   
3. **Combination Layer**: Concatenates outputs and applies final transformation

**Flow:**
```
Features → [Cross Network]                             → Concat → Dense → Prediction
Features → [Deep Network]  /
```

### Mathematical Foundation

**Cross Network Layer l:**
```
x_{l+1} = x_0 · x_l^T · w_l + b_l + x_l
```
where:
- `x_0` is input feature vector  
- `x_l` is output of layer l
- `w_l, b_l` are learnable parameters
- Complexity: O(d) per layer

**Deep Network Layer l:**
```
h_{l+1} = ReLU(W_l · h_l + b_l)
```

**Final Prediction:**
```
y = σ([x_cross; h_deep] · w_out + b_out)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.dcn import DCN
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
model = DCN(
    name="DCN_Model",
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64],
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
model.save('dcn_model.pkl')

# Load model
loaded = DCN.load('dcn_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DCN

✅ **Excellent For:**
- Feature-rich datasets (user demographics, item attributes, context)
- Ad click prediction and CTR estimation
- When explicit feature crossing matters (e.g., age × gender, category × price)
- Datasets with 100K-10M interactions
- E-commerce product recommendations

❌ **Not Ideal For:**
- Pure collaborative filtering (no features) → use Matrix Factorization
- Sequential/temporal patterns → use SASRec or LSTM
- Graph-structured data → use GNN
- Very large scale (>100M interactions) → use simpler FM

### Best Practices

1. **Feature Engineering**: DCN shines with good features - invest time here
2. **Cross Layers**: 2-3 layers sufficient (more = overfitting risk)
3. **Deep Layers**: Start with [128, 64], add depth for complex datasets
4. **Embedding Dimension**: 32-128 based on feature cardinality
5. **Regularization**: Use dropout (0.2-0.3) and L2 (1e-5)
6. **Normalization**: Normalize continuous features to [0,1] or [-1,1]
7. **Learning Rate**: 0.001 with decay works well
8. **Batch Size**: 256-512 for stability

## Further Reading

- Paper: Wang et al. 2017 - Deep & Cross Network for Ad Click Predictions
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
