# DIEN Tutorial: Deep Interest Evolution Network

## Introduction

**DIEN** models the evolution of user interests over time using AUGRU (Attention-based GRU) to capture interest dynamics.

**Paper**: Zhou et al. 2019 - Deep Interest Evolution Network

## How DIEN Works

### Architecture

**Interest Evolution with Attention:**

1. **Interest Extractor Layer**:
   - GRU over user behavior sequence
   - Extracts interest states at each time step
   
2. **Interest Evolving Layer**:
   - AUGRU (Attention-based GRU) with attention scores
   - Models how interests evolve toward target item
   - Attention controls update gate based on relevance

3. **Final Layer**: Combines evolved interests with other features

**Key Innovation**: Uses attention to model interest evolution, not just representation

**Architecture:**
```
Behavior Seq → GRU (Interest Extract) → AUGRU (Evolution) → Final Interest → Predict
                                          ↑
                                    Attention from Target
```

### Mathematical Foundation

**Interest Extractor (GRU):**
```
h_t = GRU(e_t, h_{t-1})
```

**Attention Score:**
```
a_t = softmax(e_target^T · W · h_t)
```

**AUGRU Update:**
```
u_t' = a_t · u_t               # attention-weighted update gate
h_t' = (1 - u_t') ⊙ h_{t-1} + u_t' ⊙ h̃_t
```
where h̃_t is candidate state

**Final Representation:**
```
interest = Σ_t a_t · h_t'
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.dien import DIEN
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
model = DIEN(
    name="DIEN_Model",
    gru_hidden_dim=36,
    attention_hidden_dim=36,
    seq_len=50,
    use_auxiliary_loss=True,
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
model.save('dien_model.pkl')

# Load model
loaded = DIEN.load('dien_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use DIEN

✅ **Ideal For:**
- E-commerce with browsing sequences
- Interest drift is important (fashion, trends)
- Long user sequences (20-100 items)
- When you need to understand WHY user clicked
- Evolving user preferences

❌ **Not For:**
- Static preferences
- Short sequences (<10 items)
- Real-time constraints (AUGRU is slower)
- Simple collaborative filtering tasks

### Best Practices

1. **Sequence Length**: 20-50 items (longer than DIN)
2. **GRU Hidden Dim**: 36-64 typical
3. **Attention Type**: Scaled dot-product works well
4. **Auxiliary Loss**: Add on intermediate interest states
5. **Negative Sampling**: Sample from non-clicked items in session
6. **Feature Fusion**: Combine with user/item features
7. **Learning Rate**: 0.001, decay after epochs
8. **Training**: Expensive - use GPU acceleration

## Further Reading

- Paper: Zhou et al. 2019 - Deep Interest Evolution Network
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
