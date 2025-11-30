# RBM Tutorial: Restricted Boltzmann Machine

## Introduction

**RBM** uses a probabilistic graphical model with visible (ratings) and hidden (latent features) units to learn user preferences.

**Paper**: Salakhutdinov et al. 2007 - Restricted Boltzmann Machines for Collaborative Filtering

## How RBM Works

### Architecture

**Energy-Based Probabilistic Model:**

1. **Visible Units**: User ratings for items
2. **Hidden Units**: Latent user preferences
3. **Bipartite Structure**: No visible-visible or hidden-hidden connections
4. **Training**: Contrastive Divergence (CD-k)

**Key Innovation**: Generative model that can sample new ratings

**Architecture:**
```
Visible (Ratings) ↔ Weights W ↔ Hidden (Features)
      ↓                              ↓
  Biases a                      Biases b
```

No connections within visible or hidden layers!

### Mathematical Foundation

**Energy Function:**
```
E(v,h) = -Σ_i a_i·v_i - Σ_j b_j·h_j - Σ_i Σ_j v_i·h_j·w_ij
```

**Probability:**
```
P(v,h) = exp(-E(v,h)) / Z
where Z = Σ_{v',h'} exp(-E(v',h'))  # partition function
```

**Conditional Probabilities:**
```
P(h_j=1|v) = σ(b_j + Σ_i v_i·w_ij)
P(v_i=1|h) = σ(a_i + Σ_j h_j·w_ij)
```

**Contrastive Divergence Update:**
```
ΔW = ε(<v_0 h_0^T> - <v_k h_k^T>)
```
where v_0 is data, v_k is reconstruction after k Gibbs steps

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.rbm import RBM
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
model = RBM(
    name="RBM_Model",
    n_hidden=200,
    learning_rate=0.01,
    momentum=0.5,
    n_epochs=100,
    cd_steps=1,
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
model.save('rbm_model.pkl')

# Load model
loaded = RBM.load('rbm_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use RBM

✅ **Ideal For:**
- Explicit ratings (discrete values)
- Need generative capability (sample ratings)
- Feature learning from ratings
- Cold-start scenarios (can infer from partial data)
- Research and experimentation

❌ **Not For:**
- Large-scale production (slow training)
- Implicit feedback
- need online updates (batch model)
- Interpretability required

### Best Practices

1. **Hidden Units**: 100-500 typical
2. **CD Steps (k)**: k=1 works well (CD-1)
3. **Learning Rate**: 0.01-0.1, use learning rate decay
4. **Momentum**: 0.5 → 0.9 over training
5. **Weight Decay**: 0.0001-0.001
6. **Mini-batch**: 100-1000 users per batch
7. **Epochs**: 50-200 epochs
8. **Initialization**: Small random weights ~N(0, 0.01)

## Further Reading

- Paper: Salakhutdinov et al. 2007 - Restricted Boltzmann Machines for Collaborative Filtering
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
