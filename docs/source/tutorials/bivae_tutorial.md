# BiVAE Tutorial: Bilateral Variational Autoencoder

## Introduction

**BiVAE** uses variational inference to jointly learn latent representations for both users and items with uncertainty quantification.

**Paper**: Collaborative Filtering via Variational Autoencoders

## How BiVAE Works

### Architecture

**Dual Variational Autoencoder:**

1. **User Encoder**: q(z_u | r_u) → μ_u, σ_u
2. **Item Encoder**: q(z_i | r_i) → μ_i, σ_i
3. **Sampling**: z_u ~ N(μ_u, σ_u²), z_i ~ N(μ_i, σ_i²)
4. **Decoder**: p(r | z_u, z_i) reconstructs rating
5. **KL Regularization**: Towards prior N(0,I)

**Key Innovation**: Probabilistic latent factors with uncertainty

**Architecture:**
```
User → Encode → μ_u, σ_u → Sample → z_u                                           Decode → Rating
Item → Encode → μ_i, σ_i → Sample → z_i /
```

### Mathematical Foundation

**ELBO (Evidence Lower Bound):**
```
L = E_{z~q}[log p(r|z_u,z_i)] - KL(q(z_u,z_i) || p(z_u)p(z_i))
```

**Reconstruction:**
```
log p(r|z_u,z_i) = -||r - f(z_u,z_i)||²
```

**KL Divergence:**
```
KL = -0.5 Σ(1 + log σ² - μ² - σ²)
```

**Reparameterization Trick:**
```
z = μ + σ ⊙ ε,  where ε ~ N(0,I)
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.bivae import BiVAE
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
model = BiVAE(
    name="BiVAE_Model",
    latent_dim=20,
    encoder_dims=[200, 100],
    kl_anneal_rate=0.01,
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
model.save('bivae_model.pkl')

# Load model
loaded = BiVAE.load('bivae_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use BiVAE

✅ **Good For:**
- Uncertainty quantification
- Generative modeling (sample ratings)
- Cold-start with priors
- Small-medium datasets
- Research and experimentation

❌ **Not For:**
- Large-scale production (slow)
- Need interpretability
- Real-time inference
- When point estimates suffice

### Best Practices

1. **Latent Dimension**: 20-50
2. **KL Annealing**: Start with β=0.01 → 1.0
3. **Encoder**: [200, 100] typical
4. **Decoder**: Symmetric [100, 200]
5. **Activation**: Tanh or ReLU
6. **Learning Rate**: 0.001 with decay
7. **Warmup**: 10-20 epochs before full KL
8. **Batch Size**: 128-512

## Further Reading

- Paper: Collaborative Filtering via Variational Autoencoders
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
