# GeoIMC Tutorial: Geographic Inductive Matrix Completion

## Introduction

**GeoIMC** leverages geographic information and graph structure for location-based recommendations using inductive matrix completion.

**Paper**: Spatial Matrix Completion

## How GeoIMC Works

### Architecture

**Spatial-Aware Matrix Completion:**

1. **Geographic Graph**: Locations connected by proximity
2. **Graph Convolution**: Propagate location features
3. **User Preferences**: Latent factors
4. **Inductive Completion**: Generalize to new locations
5. **Spatial Regularization**: Nearby locations similar

**Key Innovation**: Combines CF with geographic proximity

**Architecture:**
```
Location Graph → GCN → Location Embeddings                                             → MF → Rating Prediction
User Features  → MLP → User Embeddings     /
```

### Mathematical Foundation

**Graph Convolution on Locations:**
```
h_l^(k+1) = σ(Σ_{l'∈N(l)} W^k · h_l'^(k) / |N(l)|)
```

**Matrix Factorization:**
```
r̂_ul = p_u^T · q_l + b_u + b_l
```

**Spatial Regularization:**
```
L_spatial = λ_s · Σ_{l,l'∈neighbors} ||q_l - q_l'||²
```

**Total Loss:**
```
L = L_MF + L_spatial + L_graph
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.geoimc import GeoIMC
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
model = GeoIMC(
    name="GeoIMC_Model",
    n_factors=64,
    gcn_layers=2,
    spatial_reg=0.5,
    neighbor_radius_km=2.0,
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
model.save('geoimc_model.pkl')

# Load model
loaded = GeoIMC.load('geoimc_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use GeoIMC

✅ **Perfect For:**
- Point-of-Interest (POI) recommendation
- Location-based services (Yelp, Foursquare)
- Check-in data
- When geography matters
- Cold-start locations (inductive)

❌ **Not For:**
- No geographic structure
- Pure collaborative filtering
- When location doesn't matter (movies, books)
- Very sparse check-ins (<100 per user)

### Best Practices

1. **Graph Construction**: k-NN or radius-based (1-5km)
2. **GCN Layers**: 2-3 layers
3. **MF Factors**: 50-100
4. **Spatial Regularization**: λ_s = 0.1-1.0
5. **Distance Weighting**: Inverse distance or Gaussian
6. **Features**: Include category, popularity
7. **Negative Sampling**: Geographic-aware sampling
8. **Train/Test**: Split by time, not random

## Further Reading

- Paper: Spatial Matrix Completion
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
