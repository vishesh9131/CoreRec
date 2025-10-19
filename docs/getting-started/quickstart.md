# Quick Start Guide

Get up and running with CoreRec in minutes! This guide will walk you through building your first recommendation system.

## 5-Minute Quick Start

### Step 1: Install CoreRec

```bash
pip install --upgrade corerec
```

### Step 2: Prepare Your Data

CoreRec works with simple user-item interaction data:

```python
import numpy as np

# Example: User-Item interactions
user_ids = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
item_ids = [101, 102, 101, 103, 102, 104, 103, 105, 104, 105]
ratings = [5.0, 4.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
```

### Step 3: Train a Model

Let's train a Deep & Cross Network (DCN):

```python
from corerec.engines.dcn import DCN

# Initialize model
model = DCN(
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64, 32],
    epochs=10,
    batch_size=32,
    device='cpu'
)

# Train model
model.fit(user_ids, item_ids, ratings)
print("✓ Model trained successfully!")
```

### Step 4: Get Recommendations

```python
# Get top-10 recommendations for user 1
recommendations = model.recommend(user_id=1, top_n=10)
print(f"Recommendations for user 1: {recommendations}")

# Predict rating for a specific user-item pair
score = model.predict(user_id=1, item_id=106)
print(f"Predicted score: {score:.2f}")
```

That's it! You've built your first recommendation system with CoreRec! 🎉

---

## Complete Examples by Engine Type

### Example 1: Unionized Filter Engine

Collaborative filtering with matrix factorization:

```python
from corerec.engines.unionizedFilterEngine.fast import FastRecommender

# Initialize
model = FastRecommender(
    n_factors=50,
    n_epochs=20,
    learning_rate=0.01
)

# Train
model.fit(user_ids, item_ids, ratings)

# Recommend
recs = model.recommend(user_id=1, top_k=10)
print(recs)
```

### Example 2: Content Filter Engine

Content-based filtering with TF-IDF:

```python
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender
import pandas as pd

# Prepare content data
items = pd.DataFrame({
    'item_id': [101, 102, 103],
    'description': [
        'Action movie with explosions',
        'Romantic comedy film',
        'Action thriller adventure'
    ]
})

# Initialize and fit
model = TFIDFRecommender(feature_column='description')
model.fit(items)

# Get similar items
similar = model.recommend_similar(item_id=101, top_k=5)
print(similar)
```

### Example 3: Deep Learning Models

Neural collaborative filtering with embeddings:

```python
from corerec.engines.deepfm import DeepFM

# Initialize DeepFM
model = DeepFM(
    embedding_dim=64,
    hidden_layers=[128, 64, 32],
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    device='cuda'  # Use GPU if available
)

# Train
model.fit(user_ids, item_ids, ratings)

# Batch recommendations
batch_recs = model.batch_recommend(
    user_ids=[1, 2, 3],
    top_n=10
)
print(batch_recs)
```

---

## Working with Real Data

### Loading Data from CSV

```python
import pandas as pd

# Load data
df = pd.read_csv('interactions.csv')

# Expected format:
# user_id, item_id, rating, timestamp (optional)

user_ids = df['user_id'].tolist()
item_ids = df['item_id'].tolist()
ratings = df['rating'].tolist()
```

### Loading Data with cr_learn

CoreRec includes a dataset loader:

```python
from corerec.data import load_dataset

# Load MovieLens data
data = load_dataset('movielens-100k')

user_ids = data['user_id']
item_ids = data['item_id']
ratings = data['rating']
```

### Using Sample Data

CoreRec provides sample datasets:

```python
from corerec.utils.example_data import get_sample_data

# Get Netflix-style sample data
data = get_sample_data('netflix')

user_ids = data['user_ids']
item_ids = data['item_ids']
ratings = data['ratings']
```

---

## Model Evaluation

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data
train_users, test_users, train_items, test_items, train_ratings, test_ratings = \
    train_test_split(user_ids, item_ids, ratings, test_size=0.2)

# Train
model.fit(train_users, train_items, train_ratings)

# Evaluate
from corerec.evaluation import evaluate_model

metrics = evaluate_model(
    model,
    test_users,
    test_items,
    test_ratings,
    metrics=['rmse', 'mae', 'precision@10', 'recall@10']
)

print(metrics)
```

### Available Metrics

```python
from corerec.metrics import (
    rmse, mae,              # Rating prediction
    precision_at_k,         # Ranking quality
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    hit_rate_at_k
)

# Calculate metrics
rmse_score = rmse(test_ratings, predictions)
precision = precision_at_k(true_items, recommended_items, k=10)
ndcg = ndcg_at_k(true_items, recommended_items, k=10)
```

---

## Saving and Loading Models

### Save a Trained Model

```python
# Save model
model.save('models/dcn_model.pkl')

# Or save as PyTorch checkpoint
model.save('models/dcn_model.pth', format='torch')
```

### Load a Saved Model

```python
# Load model
from corerec.engines.dcn import DCN

model = DCN.load('models/dcn_model.pkl')

# Use loaded model
recs = model.recommend(user_id=1, top_n=10)
```

---

## Visualization

### Visualize Graph Structure

```python
import corerec.vish_graphs as vg

# Create interaction graph
adj_matrix = model.get_interaction_matrix()

# 2D visualization
vg.draw_graph(adj_matrix, top_nodes=[1, 2, 3])

# 3D visualization
vg.draw_graph_3d(adj_matrix, top_nodes=[1, 2, 3])
```

### Visualize Training Progress

```python
import matplotlib.pyplot as plt

# Get training history
history = model.history

plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

---

## Advanced Configuration

### Using Configuration Files

```python
from corerec.config import ConfigManager

# Load config from YAML
config = ConfigManager.from_yaml('config.yaml')

# Initialize model with config
model = DCN(**config.model_params)
model.fit(user_ids, item_ids, ratings, **config.training_params)
```

Example `config.yaml`:

```yaml
model_params:
  embedding_dim: 64
  num_cross_layers: 3
  deep_layers: [128, 64, 32]
  dropout: 0.2

training_params:
  epochs: 20
  batch_size: 256
  learning_rate: 0.001
  device: cuda
```

### Custom Optimizers

```python
from corerec.cr_boosters.adam import Adam

# Use custom optimizer
model = DCN(
    embedding_dim=64,
    num_cross_layers=3,
    optimizer=Adam(learning_rate=0.001)
)
```

Available optimizers:
- `Adam`, `Nadam`, `Adamax`
- `Adadelta`, `Adagrad`
- `RMSprop`, `SGD`
- `LBFGS`, `ASGD`
- `SparseAdam`

---

## Next Steps

Now that you've built your first recommender, explore more:

- **[Architecture Overview](architecture.md)**: Understand CoreRec's design
- **[User Guide](../user-guide/index.md)**: Deep dive into features
- **[Engines](../engines/index.md)**: Explore all available engines
- **[Examples](../examples/index.md)**: Real-world use cases
- **[API Reference](../api/index.md)**: Detailed API documentation

---

## Common Use Cases

### E-commerce Product Recommendations

```python
# Product recommendation system
from corerec.engines.deepfm import DeepFM

model = DeepFM(embedding_dim=128, hidden_layers=[256, 128, 64])
model.fit(customer_ids, product_ids, purchase_amounts)

# Recommend products
recs = model.recommend(customer_id=12345, top_n=10)
```

### Movie Recommendations

```python
# Movie recommendation with sequential model
from corerec.engines.sasrec import SASRec

model = SASRec(hidden_units=64, num_blocks=2, num_heads=4)
model.fit(user_ids, movie_ids, timestamps)

# Get next movie recommendations
recs = model.recommend(user_id=456, top_n=5)
```

### Music Recommendations

```python
# Music recommendation with multi-interest
from corerec.engines.mind import MIND

model = MIND(embedding_dim=64, num_interests=4)
model.fit(user_ids, song_ids, listen_times)

# Diverse music recommendations
recs = model.recommend(user_id=789, top_n=20)
```

---

!!! tip "Pro Tips"
    - Start with simple models (FastRecommender, TF-IDF) before moving to deep learning
    - Use GPU (`device='cuda'`) for large datasets
    - Experiment with hyperparameters using grid search
    - Monitor training with TensorBoard or Weights & Biases
    - Use cross-validation for robust evaluation

!!! example "Need Help?"
    - Check [Examples](../examples/index.md) for more code samples
    - Visit [GitHub Issues](https://github.com/vishesh9131/CoreRec/issues) for support
    - Read [Best Practices](../user-guide/best-practices.md) for optimization tips


