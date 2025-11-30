# Basic Usage Examples

This guide provides basic examples for getting started with CoreRec.

## Quick Start

### Simple Recommendation

```python
from corerec.engines.dcn import DCN
import numpy as np

# Prepare data
user_ids = np.array([0, 0, 1, 1, 2, 2])
item_ids = np.array([0, 1, 0, 2, 1, 2])
ratings = np.array([5, 4, 5, 3, 4, 5])

# Create and train model
model = DCN(embedding_dim=32, epochs=10, verbose=True)
model.fit(user_ids, item_ids, ratings)

# Make predictions
score = model.predict(user_id=0, item_id=2)
print(f"Predicted score: {score}")

# Get recommendations
recommendations = model.recommend(user_id=0, top_k=5)
print(f"Recommendations: {recommendations}")
```

## Common Patterns

### Loading Data

```python
import pandas as pd

# From CSV
data = pd.read_csv('ratings.csv')
user_ids = data['user_id'].values
item_ids = data['item_id'].values
ratings = data['rating'].values

# From DataFrame
df = pd.DataFrame({
    'user_id': [0, 0, 1, 1],
    'item_id': [0, 1, 0, 2],
    'rating': [5, 4, 5, 3]
})
```

### Training a Model

```python
from corerec.engines.deepfm import DeepFM

model = DeepFM(
    embedding_dim=64,
    epochs=20,
    learning_rate=0.001,
    verbose=True
)

model.fit(
    user_ids=user_ids,
    item_ids=item_ids,
    ratings=ratings
)
```

### Making Predictions

```python
# Single prediction
score = model.predict(user_id=1, item_id=10)

# Batch predictions
scores = []
for user_id, item_id in zip(user_ids, item_ids):
    score = model.predict(user_id, item_id)
    scores.append(score)
```

### Generating Recommendations

```python
# Top-K recommendations
recommendations = model.recommend(user_id=1, top_k=10)

# Exclude known items
known_items = [0, 1, 2]
recommendations = model.recommend(
    user_id=1,
    top_k=10,
    exclude_items=known_items
)
```

### Saving and Loading Models

```python
# Save model
model.save('my_model.pkl')

# Load model
from corerec.engines.dcn import DCN
loaded_model = DCN.load('my_model.pkl')
```

## Model Selection

### For Collaborative Filtering

```python
from corerec.engines.unionizedFilterEngine.mf_base.als_recommender import ALSRecommender
import numpy as np

# Create user-item matrix
user_item_matrix = np.random.rand(100, 50)  # 100 users, 50 items

model = ALSRecommender(n_factors=50, n_iterations=20)
model.fit(user_item_matrix)
```

### For Deep Learning

```python
from corerec.engines.dcn import DCN

model = DCN(
    embedding_dim=64,
    epochs=20,
    learning_rate=0.001
)
model.fit(user_ids, item_ids, ratings)
```

### For Sequential Data

```python
from corerec.engines.sasrec import SASRec

model = SASRec(
    embedding_dim=64,
    n_layers=2,
    max_len=50
)
model.fit(user_sequences, item_ids)
```

## Next Steps

- See [Advanced Usage](advanced_usage.md) for more complex examples
- Check [Production Deployment](production_deployment.md) for deployment guides
- Explore [Tutorials](../tutorials/index.md) for model-specific tutorials

