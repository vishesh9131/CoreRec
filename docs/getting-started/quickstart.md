# Quick Start Guide

Get your first recommendation model running in under 5 minutes.

## 1. Installation

Install CoreRec using pip:

```bash
pip install corerec
```

## 2. Hello World: Matrix Factorization

This example loads a sample dataset, trains a simple Collaborative Filtering model, and generates recommendations.

```python
import pandas as pd
from corerec.engines.collaborative import FastRecommender

# 1. Load Data (User-Item-Rating)
# Replace with your own CSV or dataset
data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4],
    'item_id': [101, 102, 101, 103, 102, 104, 101],
    'rating':  [5, 3, 4, 5, 2, 5, 4]
})

# 2. Initialize Model
# 'FastRecommender' uses optimized Matrix Factorization
model = FastRecommender(embedding_dim=64, epochs=10)

# 3. Train
print("Training model...")
model.fit(
    interaction_matrix=None,  # Not needed for FastRecommender, it handles raw data
    user_ids=data['user_id'].tolist(),
    item_ids=data['item_id'].tolist()
)

# 4. Recommend
user_id = 1
recs = model.recommend(user_id=user_id, top_n=3)

print(f"Top 3 recommendations for User {user_id}: {recs}")
```

## 3. Advanced: Deep Learning (SASRec)

For sequential data (training on user history), use the Deep Learning engine.

```python
from corerec.engines import SASRec

# SASRec expects sequential input
model = SASRec(
    item_num=1000,  # Total number of items
    hidden_units=64,
    num_blocks=2,
    num_heads=2
)

# ... (Training loop with standard PyTorch DataLoader) ...
```

## Next Steps

Now that you have your first model running:
*   Explore the [**User Guide**](../user-guide/index.md) for more complex pipelines.
*   Learn how to [**Visualize Your Graph**](../utilities/visualization.md) with VishGraphs.
