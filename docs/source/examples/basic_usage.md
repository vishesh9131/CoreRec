# Basic Usage Examples

This guide provides basic examples for getting started with CoreRec production models.

## Quick Start

### Simple Recommendation (DCN)

```python
from corerec.engines.dcn import DCN
import numpy as np

user_ids = np.array([0, 0, 1, 1, 2, 2])
item_ids = np.array([0, 1, 0, 2, 1, 2])
ratings = np.array([5, 4, 5, 3, 4, 5])

model = DCN(embedding_dim=32, epochs=10, verbose=True)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)

score = model.predict(user_id=0, item_id=2)
print(f"Predicted score: {score}")

recommendations = model.recommend(user_id=0, top_k=5)
print(f"Recommendations: {recommendations}")
```

## Common Patterns

### Loading Data from CSV

```python
import pandas as pd

data = pd.read_csv('ratings.csv')
user_ids = data['user_id'].values
item_ids = data['item_id'].values
ratings = data['rating'].values
```

### Training a Deep Learning Model

```python
from corerec.engines.deepfm import DeepFM

model = DeepFM(embedding_dim=64, epochs=20, learning_rate=0.001, verbose=True)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
```

### Collaborative Filtering (SAR)

```python
from corerec.engines.collaborative import SAR
import pandas as pd

df = pd.DataFrame({
    'userID': [0, 0, 1, 1],
    'itemID': [0, 1, 0, 2],
    'rating': [5, 4, 5, 3]
})

model = SAR(similarity_type='jaccard')
model.fit(df)
recs = model.recommend(user_id=0, top_k=10)
```

### Neural Collaborative Filtering (NCF)

```python
from corerec.engines.collaborative import NCF

ncf_df = pd.DataFrame({'user_id': [0, 0, 1], 'item_id': [0, 1, 2], 'rating': [1, 1, 1]})
model = NCF(num_epochs=10, verbose=True)
model.fit(ncf_df)
recs = model.recommend(user_id=0, top_k=10)
```

### Sequential Models (SASRec)

```python
from corerec.engines.sasrec import SASRec
import numpy as np

user_list = [0, 1]
item_list = [0, 1, 2]
interaction_matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)

model = SASRec(num_epochs=5, hidden_units=32, max_seq_length=10, verbose=True)
model.fit(user_list, item_list, interaction_matrix)
recs = model.recommend(user_list[0], top_k=5)
```

### Saving and Loading Models

```python
model.save('artifacts/my_model')  # safe bundle default
from corerec.engines.dcn import DCN
loaded = DCN.load('artifacts/my_model')
```

## Next Steps

- See [Advanced Usage](advanced_usage.md) for ensemble patterns
- Check [Production Deployment](production_deployment.md) for deployment guides
- Explore [Tutorials](../tutorials/index.md) for model-specific walkthroughs
