# Examples

This section provides comprehensive examples for using CoreRec in various scenarios. All examples are runnable and include complete code.

Runnable scripts live in the repository's [`examples/`](https://github.com/vishesh9131/CoreRec/tree/main/examples) folder. The code on this page uses `sample_data/events.csv` from the repository (about 33k events with `user_id`, `item_id`, `rating`, `timestamp`), so run it from the repository root, or point `pd.read_csv` at your own file with the same columns.

## By Use Case

### E-commerce

```python
import pandas as pd
from corerec.engines import DeepFM

events = pd.read_csv('sample_data/events.csv')

# Product recommendations
model = DeepFM(embedding_dim=16, hidden_layers=[64, 32], epochs=2)
model.fit(events['user_id'].tolist(), events['item_id'].tolist(), events['rating'].tolist())

# Get personalized product recommendations
recommendations = model.recommend(user_id=events['user_id'].iloc[0], top_k=10)
```

### Movie Recommendations

```python
import pandas as pd
from corerec.engines import SASRec

events = pd.read_csv('sample_data/events.csv').sort_values('timestamp')

# Sequential movie recommendations: events in time order, one row each
model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=1,
    epochs=2,
    max_seq_length=20,
    verbose=False
)
model.fit(events['user_id'].tolist(), events['item_id'].tolist(), events['rating'].tolist())

# Get next movie recommendations
next_movies = model.recommend(user_id=events['user_id'].iloc[0], top_k=5)
```

### Music Streaming

`MIND` was removed in 0.7.0. `HSTU` models a listening history as a sequence
and takes the timestamps directly:

```python
import pandas as pd
from corerec.engines import HSTU

events = pd.read_csv('sample_data/events.csv').sort_values('timestamp')

model = HSTU(embedding_dim=32, epochs=1)
model.fit(events['user_id'].tolist(), events['item_id'].tolist(),
          timestamps=events['timestamp'].tolist())

# Next tracks for a listener
recommendations = model.recommend(user_id=events['user_id'].iloc[0], top_k=20)
```

### News Articles

```python
import pandas as pd
from corerec.engines import TFIDFRecommender

# Content-based recommendations from text (any id -> text table works)
articles = pd.read_csv('sample_data/netflix_demo.csv')
docs = dict(zip(articles['content_id'], articles['title'] + '. ' + articles['description']))

model = TFIDFRecommender()
model.fit(list(docs), docs)

# Get articles matching a query
similar_articles = model.recommend_by_text('documentary', top_k=10)
```

### Social Networks

`GNNRec` was removed in 0.7.0. `LightGCN` learns from any bipartite graph, so
"user follows account" edges work the same way as "user bought item":

```python
import numpy as np
from corerec.engines import LightGCN

# User-user interaction data: who follows whom
rng = np.random.default_rng(0)
follower = rng.integers(0, 300, 4000).tolist()
followed = rng.integers(0, 300, 4000).tolist()

model = LightGCN(n_factors=32, n_layers=2, epochs=10)
model.fit(follower, followed, [1.0] * len(follower))

# Recommend new connections
friend_suggestions = model.recommend(user_id=follower[0], top_k=10)
```

## Complete End-to-End Examples

### Example 1: Movie Recommendation System

```python
import pandas as pd
from corerec.engines import DCN
from corerec.evaluation import evaluate

# 1. Load data
events = pd.read_csv('sample_data/events.csv').sort_values('timestamp')

# 2. Train/test split: the newest 20% of events is the test set
cut = int(len(events) * 0.8)
train, test = events.iloc[:cut], events.iloc[cut:]

# 3. Initialize and train model
model = DCN(
    embedding_dim=16,
    num_cross_layers=3,
    deep_layers=[64, 32],
    epochs=2,
    batch_size=256,
    device='auto'          # CUDA or Apple MPS when available
)
model.fit(train['user_id'].tolist(), train['item_id'].tolist(), train['rating'].tolist())

# 4. Evaluate: ranking metrics through model.recommend(), seen items removed
metrics = evaluate(model, test, train_interactions=train, k=10)
print(f"NDCG@10: {metrics['NDCG@10']:.4f}  Recall@10: {metrics['Recall@10']:.4f}")

# 5. Get recommendations
user_id = test['user_id'].iloc[0]
recommendations = model.recommend(user_id=user_id, top_k=10)
print(f"Top 10 movies for user {user_id}: {recommendations}")

# 6. Save model
model.save('models/movie_recommender')
```

### Example 2: E-commerce Product Recommendations

```python
import pandas as pd
from corerec.engines import DeepFM

# 1. Load purchase data
purchases = pd.read_csv('sample_data/events.csv')

# 2. Prepare data
customer_ids = purchases['user_id'].tolist()
product_ids = purchases['item_id'].tolist()
purchase_amounts = purchases['rating'].tolist()

# 3. Initialize DeepFM
model = DeepFM(
    embedding_dim=16,
    hidden_layers=[64, 32],
    epochs=2,
    batch_size=512,
    learning_rate=0.001
)

# 4. Train model
model.fit(customer_ids, product_ids, purchase_amounts)

# 5. Batch recommendations for all customers
all_customer_ids = purchases['user_id'].unique().tolist()
batch_recommendations = model.batch_recommend(all_customer_ids[:100], top_k=10)

# 6. Export recommendations
recommendations_df = pd.DataFrame([
    {'customer_id': uid, 'recommended_products': recs}
    for uid, recs in batch_recommendations.items()
])
recommendations_df.to_csv('product_recommendations.csv', index=False)
```

### Example 3: Sequential Music Recommendations

```python
import pandas as pd
from corerec.engines import SASRec

# 1. Load listening history
listens = pd.read_csv('sample_data/events.csv').sort_values('timestamp')

# 2. SASRec reads the order of the rows: one event per row, oldest first
user_ids = listens['user_id'].tolist()
song_ids = listens['item_id'].tolist()

# 3. Initialize SASRec
model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=1,
    epochs=3,
    batch_size=128,
    max_seq_length=20,
    verbose=False
)

# 4. Train model
model.fit(user_ids, song_ids, [1.0] * len(user_ids))

# 5. Get next song recommendations
user_id = user_ids[0]
next_songs = model.recommend(user_id=user_id, top_k=20)
print(f"Next 20 songs for user {user_id}: {next_songs}")

# 6. Visualize training progress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

history = model.training_history          # [{'epoch': 1, 'loss': ...}, ...]
plt.plot([h['epoch'] for h in history], [h['loss'] for h in history], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
```

## Data Preparation Examples

### Example: Loading from CSV

```python
import pandas as pd

# Load data
df = pd.read_csv('sample_data/events.csv')

# Extract columns
user_ids = df['user_id'].tolist()
item_ids = df['item_id'].tolist()
ratings = df['rating'].tolist()

# Optional: timestamps
timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else None
```

From the command line, `corerec train sample_data/events.csv` detects the
columns itself and reports a holdout score.

### Example: Creating Synthetic Data

```python
import numpy as np

# Generate synthetic data for testing
rng = np.random.default_rng(0)
num_users = 1000
num_items = 500
num_interactions = 10000

user_ids = rng.integers(0, num_users, num_interactions).tolist()
item_ids = rng.integers(0, num_items, num_interactions).tolist()
ratings = rng.uniform(1, 5, num_interactions).tolist()
```

### Example: Using Sample Data

The repository ships small files in `sample_data/`: `events.csv`
(interactions) and `netflix_demo.csv`, `spotify_demo.csv`, `youtube_demo.csv`
(item catalogues with text). For MovieLens-1M, install the datasets extra:

```bash
pip install "corerec[datasets]"
```

```python
from cr_learn import ml_1m

data = ml_1m.load()          # downloads on first use
ratings = data['ratings']
user_ids = ratings['user_id'].tolist()
item_ids = ratings['movie_id'].tolist()
```

## Evaluation Examples

### Example: Complete Evaluation

```python
import numpy as np
import pandas as pd
from corerec.engines import ALS
from corerec.evaluation import Evaluator, evaluate

events = pd.read_csv('sample_data/events.csv').sort_values('timestamp')
cut = int(len(events) * 0.8)
train, test = events.iloc[:cut], events.iloc[cut:]
model = ALS(factors=32).fit(train['user_id'].tolist(), train['item_id'].tolist())

# Ranking metrics at one or more cutoffs
results = evaluate(model, test, train_interactions=train, k=10)
print(results)

# The same through Evaluator, with test data as {user: [relevant items]}
test_data = test.groupby('user_id')['item_id'].apply(list).to_dict()
evaluator = Evaluator(metrics=['Precision@10', 'Recall@10', 'NDCG@10'])
print(evaluator.evaluate(model, test_data))

# Rating error for a model that predicts ratings, computed directly
pairs = list(zip(test['user_id'], test['item_id']))[:500]
predictions = np.array(model.batch_predict(pairs))
truth = test['rating'].to_numpy()[:500]
rmse_score = float(np.sqrt(np.mean((truth - predictions) ** 2)))
print(f"RMSE: {rmse_score:.4f}")   # ALS scores preferences, so this is only illustrative
```

## Visualization Examples

### Example: Graph Visualization

`corerec.vish_graphs` was removed in 0.7.0. The user-item graph is a sparse
matrix you can hand to networkx or any plotting tool:

```python
import pandas as pd
from scipy.sparse import csr_matrix

events = pd.read_csv('sample_data/events.csv')
users, user_index = pd.factorize(events['user_id'])
items, item_index = pd.factorize(events['item_id'])

# Get interaction matrix (users x items)
adj_matrix = csr_matrix(([1.0] * len(events), (users, items)),
                        shape=(len(user_index), len(item_index)))
print(adj_matrix.shape, adj_matrix.nnz)
```

## Running Examples

All examples are located in the `examples/` directory of the CoreRec repository:

```bash
# Run engine quickstart
python examples/engines_quickstart.py

# Run specific engine example
python examples/engines_dcn_example.py
python examples/engines_deepfm_example.py
python examples/engines_sasrec_example.py

# Run unionized filter examples
python examples/unionized_sar_example.py

# Run content filter examples
python examples/content_filter_tfidf_example.py

# Run advanced examples
python examples/pipeline_example.py
python examples/train_and_serve.py

# Run all tests
python -m pytest tests/ -m "not docs_build"
```

## Interactive Examples

No notebooks ship with the repository. Every block on this page runs as-is in
a Jupyter cell (from the repository root):

```bash
pip install jupyter
jupyter notebook
```

## Next Steps

- Explore [Engine Documentation](../engines/index.md) for algorithm details
- Check [API Reference](../api/index.md) for method signatures
- See [Testing](../testing/index.md) for testing your implementations
