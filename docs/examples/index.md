# Examples

This section provides comprehensive examples for using CoreRec in various scenarios. All examples are runnable and include complete code.

## Quick Start Examples

Perfect for getting started quickly:

- **[Engines Quickstart](quickstart/engines-quickstart.md)** - Quick start with deep learning engines
- **[Unionized Filter Quickstart](quickstart/unionized-quickstart.md)** - Collaborative filtering examples
- **[Content Filter Quickstart](quickstart/content-filter-quickstart.md)** - Content-based filtering examples

## Engine-Specific Examples

### Deep Learning Models

Production-ready deep learning models:

- **[DCN Example](engines/dcn-example.md)** - Deep & Cross Network for feature interactions
- **[DeepFM Example](engines/deepfm-example.md)** - Deep Factorization Machines
- **[GNNRec Example](engines/gnnrec-example.md)** - Graph Neural Networks
- **[MIND Example](engines/mind-example.md)** - Multi-Interest Network with Dynamic Routing
- **[NASRec Example](engines/nasrec-example.md)** - Neural Architecture Search
- **[SASRec Example](engines/sasrec-example.md)** - Self-Attentive Sequential Recommendations

### Unionized Filter Engine

Collaborative filtering algorithms:

- **[FastRecommender Example](unionized/fast-example.md)** - FastAI-style embedding recommender
- **[SAR Example](unionized/sar-example.md)** - Smart Adaptive Recommendations
- **[RBM Example](unionized/rbm-example.md)** - Restricted Boltzmann Machines
- **[RLRMC Example](unionized/rlrmc-example.md)** - Riemannian Low-Rank Matrix Completion
- **[GeoMLC Example](unionized/geomlc-example.md)** - Geometric Matrix Learning

### Content Filter Engine

Content-based filtering:

- **[TF-IDF Example](content-filter/tfidf-example.md)** - Text-based recommendations

## Advanced Examples

Real-world use cases and complex scenarios:

- **[Instagram Reels Recommendation](advanced/instagram-reels.md)** - Build an Instagram Reels-style recommendation system
- **[YouTube MoE](advanced/youtube-moe.md)** - Mixture of Experts for video recommendations
- **[DIEN Example](advanced/dien-example.md)** - Deep Interest Evolution Network

## Demo Frontends

Interactive web interfaces:

- **[ImShow Connector](frontends/imshow-connector.md)** - Plug-and-play web interface for recommendations

## By Use Case

### E-commerce

```python
from corerec.engines.deepfm import DeepFM

# Product recommendations
model = DeepFM(embedding_dim=128, hidden_layers=[256, 128, 64])
model.fit(customer_ids, product_ids, purchase_amounts)

# Get personalized product recommendations
recommendations = model.recommend(customer_id=12345, top_n=10)
```

### Movie Recommendations

```python
from corerec.engines.sasrec import SASRec
from scipy.sparse import csr_matrix

# Sequential movie recommendations
model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=4,
    num_epochs=50,
    max_seq_length=20
)

# Create interaction matrix
interaction_matrix = csr_matrix(user_item_matrix)
model.fit(interaction_matrix, user_ids, movie_ids)

# Get next movie recommendations
next_movies = model.recommend(user_id=456, top_n=5)
```

### Music Streaming

```python
from corerec.engines.mind import MIND

# Multi-interest music recommendations
model = MIND(
    embedding_dim=64,
    num_interests=4,  # Capture diverse music tastes
    epochs=30
)

model.fit(user_ids, song_ids, listen_times)

# Get diverse music recommendations
recommendations = model.recommend(user_id=789, top_n=20)
```

### News Articles

```python
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender
import pandas as pd

# Content-based news recommendations
articles = pd.DataFrame({
    'article_id': [1, 2, 3, 4, 5],
    'title': [...],
    'content': [...]
})

model = TFIDFRecommender(feature_column='content')
model.fit(articles)

# Get similar articles
similar_articles = model.recommend_similar(article_id=1, top_k=10)
```

### Social Networks

```python
from corerec.engines.gnnrec import GNNRec

# Friend/connection recommendations
model = GNNRec(
    embedding_dim=64,
    num_gnn_layers=3,
    epochs=50
)

# User-user interaction data
model.fit(user_ids_from, user_ids_to, interaction_weights)

# Recommend new connections
friend_suggestions = model.recommend(user_id=123, top_n=10)
```

## Complete End-to-End Examples

### Example 1: Movie Recommendation System

```python
import pandas as pd
from corerec.engines.dcn import DCN
from sklearn.model_selection import train_test_split
from corerec.evaluation import evaluate_model

# 1. Load data
ratings = pd.read_csv('movielens_ratings.csv')
user_ids = ratings['user_id'].tolist()
movie_ids = ratings['movie_id'].tolist()
ratings_values = ratings['rating'].tolist()

# 2. Train/test split
train_users, test_users, train_movies, test_movies, train_ratings, test_ratings = \
    train_test_split(user_ids, movie_ids, ratings_values, test_size=0.2)

# 3. Initialize and train model
model = DCN(
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64, 32],
    epochs=20,
    batch_size=256,
    device='cuda'
)

model.fit(train_users, train_movies, train_ratings)

# 4. Evaluate
metrics = evaluate_model(
    model,
    test_users,
    test_movies,
    test_ratings,
    metrics=['rmse', 'mae', 'precision@10', 'ndcg@10']
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Precision@10: {metrics['precision@10']:.4f}")

# 5. Get recommendations
user_id = test_users[0]
recommendations = model.recommend(user_id=user_id, top_k=10)
print(f"Top 10 movies for user {user_id}: {recommendations}")

# 6. Save model
model.save('models/movie_recommender.pkl')
```

### Example 2: E-commerce Product Recommendations

```python
from corerec.engines.deepfm import DeepFM
import pandas as pd

# 1. Load purchase data
purchases = pd.read_csv('purchase_history.csv')

# 2. Prepare data
customer_ids = purchases['customer_id'].tolist()
product_ids = purchases['product_id'].tolist()
purchase_amounts = purchases['amount'].tolist()

# 3. Initialize DeepFM
model = DeepFM(
    embedding_dim=128,
    hidden_layers=[256, 128, 64],
    epochs=30,
    batch_size=512,
    learning_rate=0.001
)

# 4. Train model
model.fit(
    customer_ids,
    product_ids,
    purchase_amounts,
    validation_split=0.2
)

# 5. Batch recommendations for all customers
all_customer_ids = purchases['customer_id'].unique().tolist()
batch_recommendations = model.batch_recommend(
    user_ids=all_customer_ids,
    top_n=10
)

# 6. Export recommendations
recommendations_df = pd.DataFrame([
    {'customer_id': uid, 'recommended_products': recs}
    for uid, recs in batch_recommendations.items()
])

recommendations_df.to_csv('product_recommendations.csv', index=False)
```

### Example 3: Sequential Music Recommendations

```python
from corerec.engines.sasrec import SASRec
from scipy.sparse import csr_matrix
import numpy as np

# 1. Load listening history
listens = pd.read_csv('listening_history.csv')

# 2. Create user-item interaction matrix
user_ids = listens['user_id'].unique()
song_ids = listens['song_id'].unique()

# Create sparse matrix
interaction_matrix = csr_matrix(
    (listens['play_count'], (listens['user_id'], listens['song_id']))
)

# 3. Initialize SASRec
model = SASRec(
    hidden_units=64,
    num_blocks=2,
    num_heads=4,
    num_epochs=50,
    batch_size=128,
    max_seq_length=20
)

# 4. Train model
model.fit(interaction_matrix, user_ids.tolist(), song_ids.tolist())

# 5. Get next song recommendations
user_id = user_ids[0]
next_songs = model.recommend(user_id=user_id, top_n=20)
print(f"Next 20 songs for user {user_id}: {next_songs}")

# 6. Visualize training progress
import matplotlib.pyplot as plt

history = model.history
plt.plot(history['train_loss'], label='Train Loss')
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
df = pd.read_csv('interactions.csv')

# Extract columns
user_ids = df['user_id'].tolist()
item_ids = df['item_id'].tolist()
ratings = df['rating'].tolist()

# Optional: timestamps
timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else None
```

### Example: Creating Synthetic Data

```python
import numpy as np

# Generate synthetic data for testing
num_users = 1000
num_items = 500
num_interactions = 10000

user_ids = np.random.randint(0, num_users, num_interactions).tolist()
item_ids = np.random.randint(0, num_items, num_interactions).tolist()
ratings = np.random.uniform(1, 5, num_interactions).tolist()
```

### Example: Using Sample Data

```python
from corerec.utils.example_data import get_sample_data

# Load built-in sample data
data = get_sample_data('netflix')

user_ids = data['user_ids']
item_ids = data['item_ids']
ratings = data['ratings']
```

## Evaluation Examples

### Example: Complete Evaluation

```python
from corerec.evaluation import Evaluator
from corerec.metrics import (
    rmse, mae, precision_at_k, recall_at_k, ndcg_at_k
)

# Create evaluator
evaluator = Evaluator(
    metrics=['rmse', 'mae', 'precision@10', 'recall@10', 'ndcg@10']
)

# Evaluate model
results = evaluator.evaluate(model, test_data)
print(results)

# Manual metric calculation
predictions = model.batch_predict(test_pairs)
rmse_score = rmse(test_ratings, predictions)
print(f"RMSE: {rmse_score:.4f}")
```

## Visualization Examples

### Example: Graph Visualization

```python
import corerec.vish_graphs as vg

# Get interaction matrix from model
adj_matrix = model.get_interaction_matrix()

# 2D visualization
vg.draw_graph(
    adj_matrix,
    top_nodes=[1, 2, 3, 4, 5],
    node_labels={1: 'User A', 2: 'User B', 3: 'Item X'}
)

# 3D visualization
vg.draw_graph_3d(
    adj_matrix,
    top_nodes=[1, 2, 3, 4, 5],
    recommended_nodes=[10, 11, 12]
)
```

## Running Examples

All examples are located in the `examples/` directory of the CoreRec repository:

```bash
# Run engine quickstart
python examples/engines_quickstart.py

# Run specific engine example
python examples/engines_dcn_example.py
python examples/engines_deepfm_example.py

# Run unionized filter examples
python examples/unionized_fast_example.py
python examples/unionized_sar_example.py

# Run content filter examples
python examples/content_filter_tfidf_example.py

# Run advanced examples
python examples/instagram_reels_with_real_data.py
python examples/dien_example.py

# Run all tests
python examples/run_all_algo_tests_example.py
```

## Interactive Examples

Try CoreRec in interactive Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Open example notebooks
examples/notebooks/quickstart.ipynb
examples/notebooks/deep_learning_models.ipynb
examples/notebooks/collaborative_filtering.ipynb
```

## Next Steps

- Explore [Engine Documentation](../engines/index.md) for algorithm details
- Read [User Guide](../user-guide/index.md) for best practices
- Check [API Reference](../api/index.md) for method signatures
- See [Testing](../testing/index.md) for testing your implementations


