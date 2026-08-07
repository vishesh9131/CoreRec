# Utilities

CoreRec provides a comprehensive set of utility functions and tools to support the recommendation workflow.

## Overview

Utilities in CoreRec are organized into several categories:

- **Evaluation Metrics**: Measure model performance
- **Visualization**: Visualize graphs and results
- **Serialization**: Save and load models
- **Configuration**: Manage model configurations
- **Device Management**: Handle CPU/GPU devices

## Evaluation Metrics

Comprehensive metrics for evaluating recommendation quality.

### Rating Prediction Metrics

For explicit feedback (ratings):

```python
from corerec.metrics import rmse, mae, mse

# Calculate RMSE (Root Mean Squared Error)
rmse_score = rmse(true_ratings, predicted_ratings)
print(f"RMSE: {rmse_score:.4f}")

# Calculate MAE (Mean Absolute Error)
mae_score = mae(true_ratings, predicted_ratings)
print(f"MAE: {mae_score:.4f}")

# Calculate MSE (Mean Squared Error)
mse_score = mse(true_ratings, predicted_ratings)
print(f"MSE: {mse_score:.4f}")
```

### Ranking Metrics

For implicit feedback (clicks, views):

```python
from corerec.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    hit_rate_at_k
)

# Precision@K
precision = precision_at_k(true_items, recommended_items, k=10)
print(f"Precision@10: {precision:.4f}")

# Recall@K
recall = recall_at_k(true_items, recommended_items, k=10)
print(f"Recall@10: {recall:.4f}")

# NDCG@K (Normalized Discounted Cumulative Gain)
ndcg = ndcg_at_k(true_items, recommended_items, k=10)
print(f"NDCG@10: {ndcg:.4f}")

# MAP@K (Mean Average Precision)
map_score = map_at_k(true_items, recommended_items, k=10)
print(f"MAP@10: {map_score:.4f}")

# Hit Rate@K
hit_rate = hit_rate_at_k(true_items, recommended_items, k=10)
print(f"Hit Rate@10: {hit_rate:.4f}")
```

### Diversity Metrics

Measure recommendation diversity:

```python
from corerec.evaluation import (
    intra_list_similarity,
    coverage,
    diversity
)

# Intra-list similarity (lower is more diverse)
ils = intra_list_similarity(recommendations, item_features)
print(f"Intra-list Similarity: {ils:.4f}")

# Catalog coverage
cov = coverage(recommendations, total_items)
print(f"Coverage: {cov:.4f}")

# Diversity
div = diversity(recommendations)
print(f"Diversity: {div:.4f}")
```

[**→ Learn more about Evaluation Metrics**](evaluation-metrics.md)

## Visualization

Visualize recommendation systems and results.

### Graph Visualization (VishGraphs)

```python
import corerec.vish_graphs as vg

# Generate random graph
graph_file = vg.generate_random_graph(100, "graph.csv")

# Read adjacency matrix
adj_matrix = vg.bipartite_matrix_maker(graph_file)

# 2D visualization
vg.draw_graph(
    adj_matrix,
    top_nodes=[1, 2, 3, 4, 5],
    recommended_nodes=[10, 11, 12],
    node_labels={1: 'User A', 10: 'Item X'}
)

# 3D visualization
vg.draw_graph_3d(
    adj_matrix,
    top_nodes=[1, 2, 3, 4, 5],
    recommended_nodes=[10, 11, 12]
)

# Bipartite graph visualization
vg.show_bipartite_relationship(adj_matrix)
```

### Training Visualization

```python
import matplotlib.pyplot as plt

# Plot training history
history = model.history

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

# Metrics plot
plt.subplot(1, 2, 2)
plt.plot(history['train_ndcg'], label='Train NDCG')
plt.plot(history['val_ndcg'], label='Val NDCG')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.legend()
plt.title('NDCG Score')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
```

### Embedding Visualization

```python
from corerec.visualization import plot_embeddings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get embeddings from model
user_embeddings = model.get_user_embeddings()
item_embeddings = model.get_item_embeddings()

# Reduce to 2D with PCA
pca = PCA(n_components=2)
user_2d = pca.fit_transform(user_embeddings)
item_2d = pca.fit_transform(item_embeddings)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(user_2d[:, 0], user_2d[:, 1], alpha=0.5, label='Users')
plt.scatter(item_2d[:, 0], item_2d[:, 1], alpha=0.5, label='Items')
plt.legend()
plt.title('User and Item Embeddings')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

[**→ Learn more about Visualization**](visualization.md)

## Serialization

Save and load models efficiently.

### Basic Serialization

```python
from corerec.serialization import ModelSerializer

# Initialize serializer
serializer = ModelSerializer()

# Save model
serializer.save(
    model,
    'models/my_model.pkl',
    metadata={'version': '1.0', 'date': '2024-01-01'}
)

# Load model
loaded_model = serializer.load('models/my_model.pkl')

# Get metadata
metadata = serializer.get_metadata('models/my_model.pkl')
print(metadata)
```

### Format-Specific Serialization

```python
# Save as pickle (default)
model.save('model.pkl', format='pickle')

# Save as PyTorch checkpoint
model.save('model.pth', format='torch')

# Save as JSON (metadata only)
model.save('model.json', format='json')

# Save as ONNX (for deployment)
model.save('model.onnx', format='onnx')
```

### Versioned Serialization

```python
from corerec.serialization import VersionedSerializer

serializer = VersionedSerializer(base_path='models/')

# Save with version
serializer.save(model, version='v1.0.0')

# Load latest version
model = serializer.load_latest()

# Load specific version
model = serializer.load(version='v1.0.0')

# List all versions
versions = serializer.list_versions()
print(f"Available versions: {versions}")
```

[**→ Learn more about Serialization**](serialization.md)

## Configuration Management

Manage model configurations easily.

### YAML Configuration

```python
from corerec.config import ConfigManager

# Load from YAML
config = ConfigManager.from_yaml('config.yaml')

# Access nested config
print(config.model.embedding_dim)
print(config.training.batch_size)

# Save config
config.to_yaml('config_backup.yaml')
```

Example `config.yaml`:

```yaml
model:
  name: DCN
  embedding_dim: 64
  num_cross_layers: 3
  deep_layers: [128, 64, 32]
  dropout: 0.2

training:
  epochs: 20
  batch_size: 256
  learning_rate: 0.001
  device: cuda

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
```

### JSON Configuration

```python
# Load from JSON
config = ConfigManager.from_json('config.json')

# Convert to dict
config_dict = config.to_dict()

# Update config
config.update({'model.embedding_dim': 128})
```

### Environment Variables

```python
import os

# Use environment variables
config = ConfigManager.from_env(
    prefix='COREREC_',
    defaults={'model.embedding_dim': 64}
)

# Set environment variable
os.environ['COREREC_MODEL_EMBEDDING_DIM'] = '128'
```

[**→ Learn more about Configuration**](configuration.md)

## Device Management

Models take a `device` argument and otherwise follow normal PyTorch conventions;
CoreRec has no device-manager abstraction of its own.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = TwoTower(embedding_dim=64, device=device)
model.fit(user_ids, item_ids, ratings)
```

### Multiple GPUs

```python
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs available")
    model.model = torch.nn.DataParallel(model.model)
```

### Memory

```python
if torch.cuda.is_available():
    print(f"allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.empty_cache()
```

## Example Data

CoreRec does not bundle a sample-data helper. The example script generates its
own interactions, which is the pattern to copy for quick experiments:

```python
from examples.train_and_serve import build_interactions

users, items, ratings, _, _ = build_interactions(seed=0)
```

For a real dataset, `corerec[datasets]` provides MovieLens via `cr_learn`:

```python
from cr_learn import ml_1m          # pip install corerec[datasets]

ratings_df = ml_1m.load()["ratings"]
user_ids = ratings_df["user_id"].values
item_ids = ratings_df["movie_id"].values
ratings = ratings_df["rating"].values
```

## Logging and Debugging

### Setup Logging

```python
import logging
from corerec.utils import get_logger

# Setup logger
logger = get_logger(
    name='corerec',
    log_file='corerec.log',
    level=logging.INFO,
)

# Use logger
logger.info("Training started")
logger.debug("Batch size: 256")
logger.warning("Low GPU memory")
logger.error("Training failed")
```

### Debug Mode

```python
# Enable debug mode
model = DCN(verbose=True, debug=True)

# This will print:
# - Architecture summary
# - Training progress
# - Memory usage
# - Timing information
```

## Performance Profiling

CoreRec does not ship its own profiler. Use the standard library and the
existing Python tooling, which is better maintained than anything a
recommender library should be writing:

```python
import cProfile
import pstats

from corerec.utils import print_system_info

print_system_info()   # CPU / BLAS / torch build, useful when comparing timings

profiler = cProfile.Profile()
profiler.enable()
model.fit(user_ids, item_ids, ratings)
profiler.disable()

pstats.Stats(profiler).sort_stats("cumtime").print_stats(20)
```

### Memory Profiling

`pip install memory-profiler`, then:

```python
from memory_profiler import profile

@profile
def train_model():
    model = DCN(embedding_dim=64)
    model.fit(user_ids, item_ids, ratings)

train_model()
```

## Utility Functions

### Data Processing

```python
from corerec.utils import (
    train_test_split,
    normalize_ratings,
    create_interaction_matrix
)

# Train/test split
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

# Normalize ratings
normalized_ratings = normalize_ratings(
    ratings,
    method='min-max'  # or 'z-score'
)

# Create interaction matrix
interaction_matrix = create_interaction_matrix(
    user_ids,
    item_ids,
    ratings
)
```

### Negative Sampling

```python
from corerec.utils import negative_sampling

# Sample negative items
negative_items = negative_sampling(
    user_id=123,
    positive_items=[1, 2, 3],
    all_items=list(range(1, 1000)),
    num_negatives=10
)
```

### Batch Processing

```python
from corerec.utils import batch_iterator

# Iterate in batches
for batch in batch_iterator(data, batch_size=256):
    # Process batch
    model.train_step(batch)
```

## Next Steps

- Explore detailed utility documentation:
  - [Evaluation Metrics](evaluation-metrics.md)
  - [Visualization](visualization.md)
  - [Serialization](serialization.md)
  - [Configuration](configuration.md)
  - [Device Management](device-management.md)
- See [Examples](../examples/index.md) for usage examples
- Read [Best Practices](../user-guide/best-practices.md) for optimization tips


