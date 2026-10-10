# Utilities

CoreRec provides a comprehensive set of utility functions and tools to support the recommendation workflow.

## Overview

Utilities in CoreRec are organized into several categories:

- **Evaluation Metrics**: Measure model performance (`corerec.evaluation`)
- **Visualization**: Plot training curves and embeddings (with matplotlib)
- **Serialization**: Save and load models (`model.save` / `load`, `corerec.serialization`)
- **Configuration**: Load YAML/JSON configs (`corerec.utils.load_config`)
- **Device Management**: Pick CUDA, Apple MPS or CPU (`corerec.device`)

The examples on this page share this setup:

```python
import numpy as np
from corerec.engines import ALS

rng = np.random.default_rng(0)
users = rng.integers(0, 50, 2000).tolist()
items = rng.integers(0, 200, 2000).tolist()
train = list(zip(users, items))[:1600]
test = list(zip(users, items))[1600:]

model = ALS(factors=16, iterations=10)
model.fit([u for u, _ in train], [i for _, i in train])
```

## Evaluation Metrics

Comprehensive metrics for evaluating recommendation quality.

### Rating Prediction Metrics

For explicit feedback (ratings). CoreRec's models rank items rather than
predict ratings, so there are no RMSE/MAE helpers; compute them from
`predict()` with numpy:

```python
true_ratings = np.array([4.0, 3.0, 5.0])
predicted_ratings = np.array([3.5, 3.0, 4.5])  # e.g. [model.predict(u, i) for u, i in pairs]

rmse_score = np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))
mae_score = np.mean(np.abs(true_ratings - predicted_ratings))
mse_score = np.mean((true_ratings - predicted_ratings) ** 2)
print(f"RMSE: {rmse_score:.4f}  MAE: {mae_score:.4f}  MSE: {mse_score:.4f}")
```

### Ranking Metrics

For implicit feedback (clicks, views). `evaluate()` runs them all through
`model.recommend()`, excluding each user's training items:

```python
from corerec.evaluation import evaluate

print(evaluate(model, test, train_interactions=train, k=10))
# {'NDCG@10': ..., 'MAP@10': ..., 'MRR@10': ..., 'Precision@10': ...,
#  'Recall@10': ..., 'HitRate@10': ..., 'n_users': ..., 'n_errors': 0}
```

For one list at a time, use `RankingMetrics` directly:

```python
from corerec.evaluation import RankingMetrics as M

recommended_items = model.recommend(users[0], top_k=10)
true_items = [i for u, i in test if u == users[0]]

print(f"Precision@10: {M.precision_at_k(recommended_items, true_items, k=10):.4f}")
print(f"Recall@10:    {M.recall_at_k(recommended_items, true_items, k=10):.4f}")
print(f"NDCG@10:      {M.ndcg_at_k(recommended_items, true_items, k=10):.4f}")
print(f"MAP@10:       {M.map_at_k(recommended_items, true_items, k=10):.4f}")
print(f"Hit Rate@10:  {M.hit_rate_at_k(recommended_items, true_items, k=10):.4f}")
```

### Diversity Metrics

Measure recommendation diversity:

```python
from collections import Counter
from corerec.evaluation import DiversityMetrics as D

recommendations = list(model.batch_recommend(sorted(set(users)), top_k=10).values())

# Catalog coverage: share of all items that were recommended to anyone
print(f"Coverage: {D.coverage(recommendations, total_items=len(set(items))):.4f}")

# Gini over how often each item was recommended (0 = even, 1 = concentrated)
counts = Counter(i for recs in recommendations for i in recs)
print(f"Gini: {D.gini_coefficient(counts):.4f}")

# Share of unique items within each list, averaged
print(f"Intra-list diversity: {D.intra_list_diversity(recommendations):.4f}")
```


## Visualization

Visualize recommendation systems and results.

### Graph Visualization (VishGraphs)

The `vish_graphs` module was removed in 0.7. To draw the user-item graph, build
it from your interactions with a graph library such as `networkx`.

### Training Visualization

Models built with `corerec.nn.Recommender` record the training loss per epoch
in `history_`, and validation NDCG@10 in `val_history_` when you pass
`validation=`:

```python
import matplotlib
matplotlib.use("Agg")  # no window needed; drop this line in a notebook
import matplotlib.pyplot as plt
import pandas as pd
from corerec.nn import Recommender, MatrixFactorization

df = pd.DataFrame(train, columns=["user_id", "item_id"])
val = pd.DataFrame(test, columns=["user_id", "item_id"])
rec = Recommender(MatrixFactorization, {"dim": 16}, epochs=10).fit(df, validation=val)

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(rec.history_, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

# Metrics plot
plt.subplot(1, 2, 2)
plt.plot(rec.val_history_, label='Val NDCG@10')
plt.xlabel('Epoch')
plt.ylabel('NDCG')
plt.legend()
plt.title('NDCG Score')

plt.tight_layout()
plt.savefig('training_history.png')
```

### Embedding Visualization

```python
from sklearn.decomposition import PCA

# Get embeddings from the model (here, the MatrixFactorization template above)
user_embeddings = rec.model.users.weight.detach().cpu().numpy()
item_embeddings = rec.model.items.weight.detach().cpu().numpy()[1:]  # row 0 is padding

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
plt.savefig('embeddings.png')
```


## Serialization

Save and load models efficiently.

### Basic Serialization

Every model has `save()` and a `load()` classmethod:

```python
model.save('models/my_model')
loaded_model = ALS.load('models/my_model')
assert loaded_model.recommend(users[0], top_k=5) == model.recommend(users[0], top_k=5)
```

To load a file without knowing which class wrote it, use
`corerec.serving.ModelLoader`. For plain Python objects such as metadata,
`corerec.serialization` has `save_to_file` / `load_from_file`:

```python
from corerec.serialization import save_to_file, load_from_file

save_to_file({'version': '1.0', 'trained_on': '2026-10-01'}, 'models/my_model_meta.json')
print(load_from_file('models/my_model_meta.json'))
```

### Format-Specific Serialization

`save()` takes no `format=` argument: each model writes its own format (the
classic models a safe npz + JSON bundle, the PyTorch models a checkpoint). For
deployment without Python, export the PyTorch models to ONNX:

```python
from corerec.engines import TwoTower
from corerec.export import to_onnx

tt = TwoTower(embedding_dim=16, epochs=2, verbose=False).fit(users, items)
to_onnx(tt, 'model.onnx')  # needs pip install "corerec[onnx]"
```

### Versioned Serialization

There is no versioning class; put the version in the path and keep the old
directories:

```python
model.save('models/als/v1.0.0')
model = ALS.load('models/als/v1.0.0')
```


## Configuration Management

Manage model configurations easily.

### YAML Configuration

`load_config` reads YAML or JSON into a plain dict, so a model is built with
`**`:

```python
from corerec.engines import DCN
from corerec.utils import load_config

with open('config.yaml', 'w') as f:  # the example config below
    f.write("model:\n  embedding_dim: 64\n  num_cross_layers: 3\n  deep_layers: [128, 64, 32]\n"
            "  dropout: 0.2\ntraining:\n  epochs: 20\n  batch_size: 256\n"
            "  learning_rate: 0.001\n  device: auto\n")

config = load_config('config.yaml')
print(config['model']['embedding_dim'])
print(config['training']['batch_size'])

dcn = DCN(**config['model'], **config['training'])
```

Example `config.yaml`:

```yaml
model:
  embedding_dim: 64
  num_cross_layers: 3
  deep_layers: [128, 64, 32]
  dropout: 0.2

training:
  epochs: 20
  batch_size: 256
  learning_rate: 0.001
  device: auto
```

### JSON Configuration

```python
import json
from corerec.utils import merge_configs

with open('config.json', 'w') as f:
    json.dump(config, f)

# Load from JSON
config = load_config('config.json')

# Update config: nested keys are merged, not replaced
config = merge_configs(config, {'model': {'embedding_dim': 128}})
print(config['model'])
```

### Environment Variables

There is no environment-variable loader; read overrides yourself and merge
them in:

```python
import os

os.environ['COREREC_EMBEDDING_DIM'] = '128'
override = {'model': {'embedding_dim': int(os.environ.get('COREREC_EMBEDDING_DIM', 64))}}
config = merge_configs(config, override)
```


## Device Management

Handle CPU/GPU devices efficiently.

### Basic Device Management

```python
import torch
from corerec.device import resolve_device, mps_available

# 'auto' picks CUDA, then Apple MPS, then CPU
device = resolve_device('auto')
print(f"Using device: {device}")

# Check availability
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
print(f"Apple MPS available: {mps_available()}")

# PyTorch models take the device directly
tt = TwoTower(embedding_dim=16, epochs=1, device='auto', verbose=False)
```

### Multi-GPU Support

The built-in models train on one device. To pick a specific GPU, pass it as
the device:

```python
# Use specific GPU (only on a machine with CUDA)
if torch.cuda.is_available():
    dcn = DCN(device='cuda:0')
```

### Memory Management

These are plain PyTorch calls:

```python
if torch.cuda.is_available():
    # Monitor GPU memory
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Clear cache
    torch.cuda.empty_cache()

    # Set memory limit (fraction of the device's memory)
    torch.cuda.set_per_process_memory_fraction(0.5)
```


## Example Data

Generate sample data for testing.

Real datasets come from `cr_learn` (`pip install cr_learn`), for example
`from cr_learn import ml_1m` and `ml_1m.load()`. For a quick synthetic set:

```python
num_users, num_items, num_interactions = 1000, 500, 10000
rng = np.random.default_rng(42)

user_ids = rng.integers(0, num_users, num_interactions).tolist()
item_ids = rng.integers(0, num_items, num_interactions).tolist()
ratings = rng.integers(1, 6, num_interactions).astype(float).tolist()  # rating_range (1, 5)
```

## Logging and Debugging

### Setup Logging

```python
import logging
from corerec.utils import setup_logging, get_logger

# Setup logging (console, plus a file if log_file is given)
setup_logging(log_file='corerec.log', console_level=logging.INFO)

# Use logger
logger = get_logger('corerec')
logger.info("Training started")
logger.debug("Batch size: 256")
logger.warning("Low GPU memory")
logger.error("Training failed")
```

### Debug Mode

```python
# Print training progress
dcn = DCN(epochs=2, verbose=True)

# Versions, BLAS backend and devices, for bug reports
from corerec.utils import print_system_info
print_system_info()
```

## Performance Profiling

### Profile Training

CoreRec has no profiler of its own; the standard library's works:

```python
import cProfile
import pstats

# Profile training
with cProfile.Profile() as profiler:
    ALS(factors=16).fit(user_ids, item_ids, ratings)

# Print statistics
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)

# Save profile
stats.dump_stats('profile.prof')
```

### Memory Profiling

```python
import tracemalloc

def train_model():
    m = ALS(factors=16)
    m.fit(user_ids, item_ids, ratings)

tracemalloc.start()
train_model()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory: {peak / 1e6:.1f} MB")
```

## Utility Functions

### Data Processing

```python
import pandas as pd
from scipy.sparse import csr_matrix
from corerec.utils import validate_fit_inputs

data = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'rating': ratings})

# Check inputs the way fit() does (raises ValidationError on bad input)
validate_fit_inputs(user_ids, item_ids, ratings)

# Train/test split
test_data = data.sample(frac=0.2, random_state=42)
train_data = data.drop(test_data.index)

# Normalize ratings (min-max)
r = data['rating']
normalized_ratings = (r - r.min()) / (r.max() - r.min())

# Create interaction matrix
interaction_matrix = csr_matrix((data['rating'], (data['user_id'], data['item_id'])))
```

### Negative Sampling

Each deep model samples negatives inside `fit()` (see `num_negatives` on
`DCN`, `TwoTower`, `corerec.nn.Recommender`). To draw them yourself:

```python
positive_items = {1, 2, 3}
candidates = np.setdiff1d(np.arange(1, 1000), list(positive_items))
negative_items = rng.choice(candidates, size=10, replace=False)
```

### Batch Processing

```python
# Recommend for many users at once: {user_id: [items]}
recs = model.batch_recommend(sorted(set(users))[:256], top_k=10)

# Score many (user, item) pairs
scores = model.batch_predict(list(zip(users[:256], items[:256])))
```

## Next Steps

- Explore detailed utility documentation:
- See [Examples](../examples/index.md) for usage examples
