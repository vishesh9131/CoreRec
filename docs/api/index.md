# API Reference

Welcome to the CoreRec API Reference. This section provides detailed documentation for all classes, methods, and functions in the CoreRec library.

## Core API Modules

### Base Classes

- **[BaseRecommender](base-recommender.md)**: Unified interface for all recommendation models
- **[ModelInterface](model-interface.md)**: Interface for model components
- **[PredictorInterface](predictor-interface.md)**: Interface for prediction modules

### Quick Links

| Module | Description |
|--------|-------------|
| `corerec.api` | Core API interfaces and base classes |
| `corerec.engines` | Recommendation engines and algorithms |
| `corerec.core` | Core components (towers, encoders, losses) |
| `corerec.training` | Training pipeline and utilities |
| `corerec.data` | Data loading and preprocessing |
| `corerec.evaluation` | Evaluation metrics and tools |
| `corerec.utils` | Utility functions and helpers |

## API Organization

```
corerec/
├── api/                          # Core API interfaces
│   ├── base_recommender.py       # Base recommender class
│   ├── model_interface.py        # Model interface
│   └── predictor_interface.py    # Predictor interface
│
├── engines/                      # Recommendation engines
│   ├── unionizedFilterEngine/    # Collaborative filtering
│   ├── contentFilterEngine/      # Content-based filtering
│   ├── dcn.py                    # Deep & Cross Network
│   ├── deepfm.py                 # DeepFM
│   ├── gnnrec.py                 # GNN Recommender
│   ├── mind.py                   # MIND
│   ├── nasrec.py                 # NASRec
│   └── sasrec.py                 # SASRec
│
├── core/                         # Core components
│   ├── towers.py                 # Tower modules
│   ├── encoders.py               # Feature encoders
│   ├── losses.py                 # Loss functions
│   ├── embedding_tables/         # Embedding tables
│   └── base_model.py             # Base model class
│
├── training/                     # Training utilities
│   ├── trainer.py                # Model trainer
│   ├── callbacks.py              # Training callbacks
│   └── optimizers.py             # Optimizers
│
├── data/                         # Data handling
│   ├── data_loader.py            # Data loading
│   └── data_processor.py         # Data processing
│
├── evaluation/                   # Evaluation tools
│   ├── evaluator.py              # Model evaluator
│   └── metrics.py                # Evaluation metrics
│
└── utils/                        # Utilities
    ├── config.py                 # Configuration
    ├── serialization.py          # Model serialization
    └── device.py                 # Device management
```

## Common Patterns

### Model Initialization

All models follow the same initialization pattern:

```python
from corerec.engines.dcn import DCN

model = DCN(
    embedding_dim=64,        # Model-specific parameters
    num_cross_layers=3,
    deep_layers=[128, 64],
    epochs=20,               # Training parameters
    batch_size=256,
    learning_rate=0.001,
    device='cuda'            # Device configuration
)
```

### Training

All models use the `fit()` method:

```python
model.fit(
    user_ids,               # Required: user IDs
    item_ids,               # Required: item IDs
    ratings,                # Required: ratings/interactions
    validation_data=None,   # Optional: validation data
    callbacks=None,         # Optional: training callbacks
    verbose=True            # Optional: verbose output
)
```

### Prediction

All models support `predict()` and `recommend()`:

```python
# Predict score for specific user-item pair
score = model.predict(user_id=123, item_id=456)

# Get top-K recommendations
recs = model.recommend(
    user_id=123,
    top_k=10,
    exclude_items=[1, 2, 3],  # Optional: exclude seen items
    return_scores=True         # Optional: return scores
)
```

### Batch Operations

All models support batch processing:

```python
# Batch predictions
scores = model.batch_predict([
    (user1, item1),
    (user2, item2),
    (user3, item3)
])

# Batch recommendations
recs = model.batch_recommend(
    user_ids=[user1, user2, user3],
    top_k=10
)
```

### Model Persistence

All models support saving and loading:

```python
# Save model
model.save('model.pkl', format='pickle')
model.save('model.pth', format='torch')
model.save('model.json', format='json')

# Load model
model = DCN.load('model.pkl')
```

## API Conventions

### Naming Conventions

- **Classes**: PascalCase (e.g., `BaseRecommender`, `MLPTower`)
- **Functions**: snake_case (e.g., `load_data`, `compute_metrics`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_BATCH_SIZE`)
- **Private methods**: Prefix with `_` (e.g., `_build_network`)

### Parameter Conventions

Common parameter names across all models:

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` / `user_ids` | int/List[int] | User identifier(s) |
| `item_id` / `item_ids` | int/List[int] | Item identifier(s) |
| `rating` / `ratings` | float/List[float] | Rating values |
| `top_k` / `top_n` | int | Number of recommendations |
| `embedding_dim` | int | Embedding dimension |
| `hidden_dim` / `hidden_dims` | int/List[int] | Hidden layer sizes |
| `epochs` / `num_epochs` | int | Training epochs |
| `batch_size` | int | Batch size |
| `learning_rate` / `lr` | float | Learning rate |
| `dropout` | float | Dropout rate |
| `device` | str | Device ('cpu', 'cuda') |

### Return Value Conventions

- **Recommendations**: List of item IDs or List of (item_id, score) tuples
- **Predictions**: Float score or List of scores
- **Training**: Returns self for method chaining
- **Evaluation**: Dictionary of metric names to values

## Type Hints

CoreRec uses type hints throughout for better IDE support:

```python
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import torch

def recommend(
    self,
    user_id: int,
    top_k: int = 10,
    exclude_items: Optional[List[int]] = None,
    return_scores: bool = False
) -> Union[List[int], List[Tuple[int, float]]]:
    """
    Generate recommendations for a user.
    
    Args:
        user_id: User identifier
        top_k: Number of recommendations
        exclude_items: Items to exclude
        return_scores: Whether to return scores
    
    Returns:
        List of item IDs or (item_id, score) tuples
    """
    pass
```

## Error Handling

CoreRec uses standard Python exceptions:

```python
try:
    model = DCN.load('nonexistent.pkl')
except FileNotFoundError:
    print("Model file not found")

try:
    model.fit([], [], [])  # Empty data
except ValueError as e:
    print(f"Invalid data: {e}")

try:
    model.recommend(user_id=999999)  # Unknown user
except KeyError:
    print("User not found")
```

## Logging

CoreRec uses Python's logging module:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Model will log training progress
model = DCN(verbose=True)
model.fit(user_ids, item_ids, ratings)
```

## Configuration

Models can be configured via dictionaries or config objects:

```python
# Dictionary configuration
config = {
    'embedding_dim': 64,
    'num_cross_layers': 3,
    'deep_layers': [128, 64, 32]
}
model = DCN(**config)

# Config object
from corerec.config import ModelConfig

config = ModelConfig.from_yaml('config.yaml')
model = DCN(**config.to_dict())
```

## Next Steps

- Read the [BaseRecommender](base-recommender.md) documentation
- Explore [Engines](../engines/index.md) for algorithm-specific APIs
- Check [Examples](../examples/index.md) for usage patterns
- See [Core Components](../core/index.md) for building blocks


