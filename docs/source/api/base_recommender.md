# BaseRecommender API

The `BaseRecommender` class is the foundation of all recommendation models in CoreRec. It provides a unified interface that ensures consistency across different algorithm families.

## Overview

```python
from corerec.api.base_recommender import BaseRecommender
```

All recommendation models in CoreRec inherit from `BaseRecommender`, which enforces a standard API for:

- Training models (`fit`)
- Making predictions (`predict`)
- Generating recommendations (`recommend`)
- Saving and loading models (`save`, `load`)
- Batch operations (`batch_predict`, `batch_recommend`)

## Class Definition

```python
class BaseRecommender(ABC):
    """
    Unified base class for ALL recommendation models in CoreRec.
    
    This enforces consistent API across:
    - Collaborative filtering models
    - Content-based models
    - Hybrid models
    - Deep learning models
    - Graph-based models
    """
```

## Initialization

```python
BaseRecommender(
    name: Optional[str] = None,
    trainable: bool = True,
    verbose: bool = False
)
```

**Parameters:**
- `name`: Model name for identification (default: class name)
- `trainable`: Whether the model is trainable (default: True)
- `verbose`: Whether to print training logs (default: False)

## Abstract Methods

These methods **must** be implemented by all subclasses:

### fit()

Train the recommendation model.

```python
@abstractmethod
def fit(self, data: Union[pd.DataFrame, Dict, Any], **kwargs) -> 'BaseRecommender':
    """
    Train the recommendation model.
    
    Args:
        data: Training data (DataFrame, dict, or custom format)
        **kwargs: Additional training parameters
        
    Returns:
        self: For method chaining
    """
```

### predict()

Predict score for a single user-item pair.

```python
@abstractmethod
def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
    """
    Predict score for a single user-item pair.
    
    Args:
        user_id: User identifier
        item_id: Item identifier
        **kwargs: Additional prediction parameters
        
    Returns:
        Predicted score (higher = more relevant)
    """
```

### recommend()

Generate top-K recommendations for a user.

```python
@abstractmethod
def recommend(
    self, 
    user_id: Any, 
    top_k: int = 10,
    exclude_items: Optional[List[Any]] = None,
    **kwargs
) -> List[Any]:
    """
    Generate top-K recommendations for a user.
    
    Args:
        user_id: User identifier
        top_k: Number of recommendations to return
        exclude_items: Items to exclude from recommendations
        **kwargs: Additional recommendation parameters
        
    Returns:
        List of recommended item IDs
    """
```

## Properties

### num_users
Number of users in training data.

### num_items
Number of items in training data.

### total_users
Total number of users (including validation/test).

### total_items
Total number of items (including validation/test).

### uid_map
Mapping of user IDs to indices.

### iid_map
Mapping of item IDs to indices.

### max_rating
Maximum value among rating observations.

### min_rating
Minimum value among rating observations.

### global_mean
Average value over rating observations.

### user_ids
List of all user IDs.

### item_ids
List of all item IDs.

## Utility Methods

### clone()

Create a copy of the model with optional parameter overrides.

```python
def clone(self, new_params: Optional[Dict[str, Any]] = None) -> 'BaseRecommender':
    """
    Clone an instance of the model object.
    
    Args:
        new_params: Optional dict of parameters to override
        
    Returns:
        Cloned model instance
    """
```

### reset_info()

Reset training information (early stopping, best values, etc.).

```python
def reset_info(self) -> None:
    """Reset early stopping and training info."""
```

## Usage Example

```python
from corerec.engines.dcn import DCN

# All models inherit from BaseRecommender
model = DCN(embedding_dim=64, verbose=True)

# Standard API methods
model.fit(user_ids, item_ids, ratings)
score = model.predict(user_id=1, item_id=100)
recommendations = model.recommend(user_id=1, top_k=10)

# Model persistence
model.save('model.pkl')
loaded_model = DCN.load('model.pkl')
```

## See Also

- [Exceptions](exceptions.md) - Error handling
- [Mixins](mixins.md) - Reusable functionality
- [Engines](engines) - Available recommendation engines

