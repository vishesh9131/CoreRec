# BaseRecommender

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

## Class Signature

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
    
    def __init__(self, name: str = "BaseRecommender", verbose: bool = False):
        """
        Initialize base recommender.
        
        Args:
            name: Model name for identification
            verbose: Whether to print training logs
        """
        pass
```

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
        
    Example:
        model.fit(train_data, epochs=10).save('model.pkl')
    """
    pass
```

**Usage Example:**

```python
from corerec.engines.dcn import DCN

model = DCN(embedding_dim=64)

# Simple fit
model.fit(user_ids, item_ids, ratings)

# Fit with validation
model.fit(
    user_ids,
    item_ids,
    ratings,
    validation_data=(val_users, val_items, val_ratings),
    epochs=20
)

# Method chaining
model.fit(train_data, epochs=10).save('model.pkl')
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
        
    Example:
        score = model.predict(user_id=123, item_id=456)
    """
    pass
```

**Usage Example:**

```python
# Predict single user-item score
score = model.predict(user_id=123, item_id=456)
print(f"Predicted score: {score:.2f}")

# Predict multiple pairs
scores = [
    model.predict(123, 456),
    model.predict(123, 789),
    model.predict(456, 123)
]
```

### recommend()

Generate top-K item recommendations for a user.

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
    Generate top-K item recommendations for a user.
    
    Args:
        user_id: User identifier
        top_k: Number of recommendations to generate
        exclude_items: Items to exclude from recommendations
        **kwargs: Additional recommendation parameters
        
    Returns:
        List of recommended item IDs (sorted by relevance)
        
    Example:
        recs = model.recommend(user_id=123, top_k=10)
    """
    pass
```

**Usage Example:**

```python
# Get top-10 recommendations
recs = model.recommend(user_id=123, top_k=10)
print(f"Recommendations: {recs}")

# Exclude already seen items
seen_items = [456, 789]
recs = model.recommend(
    user_id=123,
    top_k=10,
    exclude_items=seen_items
)

# Get recommendations with scores
recs = model.recommend(
    user_id=123,
    top_k=10,
    return_scores=True
)
# Returns: [(item_id, score), ...]
```

### save()

Save model to disk.

```python
@abstractmethod
def save(self, path: Union[str, Path], format: str = 'pickle') -> None:
    """
    Save model to disk.
    
    Args:
        path: File path to save model
        format: Save format ('pickle', 'json', 'torch')
        
    Example:
        model.save('models/ncf_model.pkl')
    """
    pass
```

**Usage Example:**

```python
# Save as pickle (default)
model.save('models/my_model.pkl')

# Save as PyTorch checkpoint
model.save('models/my_model.pth', format='torch')

# Save as JSON
model.save('models/my_model.json', format='json')
```

### load()

Load model from disk.

```python
@classmethod
@abstractmethod
def load(cls, path: Union[str, Path]) -> 'BaseRecommender':
    """
    Load model from disk.
    
    Args:
        path: File path to load model from
        
    Returns:
        Loaded model instance
        
    Example:
        model = NCF.load('models/ncf_model.pkl')
    """
    pass
```

**Usage Example:**

```python
from corerec.engines.dcn import DCN

# Load model
model = DCN.load('models/dcn_model.pkl')

# Use loaded model
recs = model.recommend(user_id=123, top_k=10)
```

## Non-Abstract Methods

These methods are implemented in `BaseRecommender` and available to all models:

### batch_predict()

Predict scores for multiple user-item pairs efficiently.

```python
def batch_predict(
    self,
    pairs: List[Tuple[Any, Any]],
    **kwargs
) -> List[float]:
    """
    Predict scores for multiple user-item pairs efficiently.
    
    Args:
        pairs: List of (user_id, item_id) tuples
        **kwargs: Additional parameters
        
    Returns:
        List of predicted scores
        
    Example:
        scores = model.batch_predict([(1,10), (1,11), (2,10)])
    """
    return [self.predict(user_id, item_id, **kwargs) 
            for user_id, item_id in pairs]
```

**Usage Example:**

```python
# Batch predictions
pairs = [
    (123, 456),
    (123, 789),
    (456, 123),
    (456, 789)
]

scores = model.batch_predict(pairs)
print(scores)  # [4.5, 3.2, 4.8, 2.1]
```

### batch_recommend()

Generate recommendations for multiple users efficiently.

```python
def batch_recommend(
    self,
    user_ids: List[Any],
    top_k: int = 10,
    **kwargs
) -> Dict[Any, List[Any]]:
    """
    Generate recommendations for multiple users efficiently.
    
    Args:
        user_ids: List of user identifiers
        top_k: Number of recommendations per user
        **kwargs: Additional parameters
        
    Returns:
        Dictionary mapping user_id to list of recommended items
        
    Example:
        recs = model.batch_recommend([1, 2, 3], top_k=5)
        # {1: [10,11,12,13,14], 2: [20,21,22,23,24], ...}
    """
    return {uid: self.recommend(uid, top_k, **kwargs) 
            for uid in user_ids}
```

**Usage Example:**

```python
# Batch recommendations
user_ids = [123, 456, 789]

recs = model.batch_recommend(user_ids, top_k=5)
# {
#     123: [1, 2, 3, 4, 5],
#     456: [6, 7, 8, 9, 10],
#     789: [11, 12, 13, 14, 15]
# }
```

### get_model_info()

Get model metadata and information.

```python
def get_model_info(self) -> Dict[str, Any]:
    """
    Get model metadata and information.
    
    Returns:
        Dictionary containing model info
    """
    return {
        'name': self.name,
        'version': self._version,
        'is_fitted': self.is_fitted,
        'model_type': self.__class__.__name__,
        'module': self.__class__.__module__
    }
```

**Usage Example:**

```python
info = model.get_model_info()
print(info)
# {
#     'name': 'DCN',
#     'version': '1.0.0',
#     'is_fitted': True,
#     'model_type': 'DCN',
#     'module': 'corerec.engines.dcn'
# }
```

## Properties

### is_fitted

Boolean flag indicating whether the model has been trained.

```python
if model.is_fitted:
    recs = model.recommend(user_id=123)
else:
    print("Model not trained yet!")
```

### name

Model name for identification.

```python
print(f"Model name: {model.name}")
```

### _version

Model version string.

```python
print(f"Model version: {model._version}")
```

## Complete Example

Here's a complete example showing all `BaseRecommender` methods:

```python
from corerec.engines.dcn import DCN
import pandas as pd

# Initialize model
model = DCN(
    embedding_dim=64,
    num_cross_layers=3,
    deep_layers=[128, 64, 32],
    epochs=20,
    verbose=True
)

# Check if fitted
print(f"Is fitted: {model.is_fitted}")  # False

# Train model
user_ids = [1, 1, 2, 2, 3, 3]
item_ids = [10, 11, 10, 12, 11, 13]
ratings = [5.0, 4.0, 4.0, 5.0, 3.0, 4.0]

model.fit(user_ids, item_ids, ratings)
print(f"Is fitted: {model.is_fitted}")  # True

# Single prediction
score = model.predict(user_id=1, item_id=10)
print(f"Predicted score: {score:.2f}")

# Batch predictions
pairs = [(1, 10), (2, 11), (3, 12)]
scores = model.batch_predict(pairs)
print(f"Batch scores: {scores}")

# Single recommendation
recs = model.recommend(user_id=1, top_k=5)
print(f"Recommendations: {recs}")

# Batch recommendations
users = [1, 2, 3]
batch_recs = model.batch_recommend(users, top_k=5)
print(f"Batch recommendations: {batch_recs}")

# Get model info
info = model.get_model_info()
print(f"Model info: {info}")

# Save model
model.save('models/dcn_model.pkl')

# Load model
loaded_model = DCN.load('models/dcn_model.pkl')

# Use loaded model
new_recs = loaded_model.recommend(user_id=1, top_k=5)
print(f"New recommendations: {new_recs}")
```

## Implementation Guidelines

When implementing a new model that inherits from `BaseRecommender`:

1. **Call super().__init__()** in your constructor
2. **Implement all abstract methods** (fit, predict, recommend, save, load)
3. **Set self.is_fitted = True** after successful training
4. **Use consistent parameter names** (user_id, item_id, top_k)
5. **Handle edge cases** (unknown users, invalid inputs)
6. **Add type hints** for better IDE support
7. **Write docstrings** following the base class format
8. **Add logging** for debugging and monitoring

Example implementation:

```python
from corerec.api.base_recommender import BaseRecommender
import pickle

class MyCustomModel(BaseRecommender):
    def __init__(self, embedding_dim=64, **kwargs):
        super().__init__(name="MyCustomModel")
        self.embedding_dim = embedding_dim
        # Initialize your model components
    
    def fit(self, user_ids, item_ids, ratings, **kwargs):
        # Training logic here
        self.is_fitted = True
        return self
    
    def predict(self, user_id, item_id, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        # Prediction logic here
        return score
    
    def recommend(self, user_id, top_k=10, exclude_items=None, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        # Recommendation logic here
        return recommended_items
    
    def save(self, path, format='pickle'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
```

## See Also

- [Model Interface](model-interface.md) - Interface for model components
- [Predictor Interface](predictor-interface.md) - Interface for predictors
- [Engines](../engines/index.md) - All available models
- [Examples](../examples/index.md) - Usage examples


