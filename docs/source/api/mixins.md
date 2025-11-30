# Mixins API

Mixins provide reusable functionality that can be added to recommendation models through multiple inheritance.

## Overview

```python
from corerec.api.mixins import (
    ModelPersistenceMixin,
    BatchProcessingMixin,
    ValidationMixin,
    EarlyStoppingMixin
)
```

## Available Mixins

### ModelPersistenceMixin

Provides standardized save/load functionality.

```python
class ModelPersistenceMixin:
    """
    Mixin providing standardized save/load functionality.
    
    Usage:
        class MyModel(BaseRecommender, ModelPersistenceMixin):
            pass
        
        model = MyModel()
        model.fit(data)
        model.save('model.pkl')
        loaded = MyModel.load('model.pkl')
    """
```

**Methods:**

#### save()

Save model to disk.

```python
def save(self, path: str, **kwargs) -> None:
    """
    Save model to disk.
    
    Args:
        path: File path to save model
        **kwargs: Additional save parameters
    """
```

#### load()

Load model from disk.

```python
@classmethod
def load(cls, path: str) -> 'ModelPersistenceMixin':
    """
    Load model from disk.
    
    Args:
        path: File path to load model from
        
    Returns:
        Loaded model instance
    """
```

**Usage:**
```python
class MyModel(BaseRecommender, ModelPersistenceMixin):
    pass

model = MyModel()
model.fit(data)
model.save('model.pkl')
loaded = MyModel.load('model.pkl')
```

### BatchProcessingMixin

Provides efficient batch prediction and recommendation.

```python
class BatchProcessingMixin:
    """
    Mixin providing efficient batch prediction/recommendation.
    
    Usage:
        class MyModel(BaseRecommender, BatchProcessingMixin):
            pass
        
        model = MyModel()
        model.fit(data)
        
        # Batch predictions
        predictions = model.batch_predict(user_ids, item_ids)
        
        # Batch recommendations
        recommendations = model.batch_recommend(user_ids, top_k=10)
    """
```

**Methods:**

#### batch_predict()

Predict scores for multiple user-item pairs.

```python
def batch_predict(
    self,
    user_ids: List[Any],
    item_ids: List[Any],
    **kwargs
) -> np.ndarray:
    """
    Predict scores for multiple user-item pairs.
    
    Args:
        user_ids: List of user IDs
        item_ids: List of item IDs
        **kwargs: Additional prediction parameters
        
    Returns:
        Array of predicted scores
    """
```

#### batch_recommend()

Generate recommendations for multiple users.

```python
def batch_recommend(
    self,
    user_ids: List[Any],
    top_k: int = 10,
    **kwargs
) -> Dict[Any, List[Any]]:
    """
    Generate recommendations for multiple users.
    
    Args:
        user_ids: List of user IDs
        top_k: Number of recommendations per user
        **kwargs: Additional recommendation parameters
        
    Returns:
        Dictionary mapping user IDs to recommendation lists
    """
```

**Usage:**
```python
class MyModel(BaseRecommender, BatchProcessingMixin):
    pass

model = MyModel()
model.fit(data)

# Batch operations
user_ids = [1, 2, 3]
item_ids = [10, 20, 30]
predictions = model.batch_predict(user_ids, item_ids)

recommendations = model.batch_recommend(user_ids, top_k=10)
```

### ValidationMixin

Provides common data validation helpers.

```python
class ValidationMixin:
    """
    Mixin providing common data validation helpers.
    
    Usage:
        class MyModel(BaseRecommender, ValidationMixin):
            pass
        
        model = MyModel()
        model.validate_data(data)  # Validates input data
    """
```

**Methods:**

#### validate_data()

Validate input data format and structure.

```python
def validate_data(self, data: Any, **kwargs) -> bool:
    """
    Validate input data format and structure.
    
    Args:
        data: Data to validate
        **kwargs: Additional validation parameters
        
    Returns:
        True if data is valid, raises exception otherwise
    """
```

#### validate_user_id()

Validate user ID exists in model.

```python
def validate_user_id(self, user_id: Any) -> bool:
    """
    Validate user ID exists in model.
    
    Args:
        user_id: User ID to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
```

#### validate_item_id()

Validate item ID exists in model.

```python
def validate_item_id(self, item_id: Any) -> bool:
    """
    Validate item ID exists in model.
    
    Args:
        item_id: Item ID to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
```

### EarlyStoppingMixin

Provides early stopping functionality during training.

```python
class EarlyStoppingMixin:
    """
    Mixin providing early stopping functionality during training.
    
    Usage:
        class MyModel(BaseRecommender, EarlyStoppingMixin):
            pass
        
        model = MyModel()
        model.fit(
            data,
            validation_data=val_data,
            early_stopping=True,
            patience=5
        )
    """
```

**Methods:**

#### should_stop()

Check if training should stop early.

```python
def should_stop(self, metric_value: float, **kwargs) -> bool:
    """
    Check if training should stop early.
    
    Args:
        metric_value: Current metric value
        **kwargs: Additional parameters
        
    Returns:
        True if training should stop
    """
```

## Combining Mixins

You can combine multiple mixins:

```python
class MyModel(
    BaseRecommender,
    ModelPersistenceMixin,
    BatchProcessingMixin,
    ValidationMixin,
    EarlyStoppingMixin
):
    pass

model = MyModel()
model.fit(data)
model.save('model.pkl')
predictions = model.batch_predict(user_ids, item_ids)
```

## Best Practices

1. **Inherit from BaseRecommender first**
2. **Add mixins after BaseRecommender**
3. **Use mixins for reusable functionality**
4. **Don't override mixin methods unless necessary**

## See Also

- [BaseRecommender](base_recommender.md) - Base class API
- [Exceptions](exceptions.md) - Error handling

