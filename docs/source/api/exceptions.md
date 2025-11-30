# Exceptions API

CoreRec provides a comprehensive exception hierarchy for better error handling and debugging.

## Overview

```python
from corerec.api.exceptions import (
    CoreRecException,
    ModelNotFittedError,
    InvalidDataError,
    InvalidParameterError,
    SaveLoadError,
    RecommendationError,
    ConfigurationError
)
```

## Exception Hierarchy

```
CoreRecException (base)
├── ModelNotFittedError
├── InvalidDataError
├── InvalidParameterError
├── SaveLoadError
├── RecommendationError
└── ConfigurationError
```

## Base Exception

### CoreRecException

Base exception for all CoreRec errors.

```python
class CoreRecException(Exception):
    """Base exception for all CoreRec errors."""
    pass
```

## Specific Exceptions

### ModelNotFittedError

Raised when attempting to use a model that hasn't been fitted.

```python
class ModelNotFittedError(CoreRecException):
    """
    Exception raised when attempting to use a model that hasn't been fitted.
    
    Example:
        >>> model = SomeRecommender()
        >>> model.predict(user_id=1, item_id=10)
        ModelNotFittedError: Model must be fitted before making predictions.
    """
```

**Common Causes:**
- Calling `predict()` before `fit()`
- Calling `recommend()` before `fit()`
- Loading a model incorrectly

**Solution:**
```python
model.fit(data)  # Fit first
model.predict(user_id=1, item_id=10)  # Then predict
```

### InvalidDataError

Raised when input data is invalid or malformed.

```python
class InvalidDataError(CoreRecException):
    """
    Exception raised when input data is invalid or malformed.
    
    Example:
        >>> model.fit(data="invalid")
        InvalidDataError: Expected DataFrame or dict, got str
    """
```

**Common Causes:**
- Wrong data type
- Missing required columns
- Malformed data structure
- Empty dataset

**Solution:**
```python
# Ensure correct data format
data = pd.DataFrame({
    'user_id': user_ids,
    'item_id': item_ids,
    'rating': ratings
})
model.fit(data)
```

### InvalidParameterError

Raised when model parameters are invalid.

```python
class InvalidParameterError(CoreRecException):
    """
    Exception raised when model parameters are invalid.
    
    Example:
        >>> model = SomeRecommender(embedding_dim=-5)
        InvalidParameterError: embedding_dim must be positive, got -5
    """
```

**Common Causes:**
- Negative values where positive required
- Out of range values
- Incompatible parameter combinations
- Missing required parameters

**Solution:**
```python
# Use valid parameters
model = SomeRecommender(embedding_dim=64)  # Positive value
```

### SaveLoadError

Raised when model save/load operations fail.

```python
class SaveLoadError(CoreRecException):
    """
    Exception raised when model save/load operations fail.
    
    Example:
        >>> model.save('/invalid/path/model.pkl')
        SaveLoadError: Cannot save model to /invalid/path/model.pkl
    """
```

**Common Causes:**
- Invalid file path
- Permission issues
- Disk full
- Corrupted file during load

**Solution:**
```python
# Use valid paths with proper permissions
model.save('/valid/path/model.pkl')
model = ModelClass.load('/valid/path/model.pkl')
```

### RecommendationError

Raised when recommendation generation fails.

```python
class RecommendationError(CoreRecException):
    """
    Exception raised when recommendation generation fails.
    
    Example:
        >>> model.recommend(user_id=999999)
        RecommendationError: Unknown user_id: 999999
    """
```

**Common Causes:**
- Unknown user ID
- No items available for recommendation
- Insufficient user history
- Model not fitted

**Solution:**
```python
# Use known user IDs
known_users = model.user_ids
if user_id in known_users:
    recommendations = model.recommend(user_id=user_id)
```

### ConfigurationError

Raised when model configuration is invalid.

```python
class ConfigurationError(CoreRecException):
    """
    Exception raised when model configuration is invalid.
    
    Example:
        >>> model = SomeRecommender(config={'invalid': 'config'})
        ConfigurationError: Missing required config key: 'embedding_dim'
    """
```

**Common Causes:**
- Missing required config keys
- Invalid config values
- Incompatible configuration
- Deprecated config options

**Solution:**
```python
# Use valid configuration
config = {
    'embedding_dim': 64,
    'learning_rate': 0.001
}
model = SomeRecommender(config=config)
```

## Error Handling Best Practices

### 1. Catch Specific Exceptions

```python
try:
    model.predict(user_id=1, item_id=10)
except ModelNotFittedError:
    print("Model must be fitted first")
    model.fit(data)
except InvalidDataError as e:
    print(f"Invalid data: {e}")
```

### 2. Validate Inputs Early

```python
if not model.is_fitted:
    raise ModelNotFittedError("Fit the model first")

if user_id not in model.user_ids:
    raise RecommendationError(f"Unknown user_id: {user_id}")
```

### 3. Provide Helpful Error Messages

```python
if embedding_dim <= 0:
    raise InvalidParameterError(
        f"embedding_dim must be positive, got {embedding_dim}"
    )
```

## See Also

- [BaseRecommender](base_recommender.md) - Base class API
- [Mixins](mixins.md) - Reusable functionality

