# Advanced Usage Examples

This guide covers advanced usage patterns and techniques in CoreRec.

## Model Ensembles

### Combining Multiple Models

```python
from corerec.engines.dcn import DCN
from corerec.engines.deepfm import DeepFM
from corerec.engines.contentFilterEngine.hybrid_ensemble_methods import EnsembleRecommender

# Train individual models
dcn_model = DCN(embedding_dim=64)
dcn_model.fit(user_ids, item_ids, ratings)

deepfm_model = DeepFM(embedding_dim=64)
deepfm_model.fit(user_ids, item_ids, ratings)

# Create ensemble
ensemble = EnsembleRecommender(ensemble_strategy='weighted_average')
ensemble.add_model(dcn_model, name="DCN", weight=1.5)
ensemble.add_model(deepfm_model, name="DeepFM", weight=1.0)
ensemble.train()

# Get ensemble recommendations
recommendations = ensemble.recommend(user_id=1, top_n=10)
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'embedding_dim': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [10, 20, 30]
}

best_score = -np.inf
best_params = None

for params in ParameterGrid(param_grid):
    model = DCN(**params)
    model.fit(train_user_ids, train_item_ids, train_ratings)
    
    # Evaluate on validation set
    score = evaluate(model, val_user_ids, val_item_ids, val_ratings)
    
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best params: {best_params}, Score: {best_score}")
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in kf.split(user_ids):
    # Split data
    train_users = user_ids[train_idx]
    train_items = item_ids[train_idx]
    train_ratings = ratings[train_idx]
    
    val_users = user_ids[val_idx]
    val_items = item_ids[val_idx]
    val_ratings = ratings[val_idx]
    
    # Train model
    model = DCN(embedding_dim=64)
    model.fit(train_users, train_items, train_ratings)
    
    # Evaluate
    score = evaluate(model, val_users, val_items, val_ratings)
    scores.append(score)

print(f"Mean CV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## Custom Evaluation Metrics

### Implementing Custom Metrics

```python
def precision_at_k(model, user_ids, item_ids, ratings, k=10):
    """Calculate Precision@K"""
    precisions = []
    
    for user_id in user_ids:
        # Get recommendations
        recommendations = model.recommend(user_id, top_k=k)
        
        # Get actual relevant items
        user_items = item_ids[user_ids == user_id]
        user_ratings = ratings[user_ids == user_id]
        relevant = user_items[user_ratings >= 4]  # Threshold for relevant
        
        # Calculate precision
        if len(recommendations) > 0:
            precision = len(set(recommendations) & set(relevant)) / len(recommendations)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0

# Use custom metric
model = DCN(embedding_dim=64)
model.fit(user_ids, item_ids, ratings)
precision = precision_at_k(model, test_user_ids, test_item_ids, test_ratings, k=10)
```

## Batch Processing

### Efficient Batch Predictions

```python
from corerec.api.mixins import BatchProcessingMixin

class MyModel(BaseRecommender, BatchProcessingMixin):
    pass

model = MyModel()
model.fit(user_ids, item_ids, ratings)

# Batch predictions
batch_user_ids = [1, 2, 3, 4, 5]
batch_item_ids = [10, 20, 30, 40, 50]
predictions = model.batch_predict(batch_user_ids, batch_item_ids)

# Batch recommendations
recommendations = model.batch_recommend(batch_user_ids, top_k=10)
```

## Model Persistence

### Saving with Metadata

```python
from corerec.api.mixins import ModelPersistenceMixin

class MyModel(BaseRecommender, ModelPersistenceMixin):
    pass

model = MyModel()
model.fit(user_ids, item_ids, ratings)

# Save with metadata
metadata = {
    'training_date': '2024-01-01',
    'dataset': 'movielens-100k',
    'version': '1.0.0'
}
model.save('model.pkl', metadata=metadata)
```

## Handling Cold Start

### New User Recommendations

```python
# For new users, use content-based or hybrid approaches
from corerec.engines.contentFilterEngine.hybrid_ensemble_methods import HybridCollaborative

hybrid_model = HybridCollaborative(
    hybrid_strategy='switching',  # Switch to content for cold users
    cf_weight=0.6,
    content_weight=0.4
)

hybrid_model.train(user_item_matrix, item_features)

# Works for both warm and cold users
recommendations = hybrid_model.recommend(user_id=new_user_id, top_n=10)
```

## Performance Optimization

### Using GPU

```python
import torch

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DCN(embedding_dim=64, device=device)
model.fit(user_ids, item_ids, ratings)
```

### Parallel Processing

```python
from multiprocessing import Pool

def train_model(params):
    model = DCN(**params)
    model.fit(user_ids, item_ids, ratings)
    return evaluate(model, val_user_ids, val_item_ids, val_ratings)

# Train multiple models in parallel
with Pool(processes=4) as pool:
    scores = pool.map(train_model, param_configs)
```

## See Also

- [Basic Usage](basic_usage.md) - Basic examples
- [Production Deployment](production_deployment.md) - Deployment guides
- [Tutorials](../tutorials/index.md) - Model-specific tutorials

