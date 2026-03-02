# Model Persistence

## Saving Models

All CoreRec models support saving to disk:

```python
# Save model
model.save('my_model.pkl')
```

## Loading Models

Load a previously saved model:

```python
from corerec.engines import DeepFM

model = DeepFM.load('my_model.pkl')
recommendations = model.recommend(user_id=1, top_k=10)
```

## Model Information

Inspect a model's configuration and state:

```python
info = model.get_model_info()
print(f"Model: {info['name']}")
print(f"Users: {info['num_users']}")
print(f"Items: {info['num_items']}")
print(f"Fitted: {info['is_fitted']}")
```

## Checking Model State

```python
# Check if model has been trained
if model.is_fitted:
    recs = model.recommend(user_id=1, top_k=10)

# Check if model knows a specific user/item
if model.knows_user(42):
    score = model.predict(user_id=42, item_id=100)
```
