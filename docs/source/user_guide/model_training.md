# Model Training

## Basic Training

All CoreRec models follow the same `fit()` API:

```python
from corerec.engines.collaborative import SAR

model = SAR(similarity_type='jaccard')
model.fit(train_df)
```

For deep learning models:

```python
from corerec.engines import DeepFM

model = DeepFM(
    embedding_dim=64,
    hidden_layers=[256, 128],
    epochs=20,
    learning_rate=0.001,
    batch_size=256,
    device='cuda',
)
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
```

## Callbacks

CoreRec provides training callbacks for monitoring and control:

```python
from corerec.utils.training_utils import EarlyStopping, ModelCheckpoint

# Stop training when validation loss plateaus
early_stop = EarlyStopping(patience=5, min_delta=0.001)

# Save the best model during training
checkpoint = ModelCheckpoint(
    filepath='best_model.pkl',
    monitor='val_loss',
    save_best_only=True,
)
```

## Validation

Use the built-in validation helpers before training:

```python
from corerec.utils.validation import validate_fit_inputs

# Validates data format, column names, and types
validate_fit_inputs(train_df, col_user='userID', col_item='itemID')
```

## GPU Training

Deep learning models support GPU acceleration:

```python
model = DeepFM(device='cuda')  # Use GPU
model = DeepFM(device='cpu')   # Force CPU
model = DeepFM(device='auto')  # Auto-detect
```
