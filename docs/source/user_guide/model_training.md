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

## Required ratings

SAR, DCN, and DeepFM require explicit ratings in every input form: lists,
DataFrames, and `RecommenderDataset`. Missing ratings raise `InvalidDataError`
before changing the fitted model. DataFrames must contain the rating column
configured for that model (normally `rating`).

For positive implicit feedback, supply the positive labels explicitly:

```python
from corerec.engines import DCN

users = [1, 1, 2, 2]
items = [10, 11, 11, 12]
ratings = [1.0] * len(users)
model = DCN(epochs=1, batch_size=4, device="cpu")
model.fit(users, items, ratings)
print(model.recommend(1, top_k=1))
```

Other interaction models that support omitted ratings continue to interpret
those interactions as positive feedback.

## Choosing the task: implicit ranking vs. rating prediction

The deep CTR/graph models (`DCN`, `DeepFM`, `GNNRec`) accept a `task` argument that
controls how labels are interpreted and how the model is trained:

```python
from corerec.engines import DCN

# top-K recommendation (default): observed interactions are positives and the
# model is trained with negative sampling -> use this for ranking.
model = DCN(task="auto", num_negatives=4)        # 'auto' == 'implicit'

# rating prediction: regress the supplied rating with a linear head + MSE.
model = DCN(task="rating")
```

```{important}
Train these models with `task="implicit"` (the default) for recommendation. Passing
raw 1–5 ratings to an implicit model without negatives is the classic mistake that
makes the output collapse to a constant; CoreRec now handles this for you via
negative sampling and warns if a trained model's scores have (near-)zero variance.
Use `task="rating"` only when you genuinely want to predict the rating value.
```

## Training strong models for large-scale ranking

For million-scale implicit-feedback ranking, CoreRec ships native, GPU, sparse
trainers that produce strong embeddings and feed straight into the online serving
engine:

```python
from corerec.serving import OnlineRecommender

# trains a sparse LightGCN (SOTA-class on graph benchmarks) and builds the index
rec = OnlineRecommender.from_interactions(df, model="lightgcn", device="cuda")
```

See the [Online Serving guide](online_serving.md) for the full train-and-serve path.

## Callbacks

CoreRec provides training callbacks for monitoring and control:

```python
from corerec.utils.training_utils import EarlyStopping, ModelCheckpoint

# Stop training when validation loss plateaus
early_stop = EarlyStopping(patience=5, min_delta=0.001)

# Save the best model during training
checkpoint = ModelCheckpoint(
    filepath='artifacts/best_model',
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
model = DeepFM(device='auto')  # default: CUDA, then Apple MPS, then CPU
model = DeepFM(device='cuda')  # NVIDIA GPU
model = DeepFM(device='mps')   # Apple Silicon GPU
model = DeepFM(device='cpu')   # Force CPU
```


## Classic collaborative filtering inputs

ItemKNN, UserKNN, EASE and SLIM accept event lists, keyword arguments,
DataFrames and RecommenderDataset objects through the shared interactions
adapter. Missing ratings mean one interaction.

```python
import pandas as pd
from corerec.engines import ItemKNN

events = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3],
    "item_id": [10, 20, 20, 30, 10, 30],
    "rating": [1.0] * 6,
})
model = ItemKNN().fit(events)
print(model.recommend(1, top_k=2, exclude_seen=True))

same_model = ItemKNN().fit(
    user_ids=events["user_id"].tolist(),
    item_ids=events["item_id"].tolist(),
    ratings=events["rating"].tolist(),
)
assert same_model.user_map == model.user_map
assert same_model.item_map == model.item_map
```

Malformed lengths, missing IDs and non-finite ratings raise InvalidDataError
before fitted state is replaced. Integer and string IDs remain distinct.

DCN recommendations exclude the user's observed training items by default.
Use `model.recommend(user_id, top_k=10, exclude_seen=False)` to include them.
New DCN artifacts preserve this history in both safe and legacy formats. Older
artifacts did not store history; refit those models to restore seen-item filtering.
