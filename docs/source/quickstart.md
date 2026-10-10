# QuickStart Guide

Get started with CoreRec in 5 minutes!

## Install tutorial data

The examples below use MovieLens 1M via `cr_learn`:

```bash
pip install "corerec[datasets]"
```

The first `ml_1m.load()` downloads ~25 MB to your local cache.

## Basic Example

```python
from corerec.engines.dcn import DCN
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

# Load data (cr_learn returns dict with 'ratings' DataFrame)
data = ml_1m.load()
ratings_df = data['ratings'].head(10_000)
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create model
model = DCN(
    embedding_dim=8,
    deep_layers=[8],
    epochs=1,
    device="cpu",
    verbose=True
)

# Train
model.fit(
    user_ids=train_df['user_id'].values,
    item_ids=train_df['movie_id'].values,
    ratings=train_df['rating'].values
)

# Predict
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Recommend
recs = model.recommend(user_id=1, top_k=10)
print(f"Top-10 recommendations: {recs}")

# Save (safe bundle default — base path, not a .pkl file)
model.save('artifacts/my_dcn')

# Load
loaded_model = DCN.load('artifacts/my_dcn')
```

## Available Models

The installed registry is the authoritative list of available models:

```python
from corerec.engines import MODELS
print(sorted(MODELS))
```

Classic models include ALS, SAR, ItemKNN, UserKNN, EASE, SLIM, and Item2Vec.
Neural models include DCN, DeepFM, TwoTower, LightGCN, SASRec, HSTU, MultVAE,
and MultiDAE. TFIDFRecommender supports item text similarity.

Historical sandbox engines and GNNRec are unavailable in this release.
Use [removed models](tutorials/removed_models.md) for migration information.
Experimental towers and tracking integrations live under `corerec.experimental`.

## Next Steps

1. Read [Concepts](concepts.md) to understand recommendation systems
2. Follow [Tutorials](tutorials/index.md) for detailed walkthroughs  
3. Browse [Examples](examples/basic_usage.md) for common patterns
4. Check [API Reference](api/base_recommender.md) for all methods

## Common Workflows

### Rating Prediction

```python
from corerec.engines import DeepFM

model = DeepFM(epochs=1, embedding_dim=8, hidden_layers=[8], device="cpu")
model.fit([1, 1, 2, 2], [10, 20, 20, 30], [5., 4., 4., 3.])
score = model.predict(user_id=1, item_id=30)
```

### Sequential Recommendation

SASRec accepts interaction triplets. Event order determines each user's history;
pass `timestamps=` when the rows are not already chronological.

```python
from corerec.engines import SASRec

model = SASRec(epochs=1, hidden_units=8, num_blocks=1,
               batch_size=4, device="cpu", verbose=False)
model.fit([1, 1, 2, 2, 3, 3], [10, 20, 20, 30, 10, 30])
assert model.recommend(1, top_k=1) == [30]
```

### Graph-Based Recommendation

```python
from corerec.engines import LightGCN

model = LightGCN(n_factors=8, n_layers=1, epochs=1,
                 batch_size=4, device="cpu", verbose=False)
model.fit([1, 1, 2, 2, 3, 3], [10, 20, 20, 30, 10, 30])
assert model.recommend(1, top_k=1) == [30]
```
