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
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create model
model = DCN(
    embedding_dim=64,
    epochs=20,
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

### Production Models (Tested & Stable)

These 14 models are fully tested, CI-enforced, and recommended for production use:

- **Deep Learning**: DCN, DeepFM, GNNRec, MIND, NASRec, SASRec, TwoTower, BERT4Rec
- **Collaborative**: SAR, NCF, FAST, FASTRecommender, LightGCN
- **Content-Based**: TFIDFRecommender

### Sandbox Models (Experimental)

~50 additional models for research and exploration. These are **not production-tested** — see [Model Tiers](models/index.md#model-tiers) for details.

- **Neural Networks**: AFM, AutoInt, DIEN, DIN, DLRM, Wide&Deep, and more
- **Matrix Factorization**: SVD, ALS, A2SVD, and more
- **Graph-Based**: GeoIMC, LightGCN-Base, GNN-Base
- **Sequential**: RBM, SLiRec, SUM
- **Bayesian**: BPR, BPRMF, VMF

## Next Steps

1. Read [Concepts](concepts.md) to understand recommendation systems
2. Follow [Tutorials](tutorials/index.md) for detailed walkthroughs  
3. Browse [Examples](examples/basic_usage.md) for common patterns
4. Check [API Reference](api/base_recommender.md) for all methods

## Common Workflows

### Rating Prediction
```python
from corerec.engines.deepfm import DeepFM

model = DeepFM()
model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
score = model.predict(user_id=user_id, item_id=item_id)
```

### Top-K Recommendation (SASRec — interaction matrix)

SASRec needs a user×item **interaction matrix**, not raw triplets alone:

```python
from corerec.engines.sasrec import SASRec
import numpy as np

user_list = sorted(train_df['user_id'].unique())
item_list = sorted(train_df['movie_id'].unique())
user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {it: j for j, it in enumerate(item_list)}

train_mat = np.zeros((len(user_list), len(item_list)), dtype=np.float32)
for _, row in train_df.iterrows():
    train_mat[user_idx[row['user_id']], item_idx[row['movie_id']]] = 1.0

model = SASRec(num_epochs=5, hidden_units=64, max_seq_length=50, verbose=True)
model.fit(user_list, item_list, train_mat)
recs = model.recommend(user_id=1, top_k=10)
```

### Graph-Based (GNNRec — binarize ratings)

GNNRec uses **BCE loss**; ratings must be in **[0, 1]** (implicit feedback or normalized explicit ratings):

```python
from corerec.engines.gnnrec import GNNRec
import numpy as np

# Implicit feedback: rating >= 1 → 1.0, else 0.0
binary_ratings = (train_df['rating'].values >= 1.0).astype(np.float32)

model = GNNRec(embedding_dim=64, num_gnn_layers=3, epochs=20, verbose=True)
model.fit(
    user_ids=train_df['user_id'].values,
    item_ids=train_df['movie_id'].values,
    ratings=binary_ratings,
)
recs = model.recommend(user_id=1, top_k=10)
```
