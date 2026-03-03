# QuickStart Guide

Get started with CoreRec in 5 minutes!

## Basic Example

```python
from corerec.engines.dcn import DCN
import cr_learn

# Load data
data = cr_learn.load_dataset('movielens-100k')
train, test = data.train_test_split(test_size=0.2)

# Create model
model = DCN(
    embedding_dim=64,
    epochs=20,
    verbose=True
)

# Train
model.fit(
    user_ids=train.user_ids,
    item_ids=train.item_ids,
    ratings=train.ratings
)

# Predict
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Recommend
recs = model.recommend(user_id=1, top_k=10)
print(f"Top-10 recommendations: {recs}")

# Save
model.save('my_model.pkl')

# Load
loaded_model = DCN.load('my_model.pkl')
```

## Available Models

### Production Models (Tested & Stable)

These 13 models are fully tested, CI-enforced, and recommended for production use:

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
model.fit(users, items, ratings)
score = model.predict(user_id, item_id)
```

### Top-N Recommendation
```python
from corerec.engines.sasrec import SASRec

model = SASRec()
model.fit(users, items, ratings)
recs = model.recommend(user_id, top_k=10)
```

### Graph-Based
```python
from corerec.engines.gnnrec import GNNRec

model = GNNRec()
model.fit(users, items, ratings)
recs = model.recommend(user_id, top_k=10)
```
