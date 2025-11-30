Reviewing the markdown rules and fixing the formatting issues:


[1 tool called]


Fixing markdown formatting. Creating 5 separate tutorial files with correct formatting:


[2 tools called]


Creating 5 separate tutorial files with correct markdown formatting. The issue was presenting them as nested blocks. Here are the corrected versions:

## Tutorial 1: `tutorial_01_basic_collaborative_filtering.md`


# Building Your First Recommendation System with CoreRec

## What You'll Learn

This tutorial shows you how to build a basic movie recommendation system using CoreRec's simple collaborative filtering. Perfect for beginners who want to get started quickly.

## Prerequisites

- Python 3.7+
- Basic understanding of pandas and numpy
- pip install corerec cr-learn

## Step 1: Load Your Data

We'll use the MovieLens 100K dataset from cr_learn. It's small enough to run fast but real enough to learn from.

```python
from cr_learn import ml
from sklearn.model_selection import train_test_split
import numpy as np

# load the dataset - returns a dictionary
data = ml.load()
ratings_df = data['ratings']  # has user_id, movie_id, rating, timestamp

print(f"Total ratings: {len(ratings_df)}")
print(f"Unique users: {ratings_df['user_id'].nunique()}")
print(f"Unique movies: {ratings_df['movie_id'].nunique()}")

# split into train/test - 80/20 split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# extract arrays for the model
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_ratings = train_df['rating'].values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = test_df['rating'].values
```

## Step 2: Choose a Simple Model

For beginners, we'll use the FAST recommender - it's fast (obviously) and easy to understand. It learns user and item embeddings.

```python
from corerec.engines.unionizedFilterEngine.fast import FAST

# initialize the model with small embedding size for speed
model = FAST(
    factors=32,        # embedding dimension
    iterations=10,     # number of training iterations
    batch_size=256,
    seed=42
)

print("Training model...")
model.fit(train_users, train_items, train_ratings)
print("Done!")
```

## Step 3: Get Recommendations

Now let's recommend movies for a user:

```python
# get top 10 recommendations for user 1
user_id = 1
recommendations = model.recommend(user_id, top_n=10)

print(f"\nTop 10 recommendations for user {user_id}:")
for i, movie_id in enumerate(recommendations, 1):
    print(f"{i}. Movie ID: {movie_id}")
```

## Step 4: Basic Evaluation

Let's see how well our model performs:

```python
from sklearn.metrics import mean_squared_error

# predict ratings for test set (just first 1000 for speed)
test_pred = []
for u, i in zip(test_users[:1000], test_items[:1000]):
    try:
        pred = model.predict(u, i)
        test_pred.append(pred)
    except:
        test_pred.append(3.0)  # default rating if prediction fails

# calculate RMSE
rmse = np.sqrt(mean_squared_error(test_ratings[:1000], test_pred))
print(f"Test RMSE: {rmse:.4f}")
```

## What's Next?

You've built your first recommender! Try:
- Experiment with different `factors` values (16, 64, 128)
- Try other models like `FASTRecommender` or `SAR`
- Add more epochs to see if accuracy improves


## Tutorial 2: `tutorial_02_deep_learning_recommendations.md`


# Deep Learning for Recommendations: DeepFM Tutorial

## Overview

This tutorial introduces deep learning models for recommendations. We'll use DeepFM which combines factorization machines with deep neural networks. It's more powerful than basic collaborative filtering.

## Why DeepFM?

DeepFM learns both low-order (linear) and high-order (non-linear) feature interactions automatically. It's used in production by many companies.

## Step 1: Prepare the Data

We'll use MovieLens 1M for more data to train a deeper model:

```python
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

# load larger dataset
data = ml_1m.load()
ratings_df = data['ratings']

print(f"Dataset size: {len(ratings_df)} ratings")

# convert ratings to binary (like/dislike) for implicit feedback
# ratings >= 4 become 1, others become 0
ratings_df['interaction'] = (ratings_df['rating'] >= 4).astype(int)

# split data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_interactions = train_df['interaction'].values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_interactions = test_df['interaction'].values
```

## Step 2: Initialize DeepFM Model

```python
from corerec import engines

# DeepFM combines FM and deep network
model = engines.DeepFM(
    embedding_dim=64,           # size of embedding vectors
    hidden_layers=[128, 64, 32], # deep network layers
    epochs=5,                   # training epochs
    batch_size=512,
    learning_rate=0.001,
    device="cpu"  # use "cuda" if you have GPU
)

print(f"Model initialized: {model}")
```

## Step 3: Train the Model

Training takes longer than simple models but gives better results:

```python
print("Training DeepFM...")
model.fit(train_users, train_items, train_interactions)
print("Training complete!")
```

## Step 4: Generate Recommendations

```python
# get recommendations for a user
user_id = train_users[0]
recommendations = model.recommend(user_id, top_n=20, exclude_seen=True)

print(f"\nRecommendations for user {user_id}:")
for rank, item_id in enumerate(recommendations, 1):
    score = model.predict(user_id, item_id)
    print(f"{rank}. Item {item_id} (score: {score:.4f})")
```

## Step 5: Evaluate Performance

For implicit feedback, we use ranking metrics:

```python
from sklearn.metrics import roc_auc_score

# predict on test set
test_scores = []
for u, i in zip(test_users[:5000], test_items[:5000]):
    try:
        score = model.predict(u, i)
        test_scores.append(score)
    except:
        test_scores.append(0.0)

# calculate AUC
auc = roc_auc_score(test_interactions[:5000], test_scores)
print(f"Test AUC: {auc:.4f}")
```

## Tips for Better Results

- Increase `embedding_dim` for more capacity (but slower training)
- Add more layers in `hidden_layers` for complex patterns
- Train for more epochs if validation metrics are improving
- Try different learning rates (0.0001 to 0.01)


## Tutorial 3: `tutorial_03_sequential_recommendations.md`


# Sequential Recommendations with SASRec

## Introduction

Real users have evolving preferences. Sequential models learn from the order of interactions. SASRec uses self-attention to capture long-term patterns in user behavior.

## When to Use Sequential Models

- E-commerce (browsing history matters)
- Music/video streaming (recent plays predict next)
- News recommendations (current events influence choices)

## Step 1: Load and Prepare Sequential Data

We need to preserve the temporal order of interactions:

```python
from cr_learn import ml_1m
import pandas as pd
from collections import defaultdict

# load data
data = ml_1m.load()
ratings_df = data['ratings']

# sort by timestamp to preserve order
ratings_df = ratings_df.sort_values('timestamp')

# build user sequences
user_sequences = defaultdict(list)
for _, row in ratings_df.iterrows():
    user_sequences[row['user_id']].append((row['movie_id'], row['timestamp']))

# filter users with at least 5 interactions
min_interactions = 5
filtered_users = {u: seq for u, seq in user_sequences.items() 
                  if len(seq) >= min_interactions}

print(f"Users with >= {min_interactions} interactions: {len(filtered_users)}")
```

## Step 2: Build Interaction Matrix

SASRec needs both sequences and a sparse matrix:

```python
from scipy.sparse import csr_matrix
import numpy as np

# create user and item mappings
all_users = sorted(filtered_users.keys())
all_items = sorted(set([item for seq in filtered_users.values() 
                       for item, _ in seq]))

user_map = {u: i for i, u in enumerate(all_users)}
item_map = {i: idx for idx, i in enumerate(all_items)}

# build sparse matrix
rows, cols, data = [], [], []
for u, seq in filtered_users.items():
    for item, _ in seq:
        rows.append(user_map[u])
        cols.append(item_map[item])
        data.append(1.0)

interaction_matrix = csr_matrix((data, (rows, cols)), 
                                shape=(len(all_users), len(all_items)))
```

## Step 3: Initialize SASRec

```python
from corerec import engines

model = engines.SASRec(
    hidden_units=64,        # hidden dimension
    num_blocks=2,           # transformer blocks
    num_heads=2,           # attention heads
    num_epochs=3,
    batch_size=256,
    max_seq_length=50,     # max sequence length
    learning_rate=0.001,
    device="cpu"
)

print("SASRec model initialized")
```

## Step 4: Train with Sequences

```python
# convert item IDs to mapped indices
mapped_user_ids = [user_map[u] for u in all_users]
mapped_item_ids = [item_map[i] for i in all_items]

print("Training SASRec...")
model.fit(
    user_ids=mapped_user_ids,
    item_ids=mapped_item_ids,
    interaction_matrix=interaction_matrix
)
print("Training complete!")
```

## Step 5: Get Sequential Recommendations

```python
# recommend for a user based on their sequence
test_user_idx = 0
test_user_id = all_users[test_user_idx]

recommendations = model.recommend(
    user_id=test_user_idx,
    top_n=10,
    exclude_seen=True
)

# map back to original item IDs
recommended_items = [all_items[idx] for idx in recommendations]

print(f"\nSequential recommendations for user {test_user_id}:")
for rank, item_id in enumerate(recommended_items, 1):
    print(f"{rank}. Item {item_id}")
```

## Understanding the Results

SASRec considers:
- Recent interactions (last few items)
- Long-term patterns (attention mechanism)
- Item co-occurrence in sequences

This makes recommendations more contextually relevant than non-sequential models.


## Tutorial 4: `tutorial_04_production_system.md`


# Building a Production Recommendation System

## Overview

This tutorial covers building a complete, production-ready recommendation system with evaluation, model persistence, and serving capabilities.

## Architecture

We'll build:
1. Data preprocessing pipeline
2. Model training with validation
3. Comprehensive evaluation
4. Model saving/loading
5. Batch inference

## Step 1: Setup Data Pipeline

```python
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from pathlib import Path

# load and prepare data
data = ml_1m.load()
ratings_df = data['ratings']

# create train/validation/test splits
train_df, temp_df = train_test_split(ratings_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# extract features
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_ratings = (train_df['rating'] >= 4).astype(int).values

val_users = val_df['user_id'].values
val_items = val_df['movie_id'].values
val_ratings = (val_df['rating'] >= 4).astype(int).values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = (test_df['rating'] >= 4).astype(int).values

print(f"Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")
```

## Step 2: Train with Validation

```python
from corerec import engines
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# initialize model
model = engines.DeepFM(
    embedding_dim=128,
    hidden_layers=[256, 128, 64],
    epochs=10,
    batch_size=1024,
    learning_rate=0.001,
    device="cpu"
)

# train on training set
print("Training model...")
model.fit(train_users, train_items, train_ratings)

# validate on validation set
print("Evaluating on validation set...")
val_scores = []
for u, i in zip(val_users[:10000], val_items[:10000]):
    try:
        score = model.predict(u, i)
        val_scores.append(score)
    except:
        val_scores.append(0.0)

val_auc = roc_auc_score(val_ratings[:10000], val_scores)
print(f"Validation AUC: {val_auc:.4f}")
```

## Step 3: Comprehensive Evaluation

```python
def evaluate_model(model, users, items, ratings, name="Test"):
    """Evaluate model with multiple metrics"""
    predictions = []
    actuals = []
    
    # sample for faster evaluation
    sample_size = min(10000, len(users))
    indices = np.random.choice(len(users), sample_size, replace=False)
    
    for idx in indices:
        u, i, r = users[idx], items[idx], ratings[idx]
        try:
            pred = model.predict(u, i)
            predictions.append(pred)
            actuals.append(r)
        except:
            continue
    
    if len(predictions) == 0:
        return {}
    
    auc = roc_auc_score(actuals, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actuals, 
        [1 if p > 0.5 else 0 for p in predictions],
        average='binary',
        zero_division=0
    )
    
    print(f"\n{name} Metrics:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    return {'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}

# evaluate on test set
test_metrics = evaluate_model(model, test_users, test_items, test_ratings, "Test")
```

## Step 4: Save and Load Model

```python
# save model
model_dir = Path("saved_models")
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "production_deepfm.pkl"
model.save(str(model_path))
print(f"Model saved to {model_path}")

# also save metadata
metadata = {
    'train_size': len(train_users),
    'val_auc': val_auc,
    'test_metrics': test_metrics,
    'model_type': 'DeepFM'
}

with open(model_dir / "metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)

# load model later
loaded_model = engines.DeepFM.load(str(model_path))
print("Model loaded successfully")
```

## Step 5: Batch Inference for Production

```python
def batch_recommend(model, user_ids, top_k=10):
    """Generate recommendations for multiple users efficiently"""
    all_recommendations = {}
    
    for user_id in user_ids:
        try:
            recs = model.recommend(user_id, top_n=top_k, exclude_seen=True)
            all_recommendations[user_id] = recs
        except:
            all_recommendations[user_id] = []
    
    return all_recommendations

# example: recommend for 100 users
sample_users = np.unique(test_users)[:100]
batch_recs = batch_recommend(model, sample_users, top_k=20)

print(f"\nGenerated recommendations for {len(batch_recs)} users")
print(f"Average recommendations per user: {np.mean([len(r) for r in batch_recs.values()]):.1f}")
```

## Production Checklist

- [x] Train/validation/test splits
- [x] Model evaluation with multiple metrics
- [x] Model persistence
- [x] Batch inference capability
- [ ] Logging and monitoring (add in production)
- [ ] A/B testing framework (add in production)
- [ ] Model versioning (add in production)


## Tutorial 5: `tutorial_05_industry_standard.md`


# Industry-Standard Recommendation System: Complete Pipeline

## Overview

This tutorial builds a complete, industry-standard recommendation system with multiple models, hybrid approaches, graph-based recommendations, and production features.

## System Architecture

```
Data → Preprocessing → Multiple Models → Ensemble → Graph Enhancement → Serving
```

## Step 1: Multi-Model Training

Train multiple models and ensemble them:

```python
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np
from corerec import engines

# load data
data = ml_1m.load()
ratings_df = data['ratings']
ratings_df['interaction'] = (ratings_df['rating'] >= 4).astype(int)

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_interactions = train_df['interaction'].values

# train multiple models
models = {}

# Model 1: DeepFM
print("Training DeepFM...")
models['deepfm'] = engines.DeepFM(
    embedding_dim=128, hidden_layers=[256, 128], 
    epochs=5, batch_size=512, device="cpu"
)
models['deepfm'].fit(train_users, train_items, train_interactions)

# Model 2: DCN (Deep & Cross Network)
print("Training DCN...")
models['dcn'] = engines.DCN(
    embedding_dim=128, num_cross_layers=3, deep_layers=[256, 128],
    epochs=5, batch_size=512, device="cpu"
)
models['dcn'].fit(train_users, train_items, train_interactions)

# Model 3: GNN-based (if available)
try:
    print("Training GNNRec...")
    models['gnn'] = engines.GNNRec(
        embedding_dim=128, num_gnn_layers=2,
        epochs=5, batch_size=512, device="cpu"
    )
    models['gnn'].fit(train_users, train_items, train_interactions)
except:
    print("GNNRec not available, skipping")

print(f"Trained {len(models)} models")
```

## Step 2: Ensemble Predictions

Combine predictions from multiple models:

```python
def ensemble_predict(models, user_id, item_id, weights=None):
    """Weighted ensemble of model predictions"""
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models.keys()}
    
    predictions = []
    for name, model in models.items():
        try:
            pred = model.predict(user_id, item_id)
            predictions.append(pred * weights.get(name, 0.0))
        except:
            continue
    
    return sum(predictions) if predictions else 0.0

def ensemble_recommend(models, user_id, top_k=10, weights=None):
    """Get ensemble recommendations"""
    # get all candidate items
    all_items = np.unique(train_items)
    
    # score all items
    item_scores = []
    for item_id in all_items:
        score = ensemble_predict(models, user_id, item_id, weights)
        item_scores.append((item_id, score))
    
    # sort and return top-k
    item_scores.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in item_scores[:top_k]]

# test ensemble
test_user = train_users[0]
ensemble_recs = ensemble_recommend(models, test_user, top_k=20)
print(f"Ensemble recommendations for user {test_user}: {ensemble_recs[:10]}")
```

## Step 3: Graph-Based Enhancement

Add graph structure for better recommendations:

```python
from corerec.core_rec import GraphTransformerV2, train_model, predict
from corerec.cr_utility.dataset import GraphDataset
from torch.utils.data import DataLoader
import torch

# build user-item graph adjacency matrix
unique_users = sorted(set(train_users))
unique_items = sorted(set(train_items))
user_map = {u: i for i, u in enumerate(unique_users)}
item_map = {i: idx for idx, i in enumerate(unique_items)}

# create bipartite graph: users + items
n_nodes = len(unique_users) + len(unique_items)
adj_matrix = np.zeros((n_nodes, n_nodes))

# connect users to items they interacted with
for u, i in zip(train_users[:10000], train_items[:10000]):  # sample for speed
    u_idx = user_map[u]
    i_idx = item_map[i] + len(unique_users)
    adj_matrix[u_idx, i_idx] = 1.0
    adj_matrix[i_idx, u_idx] = 1.0  # undirected

print(f"Graph created: {n_nodes} nodes, {np.sum(adj_matrix > 0) / 2} edges")
```

## Step 4: Hybrid Content + Collaborative

Combine content-based and collaborative filtering:

```python
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender

# content-based model using movie titles/descriptions
if 'movies' in data:
    movies_df = data['movies']
    # create item content dictionary
    item_content = {}
    for _, row in movies_df.iterrows():
        # combine title and genres as content
        content = f"{row.get('title', '')} {row.get('genres', '')}"
        item_content[row['movie_id']] = content
    
    # train TF-IDF model
    content_model = TFIDFRecommender()
    item_ids = list(item_content.keys())
    contents = [item_content[iid] for iid in item_ids]
    content_model.fit(item_ids, item_content)
    
    print("Content-based model trained")
    
    # hybrid recommendation: combine collaborative + content
    def hybrid_recommend(collab_models, content_model, user_id, item_content, top_k=10):
        # get collaborative recommendations
        collab_recs = ensemble_recommend(collab_models, user_id, top_k=top_k*2)
        
        # get content similarity for user's liked items
        user_items = train_items[train_users == user_id][:5]  # recent items
        if len(user_items) > 0:
            # find items similar to user's preferences
            user_pref_text = " ".join([item_content.get(i, "") for i in user_items])
            content_recs = content_model.recommend_by_text(user_pref_text, top_n=top_k)
            
            # combine and deduplicate
            all_recs = list(set(collab_recs + content_recs))
            return all_recs[:top_k]
        else:
            return collab_recs[:top_k]
    
    hybrid_recs = hybrid_recommend(models, content_model, test_user, item_content)
    print(f"Hybrid recommendations: {hybrid_recs[:10]}")
```

## Step 5: Production Serving Setup

Create a simple serving interface:

```python
import json
from pathlib import Path

class RecommendationService:
    """Production recommendation service"""
    
    def __init__(self, models, content_model=None):
        self.models = models
        self.content_model = content_model
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {i: idx for idx, i in enumerate(unique_items)}
    
    def recommend(self, user_id, top_k=10, strategy='ensemble'):
        """Get recommendations for a user"""
        if strategy == 'ensemble':
            return ensemble_recommend(self.models, user_id, top_k)
        elif strategy == 'deepfm':
            return self.models['deepfm'].recommend(user_id, top_n=top_k)
        elif strategy == 'dcn':
            return self.models['dcn'].recommend(user_id, top_n=top_k)
        else:
            return []
    
    def predict_score(self, user_id, item_id):
        """Predict interaction score"""
        return ensemble_predict(self.models, user_id, item_id)
    
    def batch_recommend(self, user_ids, top_k=10):
        """Batch recommendations"""
        results = {}
        for uid in user_ids:
            results[uid] = self.recommend(uid, top_k)
        return results

# create service
service = RecommendationService(models)

# example API-like usage
user_recs = service.recommend(test_user, top_k=20, strategy='ensemble')
print(f"Service recommendations: {user_recs[:10]}")

# batch processing
batch_users = np.unique(train_users)[:50]
batch_results = service.batch_recommend(batch_users, top_k=15)
print(f"Processed {len(batch_results)} users")
```

## Step 6: Evaluation and Monitoring

```python
from sklearn.metrics import roc_auc_score, ndcg_score
import time

def evaluate_service(service, test_users, test_items, test_ratings, sample_size=5000):
    """Comprehensive evaluation"""
    indices = np.random.choice(len(test_users), min(sample_size, len(test_users)), replace=False)
    
    predictions = []
    actuals = []
    latencies = []
    
    for idx in indices:
        u, i, r = test_users[idx], test_items[idx], test_ratings[idx]
        
        start = time.time()
        pred = service.predict_score(u, i)
        latencies.append(time.time() - start)
        
        predictions.append(pred)
        actuals.append(r)
    
    auc = roc_auc_score(actuals, predictions)
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"\nService Performance:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Avg Latency: {avg_latency*1000:.2f}ms")
    print(f"  P95 Latency: {p95_latency*1000:.2f}ms")
    
    return {'auc': auc, 'avg_latency': avg_latency, 'p95_latency': p95_latency}

# evaluate
metrics = evaluate_service(service, test_users, test_items, test_ratings)
```

## Industry Best Practices

1. **Model Diversity**: Use multiple model types (deep, graph, content)
2. **Ensemble**: Combine predictions for robustness
3. **Evaluation**: Monitor multiple metrics (AUC, latency, diversity)
4. **Scalability**: Use batch processing for production
5. **Monitoring**: Track latency and accuracy in real-time
6. **A/B Testing**: Compare different strategies
7. **Cold Start**: Handle new users/items (content-based helps)
8. **Explainability**: Provide reasons for recommendations

## Next Steps

- Add real-time updates (online learning)
- Implement caching for frequent requests
- Add user feedback loop
- Deploy as REST API
- Set up monitoring dashboard
- Implement A/B testing framework
```

---

These 5 tutorials are formatted as separate markdown files. Each:
- Uses proper markdown headers
- Has code blocks starting at column 0 (no indentation)
- Uses ````python` for new code examples (not code references)
- Follows a consistent structure
- Can be saved as separate `.md` files

Save them in your `docs/source/tutorials/` directory or wherever you keep tutorials.