# TFIDF Tutorial: TF-IDF Based Recommender

## Introduction

**TFIDF** is a Content-Based Models model for recommendation systems. This model implements TF-IDF Based Recommender.

## How TFIDF Works

### Architecture

TFIDF uses a sophisticated architecture for recommendation tasks.

### Mathematical Foundation

The model learns user and item representations for prediction.

## Tutorial with cr_learn

TFIDFRecommender is **content-based** — it needs item IDs and their text descriptions (e.g. titles, descriptions), not user-item ratings.

### Step 1: Import and Prepare Data

```python
from corerec.engines.content_based import TFIDFRecommender

# TFIDF needs: list of item IDs and dict mapping item_id -> text description
items = list(range(20))
docs = {i: f"item {i} description with keywords topic{i % 5} category{i % 3}" for i in items}

# For real data, use item metadata from your dataset (e.g. product titles, article text)
# data = ml_1m.load()
# movies = data['movies']  # DataFrame with movie_id, title, genres
# items = movies['movie_id'].tolist()
# docs = {row['movie_id']: f"{row['title']} {row['genres']}" for _, row in movies.iterrows()}

print(f"Prepared {len(items)} items with text")
```

### Step 2: Initialize and Fit

```python
model = TFIDFRecommender()
model.fit(items, docs)

print("Training complete!")
```

### Step 3: Predict and Recommend

```python
# Similarity between two items
score = model.predict(items[0], items[1])
print(f"Similarity score: {score:.3f}")

# Top-k similar items for an item (content-based recommendations)
recommendations = model.recommend(items[0], top_k=10)

print(f"Top-10 similar items for item {items[0]}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from corerec.metrics import rmse, ndcg_at_k

# Rating prediction
predictions = [model.predict(u, i) for u, i, r in test_data]
test_rmse = rmse(test_data.ratings, predictions)
print(f"Test RMSE: {test_rmse:.4f}")

# Ranking quality
ndcg = ndcg_at_k(model, test_data, k=10)
print(f"NDCG@10: {ndcg:.4f}")
```

### Step 7: Save & Load

```python
# Save model
model.save('tfidf_model.pkl')

# Load model
loaded = TFIDFRecommender.load('tfidf_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use TFIDF

✅ Best for datasets with content-based models characteristics

### Best Practices

1. Start with default parameters\n2. Tune embedding_dim based on data\n3. Use early stopping\n4. Monitor validation metrics

## Further Reading

