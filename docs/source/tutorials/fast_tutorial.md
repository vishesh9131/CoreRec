# FAST Tutorial: Fast Approximate Similarity Filtering

## Introduction

**FAST** is a production collaborative filtering model that learns user and item embeddings with fast approximate similarity-based training.

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.collaborative import FAST
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values.tolist()
train_items = train_df['movie_id'].values.tolist()
train_ratings = train_df['rating'].values.tolist()

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize and Train

```python
model = FAST(factors=50, iterations=10, seed=42, verbose=True)
model.fit(train_users, train_items, train_ratings)
print("Training complete!")
```

### Step 3: Predict and Recommend

```python
sample_user = train_users[0]
sample_item = train_items[0]
score = model.predict(sample_user, sample_item)
print(f"Score: {score:.3f}")

recs = model.recommend(sample_user, top_k=10)
print(f"Recommendations: {recs}")
```

### Step 4: Save & Load

```python
model.save('artifacts/fast_model')
loaded = FAST.load('artifacts/fast_model')
print(loaded.recommend(sample_user, top_k=5))
```

## Key Takeaways

✅ **Use for**: Fast prototyping, medium-scale CF, embedding-based similarity  
❌ **Not for**: Complex sequential or feature-rich ranking

## Further Reading

- [Collaborative Filtering](../api/engines.md)
