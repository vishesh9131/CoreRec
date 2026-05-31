# SAR Tutorial: Smart Adaptive Recommendations

## Introduction

**SAR** (Smart Adaptive Recommendations) is an **item-based collaborative filtering** model. It uses item co-occurrence and similarity (Jaccard, lift, etc.) to produce fast, interpretable recommendations — similar in spirit to classic “users who liked this also liked…” systems.

## How SAR Works

### Architecture

SAR builds item–item similarity from user interaction history, then scores candidate items for each user from items they have already interacted with.

### Mathematical Foundation

Recommendations combine item affinity and similarity signals over the user’s history (see Microsoft Recommenders SAR for the original formulation).

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.collaborative import SAR
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
ratings_df = ratings_df.rename(columns={"user_id": "userID", "movie_id": "itemID"})
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize Model

```python
model = SAR(similarity_type='jaccard')

print(f"Initialized SAR with {model.similarity_type} similarity")
```

### Step 3: Train

```python
model.fit(train_df)

print("Training complete!")
```

### Step 4: Predict

```python
sample_user = train_df['userID'].iloc[0]
sample_item = train_df['itemID'].iloc[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")
```

### Step 5: Recommend

```python
recommendations = model.recommend(user_id=sample_user, top_k=10)

print(f"Top-10 recommendations for User {sample_user}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Save & Load

```python
model.save('artifacts/sar')
loaded = SAR.load('artifacts/sar')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
```

## Key Takeaways

### When to Use SAR

✅ **Best for:**
- Implicit or explicit feedback with item co-occurrence signal
- Fast baseline CF without neural training
- Interpretable item-similarity recommendations

❌ **Not ideal for:**
- Strict sequential next-item prediction (use SASRec / BERT4Rec)
- Rich side features only (pair with content models)

### Best Practices

1. Use `userID`, `itemID`, `rating` column names (or pass custom columns to `fit`)
2. Try `similarity_type='jaccard'` or `'lift'` for implicit data
3. Save with a base path (`artifacts/sar`) for the safe bundle default

## Further Reading

- [SAR on Microsoft Recommenders](https://github.com/recommenders-team/recommenders)
- [Production deployment](../examples/production_deployment.md)
