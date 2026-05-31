# SAR Tutorial: Smart Adaptive Recommendations

## Introduction

**SAR** is a Sequential Models model for recommendation systems. This model implements Smart Adaptive Recommendations.

## How SAR Works

### Architecture

SAR uses a sophisticated architecture for recommendation tasks.

### Mathematical Foundation

The model learns user and item representations for prediction.

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.collaborative import SAR
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

# Load dataset (returns dict with 'ratings' DataFrame)
data = ml_1m.load()
ratings_df = data['ratings']
# SAR expects userID, itemID, rating columns (rename if needed)
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
# Single prediction (use userID and itemID from your data)
sample_user = train_df['userID'].iloc[0]
sample_item = train_df['itemID'].iloc[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")
```

### Step 5: Recommend

```python
# Get top-10 recommendations for a user
recommendations = model.recommend(user_id=sample_user, top_k=10)

print(f"Top-10 recommendations for User {sample_user}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Save & Load

```python
# Save model
model.save('sar_model.pkl')

# Load model
loaded = SAR.load('sar_model.pkl')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
```

## Key Takeaways

### When to Use SAR

✅ Best for datasets with sequential models characteristics

### Best Practices

1. Start with default parameters\n2. Tune embedding_dim based on data\n3. Use early stopping\n4. Monitor validation metrics

## Further Reading

