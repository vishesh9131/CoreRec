# TwoTower Tutorial: Dual-Encoder Retrieval

## Introduction

**TwoTower** is a production dual-encoder model for large-scale candidate retrieval. It learns separate user and item embeddings and scores pairs via dot product — ideal for the first stage of a recommendation pipeline.

## How TwoTower Works

### Architecture

1. **User Tower**: Encodes user features / interaction history into an embedding
2. **Item Tower**: Encodes item features into an embedding
3. **Scoring**: Dot product (or cosine similarity) between user and item vectors

```
User features → User Tower → u_emb ─┐
                                    ├─ dot(u_emb, i_emb) → score
Item features → Item Tower → i_emb ─┘
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

TwoTower expects a user–item interaction matrix.

```python
from corerec.engines.two_tower import TwoTower
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

user_list = sorted(train_df['user_id'].unique())
item_list = sorted(train_df['movie_id'].unique())
user_idx = {u: i for i, u in enumerate(user_list)}
item_idx = {it: j for j, it in enumerate(item_list)}

train_mat = np.zeros((len(user_list), len(item_list)), dtype=np.float32)
for _, row in train_df.iterrows():
    train_mat[user_idx[row['user_id']], item_idx[row['movie_id']]] = float(row['rating'])

print(f"Loaded {len(ratings_df)} ratings")
```

### Step 2: Initialize Model

```python
model = TwoTower(
    user_input_dim=64,
    item_input_dim=64,
    embedding_dim=128,
    num_epochs=5,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)
```

### Step 3: Train

```python
model.fit(user_list, item_list, train_mat)

print("Training complete!")
```

### Step 4: Predict and Recommend

```python
sample_user = user_list[0]
sample_item = item_list[0]
score = model.predict(sample_user, sample_item)
print(f"Predicted score: {score:.3f}")

recs = model.recommend(user_id=sample_user, top_k=10)
print(f"Top-10 recommendations: {recs}")
```

### Step 5: Save & Load

```python
model.save('artifacts/two_tower')
loaded = TwoTower.load('artifacts/two_tower')
print(loaded.recommend(sample_user, top_k=5))
```

## Key Takeaways

✅ **Use for**: Large catalogs, retrieval stage, real-time serving  
❌ **Not for**: Fine-grained ranking without a second-stage ranker

## Further Reading

- [Engines API](../api/engines.md)
