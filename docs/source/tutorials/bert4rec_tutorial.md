# Bert4Rec Tutorial: BERT for Sequential Recommendation

## Introduction

**Bert4Rec** applies bidirectional transformer (BERT) to user sequences using cloze task to learn from both past and future context.

**Paper**: Sun et al. 2019 - BERT4Rec: Sequential Recommendation with Bidirectional Encoder

## How Bert4Rec Works

### Architecture

**Bidirectional Transformer:**

1. **Masked Sequence Modeling**: Randomly mask 15-20% items
2. **Bidirectional Attention**: Full attention (no causal mask)
3. **Transformer Encoder**: Stacked self-attention + FFN
4. **Prediction Head**: Predict masked items

**Key vs SASRec**: Can look both directions (not just left-to-right)

**Architecture:**
```
[i1, [MASK], i3, i4, [MASK]] → Transformer → Predict [i2, i5]
```

### Mathematical Foundation

**Full Bidirectional Attention:**
```
Attention(Q,K,V) = softmax(QK^T/√d) · V
```
No causal masking!

**Cloze Training Objective:**
```
L = -Σ_{m∈masked} log P(v_m | S_{\m})
```
where S_{\m} is sequence excluding position m

**Positional Encoding:**
```
PE(pos, 2i) = sin(pos/10000^(2i/d))
PE(pos, 2i+1) = cos(pos/10000^(2i/d))
```

## Tutorial with cr_learn

### Step 1: Import and Load Data

BERT4Rec needs a user–item **interaction matrix** (binary or implicit feedback).

```python
from corerec.engines.bert4rec import BERT4Rec
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
    train_mat[user_idx[row['user_id']], item_idx[row['movie_id']]] = 1.0

print(f"Loaded {len(ratings_df)} ratings → {len(user_list)} users, {len(item_list)} items")
```

### Step 2: Initialize Model

```python
model = BERT4Rec(
    hidden_dim=64,
    num_layers=2,
    num_heads=2,
    max_len=50,
    mask_prob=0.15,
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

### Step 4: Predict

```python
sample_user = next(iter(model.user_seqs)) if model.user_seqs else user_list[0]
sample_item = item_list[0]
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

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error

test_users = test_df['user_id'].values[:100]
test_items = test_df['movie_id'].values[:100]
test_ratings = test_df['rating'].values[:100]
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Sample Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
model.save('bert4rec.pt')

loaded = BERT4Rec.load('bert4rec.pt')
recs = loaded.recommend(sample_user, top_k=5)
print(f"Loaded model recommendations: {recs}")
```

## Key Takeaways

### When to Use Bert4Rec

✅ **Perfect For:**
- Offline training with full sequences
- Long interaction histories (50-200 items)
- Rich bidirectional patterns
- Cold start with partial sequences
- Research and benchmarking

❌ **Not For:**
- Real-time next-item prediction (needs future context)
- Online/streaming scenarios
- Very short sequences (<10 items)
- Production serving (slow inference)

### Best Practices

1. **Mask Probability**: 15-20% of items
2. **Max Sequence Length**: 50-200 items
3. **Transformer Layers**: 2-4 layers
4. **Attention Heads**: 2-4 heads
5. **Hidden Dimension**: 64-128
6. **Warmup Steps**: 1000-10000 for learning rate
7. **Batch Size**: 128-512
8. **Pre-training**: Can pre-train on large corpus

## Further Reading

- Paper: Sun et al. 2019 - BERT4Rec: Sequential Recommendation with Bidirectional Encoder
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
