# ALS Tutorial: Alternating Least Squares

```{admonition} Sandbox Model — Experimental
:class: warning
This model is part of the CoreRec **Sandbox** — an experimental collection of algorithms for research and exploration. Sandbox models are **not production-tested** and may have incomplete implementations. For production-ready models, see the [Production Models](../models/index.md#production-models-tested--stable).
```

## Introduction

**ALS** alternates between fixing user factors and solving for item factors (and vice versa) to efficiently factorize large sparse matrices.

**Paper**: Hu et al. 2008 - Collaborative Filtering for Implicit Feedback

## How ALS Works

### Architecture

**Alternating Optimization for Scalability:**

1. **Initialization**: Random user and item factors
2. **Fix Items, Solve Users**: Solve for all user factors in parallel
3. **Fix Users, Solve Items**: Solve for all item factors in parallel
4. **Repeat**: Alternate until convergence

**Key Innovation**: Closed-form solution per iteration, highly parallelizable

**Objective:**
```
min Σ_(u,i) c_ui(p_ui - x_u^T y_i)² + λ(Σ||x_u||² + Σ||y_i||²)
```

### Mathematical Foundation

**User Update (parallel over all users):**
```
x_u = (Y^T C^u Y + λI)^(-1) Y^T C^u p^u
```
where:
- Y: item factor matrix
- C^u: confidence diagonal matrix for user u
- p^u: preference vector for user u

**Item Update (parallel over all items):**
```
y_i = (X^T C^i X + λI)^(-1) X^T C^i p^i
```

**Confidence:**
```
c_ui = 1 + α · r_ui
```
for implicit feedback r_ui

## Reference Implementation

```{admonition} Implementation Note
:class: info
The code examples below are **reference implementations** for learning purposes. This is a sandbox model and has not been production-tested. Verify behavior thoroughly before using in production.
```


### Step 1: Import and Load Data

```python
from corerec.engines.als import ALS
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset with CORRECT API
data = ml_1m.load()  # Returns dict with 'ratings', 'users', 'movies'
ratings_df = data['ratings']  # DataFrame with user_id, movie_id, rating, timestamp

print(f"Loaded {len(ratings_df)} ratings")

# Split data
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Extract arrays for model
train_users = train_df['user_id'].values
train_items = train_df['movie_id'].values
train_ratings = train_df['rating'].values

test_users = test_df['user_id'].values
test_items = test_df['movie_id'].values
test_ratings = test_df['rating'].values
```

### Step 2: Initialize Model

```python
model = ALS(
    name="ALS_Model",
    n_factors=100,
    regularization=0.01,
    alpha=40,
    iterations=15,
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)

print(f"Initialized {model.name}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_users,
    item_ids=train_items,
    ratings=train_ratings
)

print("Training complete!")
```

### Step 4: Predict

```python
# Single prediction
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {score:.3f}")

# Batch predictions
test_predictions = model.batch_predict(list(zip(test_users[:100], test_items[:100])))
```

### Step 5: Recommend

```python
# Get top-10 recommendations for user
user_id = 1
recommendations = model.recommend(
    user_id=user_id,
    top_k=10
)

print(f"Top-10 recommendations for User {user_id}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {rank}. Item {item_id}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Predict all test ratings
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Test RMSE: {rmse:.4f}")
```

### Step 7: Save & Load

```python
# Save model
model.save('als_model.pkl')

# Load model
loaded = ALS.load('als_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {test_score:.3f}")
```

## Key Takeaways

### When to Use ALS

✅ **Perfect For:**
- Very large scale (billions of interactions)
- Implicit feedback (views, clicks)
- Distributed/parallel computation
- Production systems at scale
- Spark, Hadoop environments

❌ **Not For:**
- Small datasets (<10K) - SGD simpler
- Explicit ratings - SGD more flexible
- Need online updates - ALS is batch
- Dense features - use neural models

### Best Practices

1. **Factors**: 50-200 typically
2. **Confidence α**: 40 is common (tune 10-100)
3. **Regularization λ**: 0.01-0.1
4. **Iterations**: 10-20 sufficient
5. **Parallelization**: Partition by user/item blocks
6. **Implicit**: Always use confidence weighting
7. **Caching**: Cache Y^T Y and X^T X
8. **Convergence**: Monitor RMSE on validation

## Scaling & Production Considerations

If you plan to move this sandbox model toward production use, consider:

- **Data Volume**: Test with realistic dataset sizes; sandbox implementations may not be optimized for large-scale data.
- **Distributed Training**: For datasets exceeding single-machine memory, consider wrapping the model in distributed training frameworks (e.g., PyTorch DDP, Horovod).
- **Hyperparameter Tuning**: Use systematic search (Optuna, Ray Tune) rather than manual tuning. The defaults in sandbox models are not tuned for any specific domain.
- **Serving Latency**: Profile inference time. Some architectures (e.g., attention-based) may need quantization or ONNX export for low-latency serving.
- **Monitoring**: Implement A/B testing and online metrics (CTR, engagement, diversity) before replacing an existing system.
- **Validation**: Write comprehensive unit tests and integration tests before deploying. Sandbox models do not have CI coverage.

## Further Reading

- Paper: Hu et al. 2008 - Collaborative Filtering for Implicit Feedback
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
