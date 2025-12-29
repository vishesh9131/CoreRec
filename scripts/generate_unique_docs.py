#!/usr/bin/env python3
"""
Enhanced Documentation Generator with Model-Specific Content

Creates UNIQUE, detailed documentation for each model with:
- Specific architecture details
- Actual mathematical formulations  
- Real use cases and comparisons
- Model-specific best practices
"""

from pathlib import Path
from typing import Dict

# Model-specific detailed content
MODEL_DETAILS = {
    "MIND": {
        "architecture": """MIND (Multi-Interest Network with Dynamic Routing) uses a multi-interest extraction layer with dynamic routing mechanism to capture diverse user interests.

**Key Components:**
1. **Multi-Interest Extractor Layer**: Uses capsule network with dynamic routing
2. **Label-Aware Attention**: Attends to relevant interests for target item
3. **Interest Aggregation**: Combines multiple interest representations

**Architecture Flow:**
```
User Behavior → Embedding → Multi-Interest Capsules → Label-Aware Attention → Prediction
```""",
        "math": """**Multi-Interest Extraction:**
```
e_i = Embed(item_i)
interests = Capsule([e_1, e_2, ..., e_n])  # B × K × d
where K = number of interests
```

**Dynamic Routing:**
```
c_ij = exp(b_ij) / Σ_k exp(b_ik)  # routing coefficients
s_j = Σ_i c_ij * u_i             # interest capsule
v_j = squash(s_j)                # activation
```

**Label-Aware Attention:**
```
a_i = softmax(e_target^T · interest_i)
user_repr = Σ a_i · interest_i
score = σ(user_repr^T · e_target)
```""",
        "use_cases": """✅ **Ideal For:**
- E-commerce with diverse user interests (fashion, electronics, books)
- Multi-category recommendations
- Users with varied browsing patterns
- Capturing interest evolution over time

❌ **Not Ideal For:**
- Single-domain recommendations
- Very sparse data (<100 items per user)
- Real-time systems (slower than simpler models)""",
        "best_practices": """1. **Number of Interests (K)**: Start with K=4, increase for diverse catalogs
2. **Routing Iterations**: 3 iterations sufficient for most cases
3. **Sequence Length**: Use 20-50 recent items
4. **Interest Regularization**: Add diversity loss to prevent collapsed interests
5. **Training**: Use auxiliary losses for each interest capsule""",
    },
    "NASRec": {
        "architecture": """NASRec (Neural Architecture Search for Recommendations) automatically discovers optimal neural architectures using reinforcement learning-based search.

**Search Space:**
1. **Operation Types**: Convolution, LSTM, Attention, MLP, Skip connections
2. **Layer Configurations**: Hidden sizes, activation functions
3. **Connection Patterns**: Sequential, residual, dense

**Search Process:**
1. Controller RNN proposes architectures
2. Train candidate on validation set
3. Use validation performance as reward
4. Update controller with policy gradient""",
        "math": """**Architecture Encoding:**
```
arch = Controller_RNN(random_state)
arch = [layer_1_type, layer_1_config, ..., layer_n_type, layer_n_config]
```

**Reward Function:**
```
R = α · NDCG@10 - β · latency - γ · params
where α, β, γ are balancing coefficients
```

**Controller Update:**
```
∇L = E[R(arch) · ∇log P(arch|θ)]
θ ← θ + η · ∇L
```""",
        "use_cases": """✅ **Ideal For:**
- Novel recommendation domains without established architectures
- Performance-critical applications
- Research and experimentation
- When you have significant compute budget

❌ **Not Ideal For:**
- Quick prototyping (search is slow)
- Limited compute resources
- Well-understood domains (use proven architectures)
- Production systems without retraining""",
        "best_practices": """1. **Search Budget**: Minimum 50-100 architecture evaluations
2. **Early Stopping**: Stop unpromising architectures at 5 epochs
3. **Warm Start**: Initialize with known good architectures  
4. **Constrained Search**: Limit search space to reduce time
5. **Multi-Objective**: Balance performance,latency, model size""",
    },
    "SASRec": {
        "architecture": """SASRec (Self-Attentive Sequential Recommendation) uses self-attention mechanism to model item-item transitions in user sequences.

**Core Innovation:** Replaces RNN/CNN with self-attention blocks for better long-range dependencies.

**Architecture:**
```
Item Sequence → Embedding → Positional Encoding → 
Multi-Head Self-Attention Blocks (×L) → Prediction Layer
```

**Multi-Head Attention Block:**
1. Self-attention with causal masking
2. Point-wise feed-forward network
3. Layer normalization
4. Residual connections""",
        "math": """**Self-Attention:**
```
Q = E · W^Q, K = E · W^K, V = E · W^V
Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

**Causal Masking** (prevents future leakage):
```
M_ij = {0 if i ≥ j, -∞ if i < j}
Attention = softmax((QK^T + M) / √d_k) · V  
```

**Position Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Prediction:**
```
r_i = [r_i^1; r_i^2; ...; r_i^h]  # concat heads
y_i,t = E_t^T · FFN(LN(r_i + E_i))
```""",
        "use_cases": """✅ **Ideal For:**
- Sequential user behavior (browsing, listening, watching)
- Session-based recommendations  
- Long sequences (50-200 items)
- Capturing long-range dependencies
- E-commerce, streaming platforms

❌ **Not Ideal For:**
- Very short sequences (<5 items)
- Static user-item ratings
- Graph-structured data
- When interpretability is critical""",
        "best_practices": """1. **Sequence Length**: 50-200 items optimal
2. **Attention Blocks**: 2-4 blocks sufficient
3. **Attention Heads**: 2-4 heads
4. **Hidden Size**: 50-100 dimensions
5. **Dropout**: 0.2-0.5 for regularization
6. **Positional Encoding**: Essential for sequence order
7. **Learning Rate**: 0.001 with warmup (1000 steps)""",
    },
    # Add more models...
}


def create_model_specific_tutorial(
    model_name: str, model_class: str, details: Dict, output_dir: Path
):
    """Create tutorial with actual model-specific content."""

    model_lower = model_name.lower().replace("-", "_")
    default_advanced = '### Feature Engineering\n\nAdd model-specific advanced usage here.'
    default_best_practices = '1. Start with default parameters\n2. Tune based on validation set\n3. Use early stopping'

    content = f"""# {model_name} Tutorial: {details.get('title', model_name)}

## Introduction

**{model_name}** is {details.get('intro', 'a recommendation model')}

## How {model_name} Works

### Architecture

{details.get('architecture', 'TODO: Add architecture')}

### Mathematical Foundation

{details.get('math', 'TODO: Add math')}

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.{details.get('module', model_lower)} import {model_class}
import cr_learn
import numpy as np

# Load dataset
data = cr_learn.load_dataset('{details.get('dataset', 'movielens-100k')}')
print(f"Loaded {{len(data.ratings)}} ratings")

# Split data
train_data, test_data = data.train_test_split(test_size=0.2)
```

### Step 2: Initialize Model

```python
model = {model_class}(
    name="{model_name}_Model",
{details.get('init_params', '    embedding_dim=64,')}
    epochs=20,
    batch_size=256,
    learning_rate=0.001,
    verbose=True
)

print(f"Initialized {{model.name}}")
```

### Step 3: Train

```python
model.fit(
    user_ids=train_data.user_ids,
    item_ids=train_data.item_ids,
    ratings=train_data.ratings
)

print("Training complete!")
```

### Step 4: Predict

```python
# Single prediction
score = model.predict(user_id=1, item_id=100)
print(f"Predicted score: {{score:.3f}}")

# Batch predictions
pairs = [(1, 100), (2, 200), (3, 300)]
scores = model.batch_predict(pairs)
for (uid, iid), s in zip(pairs, scores):
    print(f"User {{uid}}, Item {{iid}}: {{s:.3f}}")
```

### Step 5: Recommend

```python
# Get top-10 recommendations
recommendations = model.recommend(
    user_id=1,
    top_k=10,
    exclude_items=train_data.get_user_items(1)
)

print(f"Top-10 recommendations for User 1:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {{rank}}. Item {{item_id}}")
```

### Step 6: Evaluate

```python
from corerec.metrics import rmse, ndcg_at_k

# Rating prediction
predictions = [model.predict(u, i) for u, i, r in test_data]
test_rmse = rmse(test_data.ratings, predictions)
print(f"Test RMSE: {{test_rmse:.4f}}")

# Ranking quality
ndcg = ndcg_at_k(model, test_data, k=10)
print(f"NDCG@10: {{ndcg:.4f}}")
```

### Step 7: Save & Load

```python
# Save model
model.save('{model_lower}_model.pkl')

# Load model
loaded = {model_class}.load('{model_lower}_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {{test_score:.3f}}")
```

## Advanced Usage

{details.get('advanced', default_advanced)}

## Key Takeaways

### When to Use {model_name}

{details.get('use_cases', 'Use for general recommendation tasks.')}

### Best Practices

{details.get('best_practices', default_best_practices)}

### Performance Comparison

{details.get('comparison', f'Compare {model_name} with similar models on your dataset.')}

## Further Reading

- [{model_name} API Reference](../api/engines.rst#{model_lower})
- Paper: {details.get('paper', 'See original paper for details')}
- [Code Examples](../examples/{model_lower}_advanced.md)
"""

    output_path = output_dir / f"{model_lower}_tutorial.md"
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


if __name__ == "__main__":
    base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")
    tutorials_dir = base_dir / "docs" / "source" / "tutorials"

    print("Creating model-specific documentation...")

    for model_name, details in MODEL_DETAILS.items():
        try:
            output = create_model_specific_tutorial(
                model_name=model_name,
                model_class=details.get("class", model_name),
                details=details,
                output_dir=tutorials_dir,
            )
            print(f"✓ Created detailed {model_name} tutorial")
        except Exception as e:
            print(f"✗ Failed {model_name}: {e}")
            
