#!/usr/bin/env python3
"""
Regenerate First 10 Model Tutorials with REAL Unique Content

Models: DCN, DeepFM, GNNRec, MIND, NASRec, SASRec, NCF, LightGCN, DIEN, DIN
"""

import sys

sys.path.insert(0, "/Users/visheshyadav/Documents/GitHub/CoreRec/scripts")
from pathlib import Path
from model_database import MODEL_DATABASE

# Template with all model-specific placeholders
TUTORIAL_TEMPLATE = """# {model_name} Tutorial: {full_name}

## Introduction

**{model_name}** {intro}.

**Paper**: {paper}

## How {model_name} Works

### Architecture

{architecture}

### Mathematical Foundation

{math}

## Tutorial with cr_learn

### Step 1: Import and Load Data

```python
from corerec.engines.{module_path} import {class_name}
import cr_learn
import numpy as np

# Load dataset
data = cr_learn.load_dataset('{dataset}')
print(f"Loaded {{len(data.ratings)}} ratings")

# Split data
train_data, test_data = data.train_test_split(test_size=0.2, random_state=42)
```

### Step 2: Initialize Model

```python
model = {class_name}(
    name="{model_name}_Model",
{init_params}
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
loaded = {class_name}.load('{model_lower}_model.pkl')
test_score = loaded.predict(1, 100)
print(f"Loaded model prediction: {{test_score:.3f}}")
```

## Key Takeaways

### When to Use {model_name}

{use_cases}

### Best Practices

{best_practices}

## Further Reading

- Paper: {paper}
- [API Reference](../api/engines.rst#{model_lower})
- [GitHub Issues](https://github.com/vishesh9131/CoreRec/issues)
"""


def generate_tutorial(model_name: str, output_dir: Path):
    """Generate tutorial with unique model-specific content."""

    if model_name not in MODEL_DATABASE:
        print(f"  ✗ {model_name}: No detailed content available")
        return None

    details = MODEL_DATABASE[model_name]
    model_lower = model_name.lower().replace("-", "_")

    # Determine module path
    module_map = {
        "DCN": "dcn",
        "DeepFM": "deepfm",
        "GNNRec": "gnnrec",
        "MIND": "mind",
        "NASRec": "nasrec",
        "SASRec": "sasrec",
        "NCF": "ncf",
        "LightGCN": "lightgcn",
        "DIEN": "dien",
        "DIN": "din",
    }

    content = TUTORIAL_TEMPLATE.format(
        model_name=model_name,
        model_lower=model_lower,
        full_name=details["full_name"],
        paper=details["paper"],
        intro=details["intro"],
        architecture=details["architecture"],
        math=details["math"],
        use_cases=details["use_cases"],
        best_practices=details["best_practices"],
        dataset=details["dataset"],
        init_params=details["init_params"],
        module_path=module_map.get(model_name, model_lower),
        class_name=model_name,
    )

    output_path = output_dir / f"{model_lower}_tutorial.md"
    with open(output_path, "w") as f:
        f.write(content)

    return output_path


def main():
    """Generate first batch of 10 tutorials."""

    base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")
    tutorials_dir = base_dir / "docs" / "source" / "tutorials"

    # First 10 models with full detailed content
    batch_1_models = [
        "DCN",
        "DeepFM",
        "GNNRec",
        "MIND",
        "NASRec",
        "SASRec",
        "NCF",
        "LightGCN",
        "DIEN",
        "DIN",
    ]

    print("=" * 70)
    print("Regenerating First 10 Model Tutorials with UNIQUE Content")
    print("=" * 70)
    print()

    success_count = 0

    for model_name in batch_1_models:
        try:
            output_path = generate_tutorial(model_name, tutorials_dir)
            if output_path:
                print(f"  ✓ {model_name}: {output_path.name}")
                success_count += 1
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")

    print()
    print("=" * 70)
    print(f"Successfully regenerated {success_count}/{len(batch_1_models)} tutorials")
    print("=" * 70)
    print()
    print("Each tutorial includes:")
    print("  • Specific architecture explanation")
    print("  • Real mathematical formulations")
    print("  • Unique use cases and comparisons")
    print("  • Model-specific best practices")
    print("  • Complete cr_learn examples")


if __name__ == "__main__":
    main()
