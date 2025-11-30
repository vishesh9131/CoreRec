#!/usr/bin/env python3
"""
Generate Sphinx tutorials for 21 models with CORRECT cr_learn API

Models: All batch 1 (10) + batch 2 (11): BPR, SVD, RBM, AutoInt, ALS, Bert4Rec, DLRM, FFM, Caser, BiVAE, AFM
"""

import sys

sys.path.insert(0, "/Users/visheshyadav/Documents/GitHub/CoreRec/scripts")
from pathlib import Path
from model_database import MODEL_DATABASE

# CORRECT cr_learn template
TUTORIAL_TEMPLATE_FIXED = """# {model_name} Tutorial: {full_name}

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
from cr_learn import ml_1m
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset with CORRECT API
data = ml_1m.load()  # Returns dict with 'ratings', 'users', 'movies'
ratings_df = data['ratings']  # DataFrame with user_id, movie_id, rating, timestamp

print(f"Loaded {{len(ratings_df)}} ratings")

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
print(f"Predicted score: {{score:.3f}}")

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

print(f"Top-10 recommendations for User {{user_id}}:")
for rank, item_id in enumerate(recommendations, 1):
    print(f"  {{rank}}. Item {{item_id}}")
```

### Step 6: Evaluate

```python
from sklearn.metrics import mean_squared_error
import numpy as np

# Predict all test ratings
test_pred = [model.predict(u, i) for u, i in zip(test_users, test_items)]
rmse = np.sqrt(mean_squared_error(test_ratings, test_pred))
print(f"Test RMSE: {{rmse:.4f}}")
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
- [GitHub Repository](https://github.com/vishesh9131/CoreRec)
"""


def generate_tutorial(model_name: str, output_dir: Path):
    """Generate tutorial with correct cr_learn API."""

    if model_name not in MODEL_DATABASE:
        print(f"  ✗ {model_name}: No content in database")
        return None

    details = MODEL_DATABASE[model_name]
    model_lower = model_name.lower().replace("-", "_")

    # Module path mapping
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
        "BPR": "bpr",
        "SVD": "svd",
        "RBM": "rbm",
        "AutoInt": "autoint",
        "ALS": "als",
        "Bert4Rec": "bert4rec",
        "DLRM": "dlrm",
        "FFM": "ffm",
        "Caser": "caser",
        "BiVAE": "bivae",
        "AFM": "afm",
    }

    content = TUTORIAL_TEMPLATE_FIXED.format(
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
    """Generate all 21 tutorials."""

    base_dir = Path("/Users/visheshyadav/Documents/GitHub/CoreRec")
    tutorials_dir = base_dir / "docs" / "source" / "tutorials"

    # All 21 models
    all_models = list(MODEL_DATABASE.keys())

    print("=" * 70)
    print(f"Generating {len(all_models)} Sphinx Tutorials")
    print("=" * 70)
    print()

    success_count = 0

    for model_name in all_models:
        try:
            output_path = generate_tutorial(model_name, tutorials_dir)
            if output_path:
                print(f"  ✓ {model_name}: {output_path.name}")
                success_count += 1
        except Exception as e:
            print(f"  ✗ {model_name}: {e}")

    print()
    print("=" * 70)
    print(f"Successfully generated {success_count}/{len(all_models)} tutorials")
    print("=" * 70)
    print()
    print("✅ All tutorials use CORRECT cr_learn API:")
    print("   - ml_1m.load() instead of load_dataset()")
    print("   - sklearn train_test_split")
    print("   - Proper DataFrame access")


if __name__ == "__main__":
    main()
