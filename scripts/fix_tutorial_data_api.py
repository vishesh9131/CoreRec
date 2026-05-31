#!/usr/bin/env python3
"""Bulk-fix cr_learn.load_dataset usage in tutorial markdown files."""
import re
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "source" / "tutorials"

LOAD_BLOCK = """from cr_learn import ml_1m
from sklearn.model_selection import train_test_split

data = ml_1m.load()
ratings_df = data['ratings']
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

train_users = train_df['user_id'].values.tolist()
train_items = train_df['movie_id'].values.tolist()
train_ratings = train_df['rating'].values.tolist()

print(f"Loaded {len(ratings_df)} ratings")"""

REPLACEMENTS = [
    (r"data = cr_learn\.load_dataset\(['\"]movielens-100k['\"]\)", "data = ml_1m.load()\nratings_df = data['ratings']"),
    (r"print\(f\"Loaded \{len\(data\.ratings\)\} ratings\"\)", "print(f\"Loaded {len(ratings_df)} ratings\")"),
    (r"train_data, test_data = data\.train_test_split\(test_size=0\.2\)", "train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)"),
    (r"train_data\.user_ids", "train_users"),
    (r"train_data\.item_ids", "train_items"),
    (r"train_data\.ratings", "train_ratings"),
    (r"test_data\.user_ids", "test_df['user_id'].values"),
    (r"test_data\.item_ids", "test_df['movie_id'].values"),
    (r"test_data\.ratings", "test_df['rating'].values"),
    (r"train_data\.get_user_items\(1\)", "[]"),
    (r"from corerec\.metrics import rmse, ndcg_at_k", "from sklearn.metrics import mean_squared_error\nimport numpy as np"),
    (r"predictions = \[model\.predict\(u, i\) for u, i, r in test_data\]\ntest_rmse = rmse\(test_data\.ratings, predictions\)",
     "test_pred = [model.predict(u, i) for u, i in zip(test_df['user_id'].values[:100], test_df['movie_id'].values[:100])]\ntest_rmse = np.sqrt(mean_squared_error(test_df['rating'].values[:100], test_pred))"),
    (r"ndcg = ndcg_at_k\(model, test_data, k=10\)\nprint\(f\"NDCG@10: \{ndcg:.4f\}\"\)\n", ""),
]


def ensure_imports(content: str) -> str:
    if "load_dataset" not in content and "ml_1m.load()" in content:
        return content
    if "load_dataset" not in content:
        return content

    if "from cr_learn import ml_1m" not in content:
        content = content.replace("import cr_learn\n", "")
        # inject after first python block opener in step 1 if possible
        content = re.sub(
            r"(```python\n)(from corerec[^\n]+\n)",
            r"\1\2from cr_learn import ml_1m\nfrom sklearn.model_selection import train_test_split\n",
            content,
            count=1,
        )
    return content


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "load_dataset" not in text:
        return False

    original = text
    text = ensure_imports(text)
    for pattern, repl in REPLACEMENTS:
        text = re.sub(pattern, repl, text)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    updated = 0
    for path in sorted(TUTORIAL_DIR.glob("*_tutorial.md")):
        if fix_file(path):
            updated += 1
            print(f"Updated: {path.name}")
    print(f"\nDone. Updated {updated} files.")


if __name__ == "__main__":
    main()
