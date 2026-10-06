#!/usr/bin/env python3
"""Fit several CoreRec models on the same tiny dataset with the same three calls.

Every model in corerec.engines takes fit(user_ids, item_ids, ratings) and
answers recommend(user_id, top_k). Swapping models is a one-word change.

    python examples/engines_quickstart.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from corerec.engines import ALS, DCN, DeepFM, EASE, ItemKNN, LightGCN, MultVAE, SASRec

# Three users who like disjoint groups of items, one row per interaction.
user_ids = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
item_ids = [10, 11, 12, 20, 21, 22, 30, 31, 32, 10, 11]
ratings = [1.0] * len(user_ids)

models = {
    "ALS": ALS(factors=8, iterations=5),
    "ItemKNN": ItemKNN(),
    "EASE": EASE(),
    "LightGCN": LightGCN(n_factors=8, epochs=20),
    "MultVAE": MultVAE(epochs=20),
    "DCN": DCN(embedding_dim=8, epochs=3, batch_size=4),
    "DeepFM": DeepFM(embedding_dim=8, epochs=3, batch_size=4),
    "SASRec": SASRec(hidden_units=8, num_blocks=1, epochs=3, max_seq_length=5),
}

for name, model in models.items():
    model.fit(user_ids=user_ids, item_ids=item_ids, ratings=ratings)
    # User 4 has seen 10 and 11, so a good model suggests 12 first.
    recs = model.recommend(4, top_k=3, exclude_items=[10, 11])
    print(f"{name:<9} recommends for user 4: {recs}")
