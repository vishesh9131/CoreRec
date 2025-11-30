#!/usr/bin/env python3

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Using new Engine-Level Organization API
from corerec import engines
from examples.utils_example_data import load_interactions, build_csr_from_interactions


if __name__ == "__main__":
    data = load_interactions("crlearn")
    users = data["users"]
    items = data["items"]
    timestamps = data.get("timestamps", list(range(len(users))))
    
    # For SASRec, we need to preserve temporal order, so we'll pass the raw interaction data
    # Build sequences from raw interactions with timestamps
    from collections import defaultdict
    user_item_timestamps = defaultdict(list)
    for u, i, t in zip(users, items, timestamps):
        user_item_timestamps[u].append((i, t))
    
    # Sort by timestamp for each user
    for u in user_item_timestamps:
        user_item_timestamps[u].sort(key=lambda x: x[1])
    
    # Get unique users and items
    uniq_users = sorted(set(users))
    uniq_items = sorted(set(items))
    
    # Build interaction matrix for compatibility (but sequences will use ordered data)
    mat, _, _ = build_csr_from_interactions(users, items)
    interaction_matrix = mat.toarray() if hasattr(mat, 'toarray') else np.array(mat)
    
    # New API: Direct access to SASRec from engines
    # Use smaller learning rate and more stable settings for numerical stability
    model = engines.SASRec(
        hidden_units=32,  # Smaller model for stability
        num_blocks=1,
        num_heads=1,
        num_epochs=1,
        batch_size=128,  # Smaller batch size
        max_seq_length=50,
        device="cpu",
        learning_rate=5e-5,  # Very low learning rate for stability
        l2_reg=1e-4,  # Higher regularization
        loss_type='bce',  # BCE is more stable than BPR
        dropout_rate=0.0,  # No dropout for initial testing
    )
    # Pass interaction matrix and also provide ordered sequences via user_item_timestamps
    # We'll modify fit to use timestamps if available
    model.fit(user_ids=uniq_users, item_ids=uniq_items, interaction_matrix=interaction_matrix, 
              user_item_timestamps=user_item_timestamps)
    recs = model.recommend(uniq_users[0], top_n=10, exclude_seen=True)
    print("SASRec recommendations for", uniq_users[0], ":", recs)
