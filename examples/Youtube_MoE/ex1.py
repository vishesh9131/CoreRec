import logging
from typing import List, Optional

from scipy.sparse import csr_matrix
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_base import (
    MatrixFactorizationBase,
)
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_recommender import (
    MatrixFactorizationRecommender,
)
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender
from corerec.engines.hybrid import HybridEngine
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)

# Step 1: Data Preparation
# Example user-item interaction matrix
interaction_data = [
    [1, 0, 3, 0],
    [0, 2, 0, 4],
    [3, 0, 0, 5],
    [0, 1, 2, 0],
]
interaction_matrix = csr_matrix(interaction_data)

# List of user IDs and item IDs
user_ids = [0, 1, 2, 3]  # Assuming zero-based indexing
item_ids = [0, 1, 2, 3]  # Assuming zero-based indexing

# Example item features for content-based filtering
item_features = [
    ["action", "thriller"],
    ["comedy"],
    ["action", "adventure"],
    ["documentary"],
]

# Convert item features to a list of strings
item_features_str = [" ".join(features) for features in item_features]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(item_features_str)

# Initialize Content-Based Recommender
tfidf_recommender = TFIDFRecommender(feature_matrix=tfidf_matrix)

# Step 2: Initialize Recommenders
collaborative_recommender = MatrixFactorizationRecommender(
    num_factors=20,
    learning_rate=0.01,
    reg_user=0.02,
    reg_item=0.02,
    epochs=20,
    early_stopping_rounds=5,
    n_threads=4,
)

# Step 3: Fit the Models
collaborative_recommender.fit(interaction_matrix)
tfidf_recommender.fit(data=None)  # Adjust based on actual implementation

# Step 4: Create a Hybrid Engine
hybrid_engine = HybridEngine(
    collaborative_engine=collaborative_recommender, content_engine=tfidf_recommender, alpha=0.5
)


# Step 5: Generate Recommendations
def get_recommendations(
    user_id: int, exclude_items: Optional[List[int]] = None, top_n: int = 10
) -> List[int]:
    """
    Generate top-N recommendations for a given user.

    Parameters:
    - user_id (int): The ID of the user.
    - exclude_items (Optional[List[int]]): List of item IDs to exclude from recommendations.
    - top_n (int): The number of recommendations to generate.

    Returns:
    - List[int]: List of recommended item IDs.
    """
    recommendations = hybrid_engine.recommend(user_id, top_n=top_n, exclude_items=exclude_items)
    return recommendations


# Example usage
user_id = 0  # Assuming user IDs start from 0
exclude_items = [0, 1]  # Items to exclude from recommendations
recommended_items = get_recommendations(user_id, exclude_items, top_n=2)
print(f"Recommended items for user {user_id}: {recommended_items}")
