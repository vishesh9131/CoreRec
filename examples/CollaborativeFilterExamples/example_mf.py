# Copyright 2023 The UnionizedFilterEngine Authors(@vishesh9131). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
import random

# Import the MatrixFactorization class from collaborative
from corerec.engines.collaborative.mf_base.matrix_factorization import MatrixFactorization
import cr_learn.ml_1m as ml

from corerec.judge import judge

# Load MovieLens dataset
data = ml.load()

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load user and item data
usr_df = data["users"]
items_df = data["movies"]  # Load items data
print("User data loaded with columns:", usr_df.columns.tolist())
print("Item data loaded with columns:", items_df.columns.tolist())

# Main script
if __name__ == "__main__":
    print("Loading dataset...")
    ratings_df = data["ratings"]
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    print("Training model...")
    mf_model = MatrixFactorization(
        k=10, learning_rate=0.01, lambda_reg=0.02, max_iter=20, use_bias=True, verbose=True, seed=42
    )

    mf_model.fit(
        train_df["user_id"].tolist(), train_df["movie_id"].tolist(), train_df["rating"].tolist()
    )

    print("Evaluating model...")
    k_values = [5, 10, 20]
    # Group the test data by user_id and get the set of movie_ids for each user
    relevant_items = test_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    for k in k_values:
        hit_rates = []
        ndcgs = []

        for user_id in test_df["user_id"].unique():
            recommended_items = mf_model.recommend(user_id, top_n=k)
            user_relevant_items = relevant_items.get(user_id, set())

            hit_rate = judge.hit_rate_at_k(recommended_items, user_relevant_items, k)
            ndcg = judge.ndcg_at_k(recommended_items, user_relevant_items, k)

            hit_rates.append(hit_rate)
            ndcgs.append(ndcg)

        avg_hit_rate = np.mean(hit_rates)
        avg_ndcg = np.mean(ndcgs)

        print(f"Metrics for k={k}: Hit Ratio={avg_hit_rate:.4f}, NDCG={avg_ndcg:.4f}")

    # Save the model
    # mf_model.save_model("mf_model.pkl")
    # print("Model saved as 'mf_model.pkl'")

    sample_user = test_df["user_id"].iloc[0]
    recommendations = mf_model.recommend(sample_user, top_n=10)
    print(f"Recommendations for user {sample_user}: {recommendations}")
