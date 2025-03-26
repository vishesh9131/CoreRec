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
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# Import the MatrixFactorization class from unionizedFilterEngine
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization import MatrixFactorization
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
    # Load MovieLens dataset
    print("Loading MovieLens dataset...")
    ratings_df = data['ratings']
    print(f"Loaded dataset with {len(ratings_df)} interactions")
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} interactions")
    print(f"Test set: {len(test_df)} interactions")
    
    # Initialize and train the Matrix Factorization model
    print("\nTraining Matrix Factorization model...")
    mf_model = MatrixFactorization(
        k=10,  # Latent factor dimension
        learning_rate=0.01,
        lambda_reg=0.02,
        max_iter=20,
        use_bias=True,
        verbose=True,
        seed=42
    )
    
    # Fit the model
    mf_model.fit(
        user_ids=train_df['user_id'].tolist(),
        item_ids=train_df['movie_id'].tolist(),  # Use 'movie_id' instead of 'item_id'
        ratings=train_df['rating'].tolist()
    )
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    k_values = [5, 10, 20]
    hit_ratios = []
    ndcg_values = []
    
    for k in k_values:
        print(f"\nCalculating metrics for k={k}...")
        # Get relevant items for each user in the test set
        relevant_items = test_df.groupby('user_id')['movie_id'].apply(set).to_dict()
        
        # Calculate Hit Ratio@k and NDCG@k using judge
        hr = np.mean([
            judge.hit_rate_at_k(
                mf_model.recommend(user_id, top_n=k),
                relevant_items.get(user_id, set()),
                k
            )
            for user_id in test_df['user_id'].unique()
        ])
        
        ndcg = np.mean([
            judge.ndcg_at_k(
                mf_model.recommend(user_id, top_n=k),
                relevant_items.get(user_id, set()),
                k
            )
            for user_id in test_df['user_id'].unique()
        ])
        
        hit_ratios.append(hr)
        ndcg_values.append(ndcg)
        
        print(f"Hit Ratio@{k}: {hr:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
    
    # Save the model
    print("\nSaving the trained model...")
    mf_model.save_model("mf_model.pkl")
    print("Model saved as 'mf_model.pkl'")
    
    # Generate recommendations for a sample user
    sample_user = test_df['user_id'].iloc[0]
    print(f"\nGenerating recommendations for user {sample_user}...")
    recommendations = mf_model.recommend(sample_user, top_n=10)
    print("Top 10 recommendations:")
    for i, item in enumerate(recommendations):
        print(f"  {i+1}. Movie ID {item}")
    
    print("\nDone!") 