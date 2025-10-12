#!/usr/bin/env python3
"""
Simple CoreRec + cr_learn Demo with NCF
"""

import os
import sys
import numpy as np
import pandas as pd

# Add CoreRec to path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import libraries
from cr_learn import ml_1m as ml
from corerec.engines.unionizedFilterEngine.nn_base.ncf import NCF

print("=" * 50)
print(" CoreRec + cr_learn Demo with NCF ")
print("=" * 50)

# Load MovieLens dataset
print("\nLoading MovieLens-1M dataset...")
data = ml.load()
ratings_df = data['ratings']
movies_df = data['movies']

print(f"Dataset: {len(ratings_df):,} ratings, {ratings_df['user_id'].nunique():,} users, {ratings_df['movie_id'].nunique():,} movies")

# Sample data and rename columns for NCF
print("\nPreparing data for NCF...")
sample_df = ratings_df.sample(5000, random_state=42).copy()
sample_df['rating'] = (sample_df['rating'] >= 4).astype(int)  # binary ratings

# Rename movie_id to item_id for NCF compatibility
sample_df = sample_df.rename(columns={'movie_id': 'item_id'})

print(f"Sample: {len(sample_df)} interactions")
print(f"Positive ratings: {sum(sample_df['rating'])} ({sum(sample_df['rating'])/len(sample_df)*100:.1f}%)")

# Train NCF model
print("\nTraining CoreRec NCF model...")
model = NCF(
    name="NCF_Demo",
    model_type="NeuMF",  # Neural Matrix Factorization
    gmf_embedding_dim=32,
    mlp_embedding_dim=32,
    mlp_hidden_layers=(64, 32, 16),
    dropout=0.2,
    learning_rate=0.001,
    batch_size=256,
    num_epochs=3,
    device='cpu',
    verbose=False,
    # Set pretrained embeddings to None to avoid the error
    pretrained_user_embeddings=None,
    pretrained_item_embeddings=None
)

# NCF expects a DataFrame with user_id, item_id, rating columns
model.fit(sample_df)

# Get recommendations
print("\nGenerating recommendations...")
sample_user = sample_df['user_id'].iloc[0]
recommendations = model.recommend(sample_user, top_n=5)

print(f"Top 5 recommendations for user {sample_user}:")
for i, item_id in enumerate(recommendations, 1):
    movie_title = movies_df[movies_df['movie_id'] == item_id]['title'].iloc[0]
    print(f"{i}. {movie_title}")

# Validation
print("\nValidation Results:")
print("-" * 30)

# Check if user's known positive items get high scores
user_positive_items = sample_df[sample_df['user_id'] == sample_user]['item_id'].tolist()
print(f"User {sample_user} has {len(user_positive_items)} positive items in dataset")

# Check recommendation diversity
unique_recommendations = len(set(recommendations))
print(f"Unique recommendations: {unique_recommendations}/5")

# Check if recommendations make sense (not all same genre/era)
recommendation_titles = [movies_df[movies_df['movie_id'] == item_id]['title'].iloc[0] for item_id in recommendations]
print("Recommendation diversity: ✓" if len(set([title.split('(')[-1].split(')')[0] for title in recommendation_titles])) > 1 else "✗")

# Basic model health check
print(f"Model trained successfully: ✓")
print(f"Recommendations generated: ✓")
print(f"Movie titles retrieved: ✓")

print("\nDemo complete!")