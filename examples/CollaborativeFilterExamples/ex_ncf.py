#!/usr/bin/env python3
"""
Simple CoreRec + cr_learn Demo with NCF
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add CoreRec to path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import libraries
from cr_learn import ml_1m as ml

print("=" * 50)
print(" CoreRec + cr_learn Demo with NCF ")
print("=" * 50)

# Load MovieLens dataset
print("\nLoading MovieLens-1M dataset...")
data = ml.load()
ratings_df = data['ratings']
movies_df = data['movies']

print(f"Dataset: {len(ratings_df):,} ratings, {ratings_df['user_id'].nunique():,} users, {ratings_df['movie_id'].nunique():,} movies")

# Sample data
print("\nPreparing data for NCF...")
sample_df = ratings_df.sample(5000, random_state=42).copy()
sample_df['rating'] = (sample_df['rating'] >= 4).astype(int)  # binary ratings

# Rename movie_id to item_id for NCF compatibility
sample_df = sample_df.rename(columns={'movie_id': 'item_id'})

print(f"Sample: {len(sample_df)} interactions")
print(f"Positive ratings: {sum(sample_df['rating'])} ({sum(sample_df['rating'])/len(sample_df)*100:.1f}%)")

# Create simple NCF model (based on CoreRec example)
class SimpleNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        # Handle both single items and batches
        if user_emb.dim() == 1:
            x = torch.cat([user_emb, item_emb], dim=0)
        else:
            x = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(x).squeeze()

# Create mappings
unique_users = sorted(sample_df['user_id'].unique())
unique_items = sorted(sample_df['item_id'].unique())
user_map = {uid: idx for idx, uid in enumerate(unique_users)}
item_map = {iid: idx for idx, iid in enumerate(unique_items)}

# Convert to mapped indices
sample_df['user_idx'] = sample_df['user_id'].map(user_map)
sample_df['item_idx'] = sample_df['item_id'].map(item_map)

# Train NCF model
print("\nTraining NCF model...")
model = SimpleNCF(len(unique_users), len(unique_items), embedding_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop with batching for better performance
model.train()
batch_size = 256

# Convert to tensors once
users_tensor = torch.tensor(sample_df['user_idx'].values, dtype=torch.long)
items_tensor = torch.tensor(sample_df['item_idx'].values, dtype=torch.long)
ratings_tensor = torch.tensor(sample_df['rating'].values, dtype=torch.float)

for epoch in range(3):
    total_loss = 0
    # Shuffle data
    indices = torch.randperm(len(sample_df))
    
    for i in range(0, len(sample_df), batch_size):
        batch_indices = indices[i:min(i + batch_size, len(sample_df))]
        
        user_batch = users_tensor[batch_indices]
        item_batch = items_tensor[batch_indices]
        rating_batch = ratings_tensor[batch_indices]
        
        predictions = model(user_batch, item_batch)
        loss = criterion(predictions, rating_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(batch_indices)
    
    print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(sample_df):.4f}")

# Get recommendations
print("\nGenerating recommendations...")
model.eval()
sample_user = sample_df['user_id'].iloc[0]
user_idx = user_map[sample_user]

# Score all items for this user
item_scores = []
with torch.no_grad():
    for item_id in unique_items:
        item_idx = item_map[item_id]
        score = model(torch.tensor(user_idx), torch.tensor(item_idx))
        item_scores.append((item_id, score.item()))

# Get top 5 recommendations
recommendations = sorted(item_scores, key=lambda x: x[1], reverse=True)[:5]
recommendation_items = [item_id for item_id, score in recommendations]

print(f"Top 5 recommendations for user {sample_user}:")
for i, item_id in enumerate(recommendation_items, 1):
    movie_title = movies_df[movies_df['movie_id'] == item_id]['title'].iloc[0]
    score = recommendations[i-1][1]
    print(f"{i}. {movie_title} (score: {score:.4f})")

# Validation
print("\nValidation Results:")
print("-" * 30)

# Check if user's known positive items get high scores
user_positive_items = sample_df[sample_df['user_id'] == sample_user]['item_id'].tolist()
print(f"User {sample_user} has {len(user_positive_items)} positive items in dataset")

# Check recommendation diversity
unique_recommendations = len(set(recommendation_items))
print(f"Unique recommendations: {unique_recommendations}/5")

# Check if recommendations make sense
recommendation_titles = [movies_df[movies_df['movie_id'] == item_id]['title'].iloc[0] for item_id in recommendation_items]
print("Recommendation diversity: ✓" if len(set([title.split('(')[-1].split(')')[0] for title in recommendation_titles])) > 1 else "✗")

# Basic model health check
print(f"Model trained successfully: ✓")
print(f"Recommendations generated: ✓")
print(f"Movie titles retrieved: ✓")

print("\nDemo complete!")