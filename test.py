#!/usr/bin/env python3
"""
Simple Recommendation Tower using Monolith model from CoreRec
Dataset: MovieLens-1M from cr_learn
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add CoreRec to path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import cr_learn dataset
from cr_learn import ml_1m as ml

# Import CoreRec Monolith model
from corerec.engines.monolith.monolith_model import MonolithModel
from corerec.core.towers import MLPTower


class MovieLensDataset(Dataset):
    """Simple dataset for MovieLens data"""
    
    def __init__(self, users, items, ratings, user_map, item_map):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.user_map = user_map
        self.item_map = item_map
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_map[self.users[idx]], dtype=torch.long),
            'item_id': torch.tensor(self.item_map[self.items[idx]], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }


class RecommendationTower(nn.Module):
    """Simple recommendation tower architecture"""
    
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        
        # user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # tower layers
        self.user_tower = MLPTower(
            name="user_tower",
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            config={
                'hidden_dims': [embedding_dim, embedding_dim//2],
                'activation': 'relu',
                'dropout': 0.1
            }
        )
        
        self.item_tower = MLPTower(
            name="item_tower", 
            input_dim=embedding_dim,
            output_dim=embedding_dim,
            config={
                'hidden_dims': [embedding_dim, embedding_dim//2],
                'activation': 'relu',
                'dropout': 0.1
            }
        )
        
        # prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        # get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # pass through towers
        user_features = self.user_tower(user_emb)
        item_features = self.item_tower(item_emb)
        
        # concatenate features
        combined = torch.cat([user_features, item_features], dim=1)
        
        # prediction
        output = self.prediction_head(combined)
        return output.squeeze()


def train_model(model, dataloader, num_epochs=50):
    """Simple training loop"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            user_ids = batch['user_id']
            item_ids = batch['item_id']
            ratings = batch['rating']
            
            # forward pass
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def get_recommendations(model, user_id, item_ids, user_map, item_map, top_k=10):
    """Get top-k recommendations for a user"""
    
    model.eval()
    user_tensor = torch.tensor([user_map[user_id]] * len(item_ids), dtype=torch.long)
    item_tensor = torch.tensor([item_map[item_id] for item_id in item_ids], dtype=torch.long)
    
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
    
    # get top-k items
    top_indices = torch.topk(scores, top_k).indices
    top_items = [item_ids[i] for i in top_indices]
    top_scores = [scores[i].item() for i in top_indices]
    
    return list(zip(top_items, top_scores))


def validate_recommendations(model, test_users, test_items, test_ratings, user_map, item_map, movies_df):
    """Validate recommendation quality using multiple metrics"""
    
    print("\n" + "="*60)
    print(" VALIDATION RESULTS ")
    print("="*60)
    
    model.eval()
    
    # 1. Check if model can reproduce known positive ratings
    print("\n1. Testing on Known Positive Ratings:")
    known_positive = []
    for i in range(min(100, len(test_users))):
        user_id = test_users[i]
        item_id = test_items[i]
        rating = test_ratings[i]
        
        if rating == 1:  # positive rating
            user_tensor = torch.tensor([user_map[user_id]], dtype=torch.long)
            item_tensor = torch.tensor([item_map[item_id]], dtype=torch.long)
            
            with torch.no_grad():
                score = model(user_tensor, item_tensor).item()
            
            known_positive.append(score)
    
    if known_positive:
        avg_score_positive = np.mean(known_positive)
        print(f"   Average score for known positive ratings: {avg_score_positive:.4f}")
        print(f"   Expected: > 0.5 (model should give high scores to positive items)")
    
    # 2. Check if model gives low scores to random items
    print("\n2. Testing on Random Items:")
    random_scores = []
    sample_users = list(set(test_users))[:10]
    all_items = list(set(test_items))
    
    for user_id in sample_users:
        # get 5 random items this user hasn't interacted with
        user_items = set([test_items[i] for i in range(len(test_users)) if test_users[i] == user_id])
        candidate_items = [item for item in all_items if item not in user_items]
        
        if len(candidate_items) >= 5:
            random_items = np.random.choice(candidate_items, 5, replace=False)
            
            for item_id in random_items:
                user_tensor = torch.tensor([user_map[user_id]], dtype=torch.long)
                item_tensor = torch.tensor([item_map[item_id]], dtype=torch.long)
                
                with torch.no_grad():
                    score = model(user_tensor, item_tensor).item()
                
                random_scores.append(score)
    
    if random_scores:
        avg_score_random = np.mean(random_scores)
        print(f"   Average score for random items: {avg_score_random:.4f}")
        print(f"   Expected: < 0.5 (model should give low scores to random items)")
    
    # 3. Check score distribution
    print("\n3. Score Distribution Analysis:")
    all_scores = known_positive + random_scores
    if all_scores:
        print(f"   Score range: {min(all_scores):.4f} to {max(all_scores):.4f}")
        print(f"   Score std: {np.std(all_scores):.4f}")
        print(f"   Scores > 0.8: {sum(1 for s in all_scores if s > 0.8)}")
        print(f"   Scores < 0.2: {sum(1 for s in all_scores if s < 0.2)}")
    
    # 4. Check if recommendations make sense (diversity)
    print("\n4. Recommendation Diversity Check:")
    sample_user = test_users[0]
    unique_items = list(set(test_items))
    recommendations = get_recommendations(model, sample_user, unique_items, user_map, item_map, top_k=10)
    
    print(f"   Top 10 recommendations for user {sample_user}:")
    for i, (item_id, score) in enumerate(recommendations):
        # try to get movie title if available
        movie_info = ""
        if movies_df is not None and item_id in movies_df['movie_id'].values:
            movie_title = movies_df[movies_df['movie_id'] == item_id]['title'].iloc[0]
            movie_info = f" ({movie_title})"
        
        print(f"     {i+1}. Item {item_id}{movie_info}: {score:.4f}")
    
    # 5. Check for overfitting (training vs validation performance)
    print("\n5. Overfitting Check:")
    print("   This would require a separate validation set - implement if needed")


def check_data_quality(users, items, ratings, movies_df):
    """Check the quality and characteristics of the data"""
    
    print("\n" + "="*60)
    print(" DATA QUALITY CHECK ")
    print("="*60)
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Total interactions: {len(users):,}")
    print(f"  Unique users: {len(set(users)):,}")
    print(f"  Unique items: {len(set(items)):,}")
    print(f"  Positive ratings: {sum(ratings):,} ({sum(ratings)/len(ratings)*100:.1f}%)")
    print(f"  Negative ratings: {len(ratings)-sum(ratings):,} ({(len(ratings)-sum(ratings))/len(ratings)*100:.1f}%)")
    
    # Sparsity
    sparsity = 1 - (len(users) / (len(set(users)) * len(set(items))))
    print(f"  Dataset sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    
    # User activity distribution
    user_counts = pd.Series(users).value_counts()
    print(f"\nUser Activity Distribution:")
    print(f"  Most active user: {user_counts.max()} interactions")
    print(f"  Least active user: {user_counts.min()} interactions")
    print(f"  Average interactions per user: {user_counts.mean():.2f}")
    
    # Item popularity distribution
    item_counts = pd.Series(items).value_counts()
    print(f"\nItem Popularity Distribution:")
    print(f"  Most popular item: {item_counts.max()} interactions")
    print(f"  Least popular item: {item_counts.min()} interactions")
    print(f"  Average interactions per item: {item_counts.mean():.2f}")
    
    # Check for data consistency
    print(f"\nData Consistency:")
    print(f"  All users in range: {min(users) >= 0 and max(users) < len(set(users))}")
    print(f"  All items in range: {min(items) >= 0 and max(items) < len(set(items))}")
    print(f"  Ratings are binary: {set(ratings) == {0, 1}}")


def main():
    print("\n" + "="*60)
    print(" Simple Recommendation Tower with Monolith Architecture ")
    print(" Dataset: MovieLens-1M from cr_learn ")
    print(" Model: Custom Tower + Monolith components ")
    print("="*60)
    
    # Load MovieLens-1M dataset
    print("\nLoading MovieLens-1M dataset...")
    data = ml.load()
    ratings_df = data['ratings']
    movies_df = data['movies']
    
    print(f"Dataset loaded:")
    print(f"  Total ratings: {len(ratings_df):,}")
    print(f"  Unique users: {ratings_df['user_id'].nunique():,}")
    print(f"  Unique movies: {ratings_df['movie_id'].nunique():,}")
    
    # Sample data for demo
    print("\nSampling 10000 interactions for demo...")
    sample_df = ratings_df.sample(10000, random_state=42)
    
    users = sample_df['user_id'].values
    items = sample_df['movie_id'].values
    ratings = (sample_df['rating'] >= 4).astype(float).values  # binary ratings
    
    # Create ID mappings (this fixes the embedding issue)
    unique_users = sorted(set(users))
    unique_items = sorted(set(items))
    
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    print(f"Sample data:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Positive ratings: {sum(ratings)}")
    
    # Check data quality
    check_data_quality(users, items, ratings, movies_df)
    
    # Split into train/test for validation
    print("\nCreating train/test split for validation...")
    n_test = int(len(users) * 0.2)
    indices = list(range(len(users)))
    np.random.shuffle(indices)
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    train_users = [users[i] for i in train_idx]
    train_items = [items[i] for i in train_idx]
    train_ratings = [ratings[i] for i in train_idx]
    
    test_users = [users[i] for i in test_idx]
    test_items = [items[i] for i in test_idx]
    test_ratings = [ratings[i] for i in test_idx]
    
    print(f"Train set: {len(train_users)} interactions")
    print(f"Test set: {len(test_users)} interactions")
    
    # Create dataset and dataloader
    dataset = MovieLensDataset(train_users, train_items, train_ratings, user_map, item_map)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Create model
    print("\nCreating recommendation tower model...")
    model = RecommendationTower(num_users, num_items, embedding_dim=64)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\nTraining model...")
    train_model(model, dataloader, num_epochs=3)
    
    # Validate recommendations
    validate_recommendations(model, test_users, test_items, test_ratings, user_map, item_map, movies_df)
    
    print("\n" + "="*60)
    print(" Recommendation Tower Demo Complete! ")
    print("="*60)


if __name__ == "__main__":
    main()