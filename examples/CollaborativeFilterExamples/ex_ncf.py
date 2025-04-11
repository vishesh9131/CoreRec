# examples/ijcai_ncf_example.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from collections import defaultdict
import os
from pathlib import Path

# Import base NCF model from CoreRec
from corerec.engines.unionizedFilterEngine.nn_base.ncf import NCF as BaseNCF

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Determine device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # device = 'mps'
    device = 'cpu'
print(f"Using device: {device}")

# Custom NCF implementation that fixes the issues with the original one
class CustomNCF(BaseNCF):
    def __init__(self, num_users, num_items, embedding_dim=64, mlp_layers=None, dropout=0.1):
        """Initialize a custom NCF model with fixed parameters."""
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers or [128, 64, 32]
        self.dropout = dropout
        self.device = device
        
        # Initialize user and item mappings
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        # Build PyTorch model
        self._build_model()
    
    def _build_model(self):
        """Build the PyTorch model without using pretrained embeddings."""
        # Create a simple PyTorch model for NCF
        self.user_embedding = torch.nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_dim)
        
        # MLP layers
        layers = []
        input_size = 2 * self.embedding_dim  # Concatenated user and item embeddings
        
        for output_size in self.mlp_layers:
            layers.append(torch.nn.Linear(input_size, output_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(self.dropout))
            input_size = output_size
        
        # Final prediction layer
        layers.append(torch.nn.Linear(input_size, 1))
        layers.append(torch.nn.Sigmoid())
        
        self.mlp = torch.nn.Sequential(*layers)
        self.model = self.mlp.to(self.device)
    
    def fit(self, data):
        """Train the model on the provided data."""
        # Preprocess data
        user_ids = data['user_id'].values
        item_ids = data['item_id'].values
        ratings = data['rating'].values
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        rating_tensor = torch.FloatTensor(ratings).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, rating_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(10):  # 10 epochs
            total_loss = 0
            for batch_user, batch_item, batch_rating in dataloader:
                # Forward pass
                user_emb = self.user_embedding(batch_user)
                item_emb = self.item_embedding(batch_item)
                x = torch.cat([user_emb, item_emb], dim=1)
                prediction = self.mlp(x).squeeze()
                
                # Compute loss
                loss = criterion(prediction, batch_rating)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}")
    
    def recommend(self, user_id, top_k=10, exclude_seen=True):
        """Generate recommendations for a user."""
        # Convert user_id to internal index if needed
        if user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
        else:
            user_idx = user_id  # Assume it's already an index
        
        # Get user embedding
        user_tensor = torch.LongTensor([user_idx]).to(self.device)
        user_emb = self.user_embedding(user_tensor)
        
        # Get scores for all items
        scores = []
        for item_idx in range(self.num_items):
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            item_emb = self.item_embedding(item_tensor)
            x = torch.cat([user_emb, item_emb], dim=1)
            score = self.mlp(x).item()
            scores.append((item_idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert item indices to original IDs
        recommendations = []
        for item_idx, score in scores[:top_k]:
            if item_idx in self.reverse_item_mapping:
                item_id = self.reverse_item_mapping[item_idx]
            else:
                item_id = item_idx  # Assume it's already an ID
            recommendations.append((item_id, score))
        
        return recommendations



def load_ijcai_data():
    """Load IJCAI dataset using cr_learn"""
    try:
        from cr_learn.ijcai import load
        data = load()
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_user_merchant_mappings(interactions):
    """Create user and merchant ID mappings to consecutive integers"""
    unique_users = interactions['user_id'].unique()
    unique_merchants = interactions['merchant_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    merchant_to_idx = {merchant: idx for idx, merchant in enumerate(unique_merchants)}
    
    return user_to_idx, merchant_to_idx, len(unique_users), len(unique_merchants)

def prepare_interaction_data(interactions, user_to_idx, merchant_to_idx):
    """Convert interactions to format for NCF"""
    interactions['user_idx'] = interactions['user_id'].map(user_to_idx)
    interactions['merchant_idx'] = interactions['merchant_id'].map(merchant_to_idx)
    
    # Create target labels (1 for interactions)
    interactions['label'] = 1
    
    return interactions

def generate_negative_samples(interactions, n_neg_per_pos, n_merchants):
    """Generate negative samples (user-merchant pairs with no interaction)"""
    neg_samples = []
    user_merchant_set = set(zip(interactions['user_idx'], interactions['merchant_idx']))
    
    for user_idx in tqdm(interactions['user_idx'].unique(), desc="Generating negative samples"):
        user_pos_merchants = interactions[interactions['user_idx'] == user_idx]['merchant_idx'].values
        
        # Keep track of negative merchants for this user
        neg_merchants = []
        while len(neg_merchants) < min(n_neg_per_pos * len(user_pos_merchants), n_merchants - len(user_pos_merchants)):
            merchant_idx = random.randint(0, n_merchants - 1)
            if merchant_idx not in user_pos_merchants and (user_idx, merchant_idx) not in user_merchant_set and merchant_idx not in neg_merchants:
                neg_merchants.append(merchant_idx)
                neg_samples.append({
                    'user_idx': user_idx,
                    'merchant_idx': merchant_idx,
                    'label': 0  # Negative sample
                })
    
    return pd.DataFrame(neg_samples)

def evaluate_model(model, test_data, k_values=[5, 10, 20]):
    """Evaluate model performance using Hit Rate and NDCG"""
    hr_dict = {k: [] for k in k_values}
    ndcg_dict = {k: [] for k in k_values}
    
    # Group test data by user
    user_items = defaultdict(list)
    for _, row in test_data.iterrows():
        user_items[row['user_idx']].append((row['merchant_idx'], row['label']))
    
    # For each user, get recommendations and evaluate
    for user_idx, items in tqdm(user_items.items(), desc="Evaluating"):
        # Get positive items for this user
        pos_items = [item_id for item_id, label in items if label == 1]
        if not pos_items:
            continue
            
        # Get all merchants this user hasn't interacted with
        all_items = set(range(model.num_items))
        user_items_set = set(item_id for item_id, _ in items)
        unknown_items = list(all_items - user_items_set)
        
        # Get recommendations
        recommendations = model.recommend(
            user_idx, 
            top_k=max(k_values), 
            exclude_seen=True
        )
        
        # Extract just the item IDs from recommendations
        rec_items = [item_id for item_id, _ in recommendations]
        
        # Calculate metrics
        for k in k_values:
            # Hit Rate: whether any positive item is in top-k
            hit = any(item_id in rec_items[:k] for item_id in pos_items)
            hr_dict[k].append(1.0 if hit else 0.0)
            
            # NDCG: normalized discounted cumulative gain
            dcg = 0.0
            idcg = 1.0  # Ideal DCG for a single relevant item
            for i, item_id in enumerate(rec_items[:k]):
                if item_id in pos_items:
                    dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
            ndcg_dict[k].append(dcg / idcg)
    
    # Calculate average metrics
    hr = {k: np.mean(values) for k, values in hr_dict.items()}
    ndcg = {k: np.mean(values) for k, values in ndcg_dict.items()}
    
    return hr, ndcg

def plot_metrics(hr_history, ndcg_history):
    """Plot Hit Rate and NDCG metrics over epochs"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hr_history)
    plt.title('Hit Rate@10')
    plt.xlabel('Epoch')
    plt.ylabel('HR@10')
    
    plt.subplot(1, 2, 2)
    plt.plot(ndcg_history)
    plt.title('NDCG@10')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG@10')
    
    plt.tight_layout()
    plt.savefig('ncf_metrics.png')
    plt.close()

def main():
    print("Loading IJCAI dataset...")
    data = load_ijcai_data()
    
    # Print dataset information
    print(f"Number of users: {len(data['users'])}")
    print(f"Number of merchants in training: {len(data['merchant_train']['merchant_id'].unique())}")
    
    # Create interaction dataframe from user-merchant interactions
    interactions = []
    for user_id, merchant_ids in data['user_merchant_interaction'].items():
        for merchant_id in merchant_ids:
            interactions.append({
                'user_id': user_id,
                'merchant_id': merchant_id
            })
    
    interactions_df = pd.DataFrame(interactions)
    print(f"Total interactions: {len(interactions_df)}")
    
    # Create ID mappings
    user_to_idx, merchant_to_idx, n_users, n_merchants = create_user_merchant_mappings(interactions_df)
    idx_to_merchant = {idx: merchant for merchant, idx in merchant_to_idx.items()}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    
    print(f"Number of unique users: {n_users}")
    print(f"Number of unique merchants: {n_merchants}")
    
    # Prepare interaction data
    processed_df = prepare_interaction_data(interactions_df, user_to_idx, merchant_to_idx)
    
    # Generate negative samples (1 negative per positive)
    print("Generating negative samples...")
    neg_samples_df = generate_negative_samples(processed_df, n_neg_per_pos=1, n_merchants=n_merchants)
    
    # Combine positive and negative samples
    full_df = pd.concat([processed_df, neg_samples_df], ignore_index=True)
    print(f"Total samples after adding negatives: {len(full_df)}")
    
    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)  # 0.125 of 0.8 = 0.1 of original
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare data for NCF model
    train_data = pd.DataFrame({
        'user_id': train_df['user_idx'],
        'item_id': train_df['merchant_idx'],
        'rating': train_df['label']
    })
    
    # Initialize custom NCF model
    model = CustomNCF(
        num_users=n_users,
        num_items=n_merchants,
        embedding_dim=32,
        mlp_layers=[64, 32, 16],
        dropout=0.2
    )
    
    # Set up mappings
    model.user_mapping = user_to_idx
    model.item_mapping = merchant_to_idx
    model.reverse_user_mapping = idx_to_user
    model.reverse_item_mapping = idx_to_merchant
    
    # Train the model
    print("Training NCF model...")
    model.fit(train_data)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    hr, ndcg = evaluate_model(model, test_df, k_values=[5, 10, 20])
    
    print("Test Metrics:")
    for k in [5, 10, 20]:
        print(f"HR@{k}: {hr[k]:.4f}, NDCG@{k}: {ndcg[k]:.4f}")
    
    # Sample users for recommendations
    sample_users = random.sample(list(data['users']['user_id']), 3)
    
    print("\nRecommendations for sample users:")
    for user_id in sample_users:
        print(f"\nUser ID: {user_id}")
        print("Merchant features examples:")
        user_merchants = data['user_merchant_interaction'].get(user_id, [])[:5]
        
        for merchant_id in user_merchants:
            if merchant_id in data['merchant_features']:
                print(f"- Merchant {merchant_id}: {data['merchant_features'][merchant_id]}")
        
        # Generate recommendations
        if user_id in user_to_idx:
            recommendations = model.recommend(user_id, top_k=5, exclude_seen=True)
            
            print(f"Top 5 recommended merchants:")
            for i, (merchant_id, score) in enumerate(recommendations, 1):
                features = data['merchant_features'].get(merchant_id, "No features available")
                print(f"{i}. Merchant {merchant_id} (score: {score:.4f}): {features}")
        else:
            print(f"User {user_id} not found in training data")

if __name__ == "__main__":
    main()
