import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os


# Import our SUMModel implementation
# from corerec.engines.unionizedFilterEngine.sum import SUMModel
import corerec.uf_engine as uf

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Check for available device - adding MPS support for Mac
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # Enable MPS fallback to CPU for unsupported operations
    torch.backends.mps.enable_fallback_to_cpu = False
    device = 'mps'
print(f"Using device: {device}")

# Create a simplified sample dataset
def generate_realistic_dataset(n_users=200, n_items=100, n_interactions=5000):
    """Generate a smaller synthetic dataset with realistic user preference patterns."""
    
    # Generate user IDs
    user_ids = list(range(1, n_users + 1))
    
    # Generate item IDs
    item_ids = list(range(1, n_items + 1))
    
    # Create item categories (10 categories)
    n_categories = 10
    item_categories = {}
    for item_id in item_ids:
        # Assign each item to 1-3 categories
        n_cats = random.randint(1, 3)
        item_categories[item_id] = random.sample(range(n_categories), n_cats)
    
    # Create user preferences (each user has 2-4 preferred categories)
    user_preferences = {}
    for user_id in user_ids:
        n_prefs = random.randint(2, 4)
        user_preferences[user_id] = random.sample(range(n_categories), n_prefs)
        
    # Generate interactions with timestamps
    interactions = []
    for user_id in user_ids:
        # Number of interactions per user follows a power law
        n_user_interactions = np.random.pareto(1.5) * 10 + 5
        n_user_interactions = min(int(n_user_interactions), 100)
        
        # Get user's preferred categories
        preferred_cats = user_preferences[user_id]
        
        # Generate sequence of items for this user
        for i in range(n_user_interactions):
            # 80% chance to select from preferred categories
            if random.random() < 0.8:
                # Select a random preferred category
                category = random.choice(preferred_cats)
                
                # Find items in this category
                category_items = [item for item, cats in item_categories.items() if category in cats]
                
                # Select a random item from this category
                if category_items:
                    item_id = random.choice(category_items)
                else:
                    item_id = random.choice(item_ids)
            else:
                # Random exploration
                item_id = random.choice(item_ids)
            
            # Generate timestamp (days since epoch) with recency bias
            timestamp = int(i * 10 + random.randint(0, 5))
            
            interactions.append((user_id, item_id, timestamp))
    
    # Convert to DataFrame
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'timestamp'])
    
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Remove duplicates (keep the first occurrence)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
    # Limit to n_interactions
    if len(df) > n_interactions:
        df = df.sample(n_interactions, random_state=42)
        df = df.sort_values(['user_id', 'timestamp'])
    
    return df

# Generate improved sample data
print("Generating realistic dataset...")
df = generate_realistic_dataset()
print(f"Generated dataset with {len(df)} interactions")

# Display sample of the data
print("\nSample data:")
print(df.head())

# Basic statistics
print("\nDataset statistics:")
print(f"Number of users: {df['user_id'].nunique()}")
print(f"Number of items: {df['item_id'].nunique()}")
print(f"Average interactions per user: {df.groupby('user_id').size().mean():.2f}")

# Filter users with at least 5 interactions for better sequence modeling
min_interactions = 5
user_counts = df.groupby('user_id').size()
valid_users = user_counts[user_counts >= min_interactions].index
df = df[df['user_id'].isin(valid_users)]
print(f"Filtered dataset: {len(df)} interactions, {len(valid_users)} users")

# Split data into train and test sets
# Use the last 2 interactions of each user for validation and testing
print("\nSplitting data into train, validation, and test sets...")
test_df = df.groupby('user_id').tail(1).copy()
temp_df = df.drop(test_df.index)
val_df = temp_df.groupby('user_id').tail(1).copy()
train_df = temp_df.drop(val_df.index).copy()

print(f"Training set: {len(train_df)} interactions")
print(f"Validation set: {len(val_df)} interactions")
print(f"Test set: {len(test_df)} interactions")

# Define evaluation functions first so they're available for early stopping
def calculate_hit_ratio(model, test_df, k=10):
    """Calculate Hit Ratio@k for the model."""
    hits = 0
    total = 0
    
    for user_id, group in tqdm(test_df.groupby('user_id'), desc="Evaluating"):
        # Skip users not in training data
        if user_id not in model.user_id_map:
            continue
            
        # Get the actual item the user interacted with
        actual_item = group['item_id'].iloc[0]
        
        # Get recommendations for the user
        recommendations = model.recommend(user_id, top_n=k)
        
        # Check if the actual item is in the recommendations
        if actual_item in recommendations:
            hits += 1
        
        total += 1
    
    return hits / total if total > 0 else 0

def calculate_ndcg(model, test_df, k=10):
    """Calculate NDCG@k for the model."""
    ndcg_sum = 0
    total = 0
    
    for user_id, group in tqdm(test_df.groupby('user_id'), desc="Evaluating"):
        # Skip users not in training data
        if user_id not in model.user_id_map:
            continue
            
        # Get the actual item the user interacted with
        actual_item = group['item_id'].iloc[0]
        
        # Get recommendations for the user
        recommendations = model.recommend(user_id, top_n=k)
        
        # Calculate DCG
        dcg = 0
        if actual_item in recommendations:
            # Position of the item (0-indexed)
            pos = recommendations.index(actual_item)
            # DCG formula: 1/log2(pos+2)
            dcg = 1 / np.log2(pos + 2)
        
        # Ideal DCG is always 1 (item at position 0)
        idcg = 1
        
        # NDCG = DCG / IDCG
        ndcg_sum += dcg / idcg
        total += 1
    
    return ndcg_sum / total if total > 0 else 0


# Improved SUMModel configuration
print("\nInitializing SUMModel with improved configuration...")
model = uf.sum.SUMModel(
    embedding_dim=64,
    num_interests=4,
    interest_dim=32,
    routing_iterations=2,
    dropout_rate=0.3,
    l2_reg=1e-6,
    learning_rate=0.001,
    batch_size=64,
    epochs=5,
    sequence_length=20,
    device=device
)

# Add error handling for MPS device
try:
    print(f"Training model on {model.device}...")
    
    # First fit to initialize the model with all users/items
    print("Initial model fitting...")
    model.fit(
        user_ids=train_df['user_id'].tolist(),
        item_ids=train_df['item_id'].tolist(),
        timestamps=train_df['timestamp'].tolist()
    )
except RuntimeError as e:
    if "MPS" in str(e):
        print("MPS device encountered an error. Falling back to CPU...")
        model = uf.sum.SUMModel(
            embedding_dim=64,
            num_interests=4,
            interest_dim=32,
            routing_iterations=2,
            dropout_rate=0.3,
            l2_reg=1e-6,
            learning_rate=0.001,
            batch_size=64,
            epochs=5,
            sequence_length=20,
            device='cpu'
        )
        print("Training model on CPU...")
        model.fit(
            user_ids=train_df['user_id'].tolist(),
            item_ids=train_df['item_id'].tolist(),
            timestamps=train_df['timestamp'].tolist()
        )
    else:
        raise e

# Train with early stopping based on validation performance
best_val_hr = 0
patience = 3
patience_counter = 0
best_epoch = 0

# Then train for each epoch with early stopping
for epoch in range(1, model.epochs + 1):
    print(f"\nEpoch {epoch}/{model.epochs}")
    
    # Train on training set - without specifying epochs parameter
    model.fit(
        user_ids=train_df['user_id'].tolist(),
        item_ids=train_df['item_id'].tolist(),
        timestamps=train_df['timestamp'].tolist()
    )
    
    # Evaluate on validation set
    val_hr = calculate_hit_ratio(model, val_df, k=10)
    print(f"Validation Hit Ratio@10: {val_hr:.4f}")
    
    # Check for improvement
    if val_hr > best_val_hr:
        best_val_hr = val_hr
        patience_counter = 0
        best_epoch = epoch
        
        # Save the best model
        model.save_model('best_sum_model.pkl')
        print("New best model saved!")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

print(f"\nTraining completed. Best epoch: {best_epoch} with validation HR@10: {best_val_hr:.4f}")

# Load the best model for evaluation
print("Loading best model for evaluation...")
model = uf.sum.SUMModel.load_model('best_sum_model.pkl')

# Evaluate the model
print("\nEvaluating model on test set...")

# Calculate metrics for different k values
k_values = [5, 10, 20]
hit_ratios = []
ndcg_values = []

for k in k_values:
    print(f"\nCalculating metrics for k={k}...")
    hr = calculate_hit_ratio(model, test_df, k=k)
    ndcg = calculate_ndcg(model, test_df, k=k)
    
    hit_ratios.append(hr)
    ndcg_values.append(ndcg)
    
    print(f"Hit Ratio@{k}: {hr:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")

# Plot results with improved styling
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(k_values, hit_ratios, marker='o', linewidth=2, markersize=8)
plt.title('Hit Ratio@k', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('Hit Ratio', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
for i, hr in enumerate(hit_ratios):
    plt.annotate(f'{hr:.4f}', (k_values[i], hr), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

plt.subplot(1, 2, 2)
plt.plot(k_values, ndcg_values, marker='o', color='orange', linewidth=2, markersize=8)
plt.title('NDCG@k', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('NDCG', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
for i, ndcg in enumerate(ndcg_values):
    plt.annotate(f'{ndcg:.4f}', (k_values[i], ndcg), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=10)

plt.suptitle('SUM Model Evaluation Results', fontsize=16)
plt.tight_layout()
plt.savefig('sum_model_evaluation_improved.png', dpi=300)
print("\nEvaluation plot saved as 'sum_model_evaluation_improved.png'")

# Analyze user interests with improved visualization
print("\nAnalyzing user interests...")

def analyze_user_interests(model, user_id, top_n=5):
    """Analyze the interests of a specific user with improved visualization."""
    if user_id not in model.user_id_map:
        print(f"User {user_id} not found in training data")
        return
    
    user_idx = model.user_id_map[user_id]
    
    # Get user's sequence
    if user_idx not in model.user_sequences or len(model.user_sequences[user_idx]) == 0:
        print(f"User {user_id} has no sequence")
        return
    
    sequence = model.user_sequences[user_idx]
    
    # Truncate sequence if needed
    if len(sequence) > model.sequence_length:
        sequence = sequence[-model.sequence_length:]
    
    # Prepare input
    seq_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(model.device)
    mask_tensor = torch.ones((1, len(sequence)), dtype=torch.float).to(model.device)
    
    # Get item names for the sequence
    sequence_items = [model.reverse_item_id_map[idx] for idx in sequence]
    print(f"User's sequence (last {len(sequence)} interactions):")
    for i, item in enumerate(sequence_items):
        print(f"  {i+1}. Item {item}")
    
    # Extract interests using dynamic routing
    with torch.no_grad():
        # Get item embeddings
        seq_embeddings = model.item_embeddings(seq_tensor)
        
        # Extract multiple interests using capsule network
        interests = model._dynamic_routing(seq_embeddings, mask_tensor)
        
        # Get interest scores
        interest_scores = model.interest_attention(interests.view(-1, model.interest_dim)).view(-1, model.num_interests)
        interest_weights = F.softmax(interest_scores, dim=1).squeeze(0).cpu().numpy()
        
        print(f"\nInterest weights: {', '.join([f'{w:.4f}' for w in interest_weights])}")
        
        # Create a visualization of interest weights
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, model.num_interests + 1), interest_weights, color='skyblue')
        plt.xlabel('Interest Number')
        plt.ylabel('Weight')
        plt.title(f'Interest Distribution for User {user_id}')
        plt.xticks(range(1, model.num_interests + 1))
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f'user_{user_id}_interests.png')
        print(f"Interest distribution saved as 'user_{user_id}_interests.png'")
        
        # For each interest, find the most similar items
        print("\nTop items for each interest:")
        for i in range(model.num_interests):
            interest_vector = interests[0, i].unsqueeze(0)
            projected_vector = model.output_projection(interest_vector)
            
            # Compute similarity with all items
            all_items = model.item_embeddings.weight[:-1]  # Exclude padding
            similarities = torch.matmul(projected_vector, all_items.t()).squeeze(0)
            
            # Get top-n items for this interest
            top_indices = torch.argsort(similarities, descending=True)[:top_n].cpu().numpy()
            top_items = [model.reverse_item_id_map[idx] for idx in top_indices]
            
            print(f"Interest {i+1} (weight: {interest_weights[i]:.4f}):")
            for j, item in enumerate(top_items):
                print(f"  {j+1}. Item {item}")

# Analyze interests for a sample user with more interactions
user_counts = train_df.groupby('user_id').size()
active_users = user_counts[user_counts > 10].index.tolist()
if active_users:
    sample_user = random.choice(active_users)
    print(f"Analyzing interests for active user {sample_user}...")
    analyze_user_interests(model, sample_user)
else:
    sample_user = random.choice(list(model.user_id_map.keys()))
    print(f"Analyzing interests for user {sample_user}...")
    analyze_user_interests(model, sample_user)

# Generate recommendations for the same user
print(f"\nGenerating recommendations for user {sample_user}...")
recommendations = model.recommend(sample_user, top_n=10)
print("Top 10 recommendations:")
for i, item in enumerate(recommendations):
    print(f"  {i+1}. Item {item}")

print("\nDone!")