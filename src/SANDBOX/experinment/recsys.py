import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from corerec.engines.collaborative.mf_base.matrix_factorization import MatrixFactorization
from corerec.engines.content_based.tfidf_recommender import TFIDFRecommender
from corerec.engines.hybrid import HybridEngine
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
interactions_df = pd.read_csv('interactions.csv')
items_df = pd.read_csv('items.csv')

# Split data into training and testing sets
train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)

# Extract data for collaborative filtering
user_ids = train_df['user_id'].values
item_ids = train_df['item_id'].values
ratings = train_df['rating'].values

# Initialize Matrix Factorization model
mf_model = MatrixFactorization(
    k=20,  # Number of latent factors
    learning_rate=0.01,
    regularization=0.02,
    iterations=20,
    use_bias=True
)

# Fit the model
mf_model.fit(user_ids, item_ids, ratings)

# Prepare content-based features
# Convert categories to a format suitable for TF-IDF
items_df['feature_text'] = items_df['categories'] + " " + \
                           items_df['popularity'].astype(str) + " " + \
                           items_df['price'].astype(str)

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(items_df['feature_text'])

# Initialize content-based recommender
content_recommender = TFIDFRecommender(feature_matrix=tfidf_matrix)

# Create a hybrid engine
hybrid_engine = HybridEngine(
    collaborative_engine=mf_model,
    content_engine=content_recommender,
    alpha=0.7  # Weight for collaborative filtering (0.7) vs content-based (0.3)
)

# Function to get recommendations for a user
def get_recommendations(user_id, top_n=10):
    # Get collaborative filtering recommendations
    cf_recommendations = mf_model.recommend(user_id, top_n=top_n)
    
    # Get content-based recommendations
    # For content-based, we need to find items the user has interacted with
    user_items = interactions_df[interactions_df['user_id'] == user_id]['item_id'].values
    if len(user_items) > 0:
        # Use the first few items as seeds for content-based recommendations
        seed_items = user_items[:min(3, len(user_items))]
        # Convert to indices in the feature matrix
        seed_indices = [item_id - 1 for item_id in seed_items]  # Adjust if your item_ids don't start at 1
        content_recommendations = content_recommender.recommend(seed_indices, top_n=top_n)
    else:
        # If no interactions, use popular items
        content_recommendations = list(range(top_n))
    
    # Combine recommendations with weighting
    combined_scores = {}
    
    # Add collaborative filtering scores
    for i, item_id in enumerate(cf_recommendations):
        combined_scores[item_id] = 0.7 * (top_n - i) / top_n
    
    # Add content-based scores
    for i, item_id in enumerate(content_recommendations):
        item_id = item_id + 1  # Adjust if your item_ids start at 1
        if item_id in combined_scores:
            combined_scores[item_id] += 0.3 * (top_n - i) / top_n
        else:
            combined_scores[item_id] = 0.3 * (top_n - i) / top_n
    
    # Sort by score and get top items
    recommended_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item_id for item_id, score in recommended_items]

# Test the recommendation system
test_user_id = 10  # Example user
recommendations = get_recommendations(test_user_id, top_n=10)

print(f"\nTop 10 recommendations for user {test_user_id}:")
for i, item_id in enumerate(recommendations, 1):
    item_info = items_df[items_df['item_id'] == item_id].iloc[0]
    print(f"{i}. Item {item_id} - Categories: {item_info['categories']}, Popularity: {item_info['popularity']:.2f}, Price: ${item_info['price']:.2f}")

# Evaluate the recommendation system
def evaluate_recommendations(test_data, all_users, top_n=10):
    """Evaluate recommendations using hit rate and average precision."""
    hit_count = 0
    total_precision = 0
    
    for user_id in all_users:
        # Get actual items the user interacted with in the test set
        actual_items = set(test_data[test_data['user_id'] == user_id]['item_id'].values)
        
        if not actual_items:
            continue
        
        # Get recommended items
        recommended_items = get_recommendations(user_id, top_n=top_n)
        
        # Calculate hits
        hits = len(set(recommended_items) & actual_items)
        hit_count += 1 if hits > 0 else 0
        
        # Calculate precision
        precision = hits / min(len(recommended_items), len(actual_items))
        total_precision += precision
    
    # Calculate metrics
    hit_rate = hit_count / len(all_users)
    avg_precision = total_precision / len(all_users)
    
    return hit_rate, avg_precision

# Get a sample of users for evaluation
eval_users = random.sample(list(interactions_df['user_id'].unique()), 50)
hit_rate, avg_precision = evaluate_recommendations(test_df, eval_users)

print(f"\nEvaluation results:")
print(f"Hit Rate: {hit_rate:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Visualize user-item interactions
plt.figure(figsize=(10, 6))
plt.scatter(interactions_df['user_id'], interactions_df['item_id'], 
            c=interactions_df['rating'], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Rating')
plt.xlabel('User ID')
plt.ylabel('Item ID')
plt.title('User-Item Interaction Matrix')
plt.tight_layout()
plt.savefig('user_item_interactions.png')
plt.close()

# Visualize rating distribution
plt.figure(figsize=(8, 5))
interactions_df['rating'].hist(bins=10)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.close()

print("\nVisualization files saved: user_item_interactions.png and rating_distribution.png")