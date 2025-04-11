import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_matrix
import sys
# sys.path.append('../../../')
import corerec.uf_engine as uf

np.random.seed(42)
random.seed(42)

def generate_synthetic_dataset(n_users=200, n_items=100, n_interactions=5000):
    """Generate a synthetic dataset with realistic user preference patterns."""
    
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
        
    # Generate interactions with ratings
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
            
            # Generate rating (users rate items in preferred categories higher)
            if any(cat in preferred_cats for cat in item_categories[item_id]):
                # Higher rating for preferred categories (3.5-5)
                rating = random.uniform(3.5, 5.0)
            else:
                # Lower rating for non-preferred categories (1-3.5)
                rating = random.uniform(1.0, 3.5)
            
            interactions.append((user_id, item_id, rating))
    
    # Convert to DataFrame
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])
    
    # Sort by user and rating
    df = df.sort_values(['user_id', 'rating'], ascending=[True, False])
    
    # Remove duplicates (keep the highest rating)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    
    # Limit to n_interactions
    if len(df) > n_interactions:
        df = df.sample(n_interactions, random_state=42)
        df = df.sort_values(['user_id', 'rating'], ascending=[True, False])
    
    return df, item_categories, user_preferences

def train_recommender(interactions_df):
    """Train a Matrix Factorization recommender on the given interactions."""
    
    # Extract user_ids, item_ids, and ratings
    user_ids = interactions_df['user_id'].values
    item_ids = interactions_df['item_id'].values
    ratings = interactions_df['rating'].values
    
    # Create a sparse matrix from the interactions
    n_users = interactions_df['user_id'].max()
    n_items = interactions_df['item_id'].max()
    
    # Create the interaction matrix
    interaction_matrix = csr_matrix(
        (ratings, (user_ids - 1, item_ids - 1)),  # Subtract 1 for zero-indexing
        shape=(n_users, n_items)
    )
    
    # Initialize the recommender
    recommender = uf.MF_MATRIX_FACTORIZATION_RECOMMENDER(
        num_factors=50,
        learning_rate=0.01,
        reg_user=0.02,
        reg_item=0.02,
        epochs=20,
        early_stopping_rounds=5,
        n_threads=4
    )
    
    # Train the model
    recommender.fit(interaction_matrix)
    
    return recommender, n_users, n_items

def evaluate_recommender(recommender, test_df, n_users, n_items, item_categories, user_preferences):
    """Evaluate the recommender on test data and print actual vs predicted items."""
    
    # Calculate hit rate and category alignment
    hit_count = 0
    category_match_count = 0
    total_recommendations = 0
    
    # Sample some users for evaluation
    eval_users = random.sample(range(1, n_users + 1), min(5, n_users))
    
    for user_id in eval_users:
        # Get actual items the user interacted with in test set
        actual_items = test_df[test_df['user_id'] == user_id]['item_id'].tolist()
        
        if not actual_items:
            continue
            
        # Get user's preferred categories
        user_cats = user_preferences[user_id]
        
        # Get recommendations
        recs = recommender.recommend(user_id - 1, top_n=5)
        
        # Convert back to 1-indexed
        recs = [r + 1 for r in recs]
        
        # Print actual vs predicted
        print(f"\nUser {user_id} (Preferred categories: {[f'Cat {c}' for c in user_cats]}):")
        print(f"  Actual items: {actual_items[:5]}")
        print(f"  Predicted items: {recs}")
        
        # Print categories for actual and predicted items
        print("  Actual item categories:")
        for item in actual_items[:5]:
            if item in item_categories:
                print(f"    Item {item}: {[f'Cat {c}' for c in item_categories[item]]}")
        
        print("  Predicted item categories:")
        for item in recs:
            if item in item_categories:
                print(f"    Item {item}: {[f'Cat {c}' for c in item_categories[item]]}")
        
        # Check hits
        hits = set(recs) & set(actual_items)
        hit_count += len(hits)
        
        # Check category alignment
        for rec_item in recs:
            if rec_item in item_categories:
                item_cats = item_categories[rec_item]
                if any(cat in user_cats for cat in item_cats):
                    category_match_count += 1
        
        total_recommendations += len(recs)
    
    hit_rate = hit_count / total_recommendations if total_recommendations > 0 else 0
    category_alignment = category_match_count / total_recommendations if total_recommendations > 0 else 0
    
    return {
        'hit_rate': hit_rate,
        'category_alignment': category_alignment
    }

def get_recommendations_for_user(recommender, user_id, n_items, item_categories):
    """Get and display recommendations for a specific user."""
    
    # Get recommendations
    recs = recommender.recommend(user_id - 1, top_n=10)
    
    # Convert back to 1-indexed
    recs = [r + 1 for r in recs]
    
    # Create a DataFrame with recommendations and their categories
    rec_data = []
    for item_id in recs:
        if item_id in item_categories:
            cats = item_categories[item_id]
            rec_data.append({
                'item_id': item_id,
                'categories': [f"Category {cat}" for cat in cats]
            })
    
    return pd.DataFrame(rec_data)

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

def main():
    print("Generating synthetic dataset...")
    interactions_df, item_categories, user_preferences = generate_synthetic_dataset(n_users=100, n_items=50, n_interactions=3000)
    
    print(f"Generated {len(interactions_df)} interactions between {interactions_df['user_id'].nunique()} users and {interactions_df['item_id'].nunique()} items")
    
    # Split into train and test
    train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
    
    print("Training recommender model...")
    recommender, n_users, n_items = train_recommender(train_df)
    
    print("\nEvaluating recommender (Actual vs Predicted)...")
    metrics = evaluate_recommender(recommender, test_df, n_users, n_items, item_categories, user_preferences)
    print(f"\nOverall Hit Rate: {metrics['hit_rate']:.4f}")
    print(f"Overall Category Alignment: {metrics['category_alignment']:.4f}")

if __name__ == "__main__":
    main()
