import pandas as pd
from corerec.engines.content_based.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_USER_PROFILING,
    CON_ITEM_PROFILING
)
from typing import Dict, List, Any
import os

def load_users(file_path: str) -> pd.DataFrame:
    column_names = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=column_names,
        encoding='latin-1'
    )
    return users

def load_ratings(file_path: str) -> pd.DataFrame:
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=column_names,
        encoding='latin-1'
    )
    return ratings

def load_movies(file_path: str) -> pd.DataFrame:
    column_names = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=column_names,
        encoding='latin-1'
    )
    return movies

def build_user_interactions(ratings: pd.DataFrame) -> Dict[int, List[int]]:
    user_interactions = ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
    return user_interactions

def build_item_features(movies: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    item_features = {}
    for _, row in movies.iterrows():
        movie_id = row['movie_id']
        genres = row['genres'].split('|')
        item_features[movie_id] = {genre: 1 for genre in genres}
    return item_features

def main():
    # Configuration
    data_path = 'src/SANDBOX/dataset/ml-1m'  # Update with your actual data path
    context_config_path = os.path.join(data_path, 'context_config.json')
    users_file = os.path.join(data_path, 'users.dat')
    ratings_file = os.path.join(data_path, 'ratings.dat')
    movies_file = os.path.join(data_path, 'movies.dat')
    
    # Check if context_config.json exists
    if not os.path.exists(context_config_path):
        raise FileNotFoundError(f"Context configuration file not found at: {context_config_path}")
    
    # Load Data
    print("Loading users data...")
    users_df = load_users(users_file)
    
    print("Loading ratings data...")
    ratings_df = load_ratings(ratings_file)
    
    print("Loading movies data...")
    movies_df = load_movies(movies_file)
    
    # Build User Interactions and Item Features
    print("Building user interactions...")
    user_interactions = build_user_interactions(ratings_df)
    
    print("Building item features...")
    item_features = build_item_features(movies_df)
    
    all_items = set(movies_df['movie_id'].tolist())
    
    # Initialize Recommenders
    print("Initializing recommenders...")
    user_recommender = CON_USER_PROFILING(user_attributes=users_df)
    context_recommender = CON_CONTEXT_AWARE(
        context_config_path=context_config_path,
        item_features=item_features
    )
    item_recommender = CON_ITEM_PROFILING()
    
    # Fit Recommenders
    print("Fitting User Profiling Recommender...")
    user_recommender.fit(user_interactions)
    
    print("Fitting Context Aware Recommender...")
    context_recommender.fit(user_interactions)
    
    print("Fitting Item Profiling Recommender...")
    item_recommender.fit(user_interactions, item_features)
    
    # Example Recommendation
    user_id = 1  # Replace with desired user ID
    current_context = {
        "time_of_day": "evening",
        "location": "home"
    }
    
    print(f"Generating recommendations for User {user_id} with context {current_context}...")
    recommendations = context_recommender.recommend(
        user_id=user_id,
        context=current_context,
        top_n=10
    )
    
    # Fetch and display movie titles for recommended movie IDs
    recommended_movies = movies_df[movies_df['movie_id'].isin(recommendations)]
    print(f"Top 10 recommendations for User {user_id} in context {current_context}:")
    for _, row in recommended_movies.iterrows():
        print(f"- {row['title']}")

if __name__ == "__main__":
    main()
