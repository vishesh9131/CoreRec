import pandas as pd
from corerec.engines.content_based.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_USER_PROFILING,
    CON_ITEM_PROFILING
)
from corerec.engines.content_based.fairness_explainability import (
    FAI_EXPLAINABLE,
    FAI_FAIRNESS_AWARE,
    FAI_PRIVACY_PRESERVING
)
from typing import Dict, List, Any

def load_movies(file_path: str) -> pd.DataFrame:
    """
    Load and parse the movies.dat file.

    Parameters:
    - file_path (str): Path to the movies.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing movie information.
    """
    column_names = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        file_path,
        sep='::',
        engine='python',
        names=column_names,
        encoding='latin-1'
    )
    return movies

def build_item_features(movies_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Build a dictionary mapping movie IDs to their features.

    Parameters:
    - movies_df (pd.DataFrame): DataFrame containing movie information.

    Returns:
    - Dict[int, Dict[str, Any]]: Mapping of movie IDs to feature dictionaries.
    """
    item_features = {}
    for _, row in movies_df.iterrows():
        movie_id = row['movie_id']
        genres = row['genres'].split('|')
        item_features[movie_id] = {genre: 1 for genre in genres}
    return item_features

def main():
    # Load Movies Data
    movies_file = 'src/SANDBOX/dataset/ml-1m/movies.dat'
    print("Loading movies data...")
    movies_df = load_movies(movies_file)
    
    # Build Item Features
    print("Building item features...")
    item_features = build_item_features(movies_df)
    
    # Initialize Recommenders
    print("Initializing recommenders...")
    context_recommender = CON_CONTEXT_AWARE(
        context_config_path='src/SANDBOX/dataset/ml-1m/context_config.json',
        item_features=item_features
    )
    item_recommender = CON_ITEM_PROFILING()
    
    # Initialize Fairness and Explainability Modules
    explainable = FAI_EXPLAINABLE()
    fairness_aware = FAI_FAIRNESS_AWARE()
    
    # Example User Interactions (Placeholder)
    user_interactions = {
        1: [1, 2, 3],  
        2: [4, 5, 6]   
    }
    
    # Fit Recommenders
    print("Fitting Context Aware Recommender...")
    context_recommender.fit(user_interactions)
    
    print("Fitting Item Profiling Recommender...")
    item_recommender.fit(user_interactions, item_features)
    
    # Generate Recommendations for a User
    user_id = 1
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
    
    # Ensure Fairness
    print("Ensuring fairness in recommendations...")
    recommendations = fairness_aware.ensure_fairness({user_id: recommendations}, pd.DataFrame(user_interactions))
    
    # Generate Explanations
    print("Generating explanations for recommendations...")
    for item_id in recommendations[user_id]:
        explanation = explainable.generate_explanation(user_id, item_id, current_context)
        print(f"Recommendation: {movies_df.loc[movies_df['movie_id'] == item_id, 'title'].values[0]}")
        print(f"Explanation: {explanation}")

if __name__ == "__main__":
    main()