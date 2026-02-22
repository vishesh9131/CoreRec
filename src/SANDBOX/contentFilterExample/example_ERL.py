import pandas as pd
from corerec.engines.content_based.context_personalization import (
    CON_CONTEXT_AWARE,
    CON_USER_PROFILING,
    CON_ITEM_PROFILING
)
from corerec.engines.content_based.embedding_representation_learning import (
    EMB_WORD2VEC,
    EMB_DOC2VEC,
    EMB_PERSONALIZED_EMBEDDINGS
)
from typing import Dict, List, Any
import os
import json

def load_users(file_path: str) -> pd.DataFrame:
    """
    Load and parse the users.dat file.

    Parameters:
    - file_path (str): Path to the users.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing user information.
    """
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
    """
    Load and parse the ratings.dat file.

    Parameters:
    - file_path (str): Path to the ratings.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing ratings information.
    """
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

def build_user_interactions(ratings: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Build a dictionary mapping user IDs to lists of interacted movie IDs.

    Parameters:
    - ratings (pd.DataFrame): DataFrame containing ratings information.

    Returns:
    - Dict[int, List[int]]: Dictionary of user interactions.
    """
    user_interactions = ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
    return user_interactions

def build_item_features(movies: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Build a dictionary mapping movie IDs to their genres.

    Parameters:
    - movies (pd.DataFrame): DataFrame containing movie information.

    Returns:
    - Dict[int, Dict[str, Any]]: Dictionary of item features.
    """
    item_features = {}
    for _, row in movies.iterrows():
        movie_id = row['movie_id']
        genres = row['genres'].split('|')
        item_features[movie_id] = {genre: 1 for genre in genres}
    return item_features

def prepare_embedding_data(movies: pd.DataFrame) -> List[List[str]]:
    """
    Prepare data for embedding training by tokenizing movie titles or genres.

    Parameters:
    - movies (pd.DataFrame): DataFrame containing movie information.

    Returns:
    - List[List[str]]: Tokenized sentences/documents for embedding training.
    """
    # Example: Using genres as sentences for Word2Vec
    sentences = [genres.split('|') for genres in movies['genres']]
    return sentences

def main():
    # Configuration
    data_path = 'src/SANDBOX/dataset/ml-1m' 
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
    
    # Initialize Embedding Models
    print("Initializing embedding models...")
    embedding_recommender = EMB_PERSONALIZED_EMBEDDINGS()
    
    # Prepare data for embeddings
    print("Preparing data for embeddings...")
    embedding_sentences = prepare_embedding_data(movies_df)
    
    # Train Embedding Models
    print("Training Word2Vec model...")
    embedding_recommender.train_word2vec(sentences=embedding_sentences, epochs=10)
    
    print("Training Doc2Vec model...")
    embedding_recommender.train_doc2vec(documents=embedding_sentences)
    
    # Fit Recommenders
    print("Fitting User Profiling Recommender...")
    user_recommender.fit(user_interactions)
    
    print("Fitting Context Aware Recommender...")
    context_recommender.fit(user_interactions)
    
    print("Fitting Item Profiling Recommender...")
    item_recommender.fit(user_interactions, item_features)
    
    # Generate Embedding-Based Recommendations (Example)
    user_id = 1  # Replace with desired user ID
    current_context = {
        "time_of_day": "evening",
        "location": "home"
    }
    
    print(f"Generating recommendations for User {user_id} with context {current_context} using embeddings...")
    # Example: Get user profile attributes
    user_profile = user_recommender.user_profiles.get(user_id, {})
    if not user_profile:
        print(f"No profile found for User {user_id}.")
        recommendations = []
    else:
        # Example: Aggregate embeddings based on user interacted items
        user_embedding = {}
        interacted_items = user_profile.get('interacted_items', set())
        for item_id in interacted_items:
            genres = movies_df[movies_df['movie_id'] == item_id]['genres'].values[0].split('|')
            for genre in genres:
                genre_embedding = embedding_recommender.get_word_embedding(genre)
                for idx, val in enumerate(genre_embedding):
                    user_embedding[idx] = user_embedding.get(idx, 0.0) + val
        # Compute average embedding
        num_genres = len(interacted_items) * len(genres) if interacted_items else 1
        user_embedding = {k: v / num_genres for k, v in user_embedding.items()}
        
        # Score all items based on cosine similarity with user embedding
        from numpy import dot
        from numpy.linalg import norm
        
        scores = {}
        for item_id in all_items:
            if item_id in interacted_items:
                continue
            genres = movies_df[movies_df['movie_id'] == item_id]['genres'].values[0].split('|')
            item_embedding = []
            for genre in genres:
                genre_emb = embedding_recommender.get_word_embedding(genre)
                item_embedding.extend(genre_emb)
            # Simple scoring: dot product of user and item embeddings
            # Adjust based on actual embedding dimensions and alignment
            if not item_embedding:
                continue
            # Ensure user_embedding and item_embedding have the same length
            # Here, assume user_embedding is a dictionary indexed by the same dimension as embedding vectors
            item_vector = sum(item_embedding)
            user_vector = sum(user_embedding.values())
            similarity = dot([user_vector], [item_vector]) / (norm([user_vector]) * norm([item_vector]) + 1e-10)
            scores[item_id] = similarity
        
        # Sort and get top-N
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, score in ranked_items[:10]]
    
    # Fetch and display movie titles for recommended movie IDs
    if recommendations:
        recommended_movies = movies_df[movies_df['movie_id'].isin(recommendations)]
        print(f"Top 10 embedding-based recommendations for User {user_id} in context {current_context}:")
        for _, row in recommended_movies.iterrows():
            print(f"- {row['title']}")
    else:
        print("No recommendations could be generated.")

if __name__ == "__main__":
        main()