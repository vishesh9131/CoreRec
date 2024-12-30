import pandas as pd
import json
from typing import Dict, List, Any
import os

path = 'CRLearn/CRDS/ml-1m'
default_file_path = 'CRLearn/CRDS/ml-1m'

movies_file = os.path.join(path, 'movies.dat')

def load_movies(movies_file: str) -> pd.DataFrame:
    return pd.read_csv(movies_file, sep='::', engine='python', names=['movie_id', 'title', 'genres'], encoding='latin-1')

movies_df = load_movies(movies_file)

def load(data_path: str = path) -> Dict[str, pd.DataFrame]:
    """Load and process users, ratings, and movies data from the specified path."""
    # Ensure the data_path is relative to the current working directory
    data_path = os.path.abspath(data_path)  # Convert to absolute path

    column_names_users = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(
        os.path.join(data_path, 'users.dat'),
        sep='::',
        engine='python',
        names=column_names_users,
        encoding='latin-1'
    )
    
    column_names_ratings = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        engine='python',
        names=column_names_ratings,
        encoding='latin-1'
    )
    
    column_names_movies = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        os.path.join(data_path, 'movies.dat'),
        sep='::',
        engine='python',
        names=column_names_movies,
        encoding='latin-1'
    )
    
    # Build User Interactions
    user_interactions = ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
    
    # Build Item Features
    item_features = {}
    for _, row in movies.iterrows():
        movie_id = row['movie_id']
        genres = row['genres'].split('|')
        item_features[movie_id] = {genre: 1 for genre in genres}
    
    return {
        'users': users,
        'ratings': ratings,
        'movies': movies,
        'user_interactions': user_interactions,
        'item_features': item_features
    }


def load_config(config_path: str = os.path.join(default_file_path, 'context_config.json')) -> Dict[str, Any]:

    """Load configuration from the specified path."""
    # Ensure config_path is a valid string
    if not isinstance(config_path, str):
        raise ValueError("config_path must be a valid string.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file at {config_path} does not exist.")
    
    with open(config_path, 'r') as file:
        return json.load(file)
    

def prepare_embedding_data(movies_file: str=movies_file) -> List[List[str]]:
    """
    Prepare data for embedding training by tokenizing movie titles or genres.

    Parameters:
    - movies_file (str): Path to the movies file.

    Returns:
    - List[List[str]]: Tokenized sentences/documents for embedding training.
    """
    # Load movies data from the file
    movies = pd.read_csv(movies_file, sep='::', engine='python', names=['movie_id', 'title', 'genres'], encoding='latin-1')
    
    # Example: Using genres as sentences for Word2Vec
    sentences = [genres.split('|') for genres in movies['genres']]
    return sentences

