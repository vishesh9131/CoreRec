import logging
from typing import List, Dict, Any
from corerec.engines.contentFilterEngine.probabilistic_statistical_methods import PRO_LSA
from corerec.engines.contentFilterEngine.special_techniques import SPE_INTERACTIVE_FILTERING, SPE_DYNAMIC_FILTERING
from corerec.engines.contentFilterEngine.context_personalization import CON_CONTEXT_AWARE, CON_USER_PROFILING, CON_ITEM_PROFILING
import pandas as pd
import chardet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: str) -> str:
    """
    Detect the encoding of a file.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - str: Detected encoding.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        logger.info(f"Detected encoding: {encoding} with confidence {confidence}")
        return encoding if encoding else 'utf-8'

def parse_movies(file_path: str, encoding: str = 'latin1') -> pd.DataFrame:
    """
    Parse the movies.dat file into a DataFrame.

    Parameters:
    - file_path (str): Path to the movies.dat file.
    - encoding (str): Encoding of the movies.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing movie information.
    """
    movies = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                movie_id, title, genres = parts
                movies.append({
                    'movie_id': int(movie_id),
                    'title': title,
                    'genres': genres.split('|')
                })
        logger.info(f"Parsed {len(movies)} movies from {file_path}.")
    except UnicodeDecodeError as e:
        logger.error(f"UnicodeDecodeError: {e}. Try using a different encoding.")
    except Exception as e:
        logger.error(f"An error occurred while parsing movies: {e}")
    return pd.DataFrame(movies)

def main():
    # Path to the movies.dat file
    movies_file = "src/SANDBOX/dataset/ml-1m/movies.dat"

    # Detect file encoding
    encoding = detect_encoding(movies_file)

    # Parse the movies dataset
    movies_df = parse_movies(movies_file, encoding=encoding)
    logger.info(f"Loaded {len(movies_df)} movies.")

    # Example user interactions (user_id: list of interacted movie_ids)
    user_interactions = {
        1: [1, 2, 3, 4, 5],
        2: [6, 7, 8, 9, 10],
        # Add more user interactions as needed
    }

    # Example item features (movie_id: feature_dict)
    item_features = {}
    for _, row in movies_df.iterrows():
        item_features[row['movie_id']] = {
            'genres': row['genres']
            # Add more features if necessary
        }

    # Initialize item profiling recommender
    item_profiling = CON_ITEM_PROFILING()
    item_profiling.fit(user_interactions, item_features)

    # Initialize user profiling recommender
    user_profiling = CON_USER_PROFILING()
    user_profiling.fit(user_interactions)

    # Initialize base LSA recommender
    lsa = PRO_LSA(n_components=20)
    genre_texts = movies_df['genres'].apply(lambda genres: ' '.join(genres)).tolist()
    lsa.fit(genre_texts)

    # Wrap the LSA recommender with DynamicFilteringRecommender
    dynamic_filter = SPE_DYNAMIC_FILTERING(base_recommender=lsa)

    # Wrap the DynamicFilteringRecommender with InteractiveFilteringRecommender
    interactive_filter = SPE_INTERACTIVE_FILTERING(base_recommender=dynamic_filter)

    # Generate initial recommendations for a user
    user_id = 1
    query = "Toy Story (1995)"
    top_n = 5

    logger.info(f"Generating initial Top {top_n} recommendations for '{query}':")
    initial_recommendations = interactive_filter.recommend(user_id, query, top_n)
    for idx in initial_recommendations:
        movie = movies_df[movies_df['movie_id'] == idx].iloc[0]
        print(f"{movie['movie_id']}:: {movie['title']}::{ '|'.join(movie['genres']) }")

    # Simulate user feedback
    # Let's assume user likes movie_id 3 and dislikes movie_id 4
    interactive_filter.collect_feedback(user_id, item_id=3, feedback_score=1.0)  # Positive feedback
    interactive_filter.collect_feedback(user_id, item_id=4, feedback_score=-1.0)  # Negative feedback

    # Generate refined recommendations based on feedback
    logger.info(f"\nGenerating refined Top {top_n} recommendations for '{query}' after feedback:")
    refined_recommendations = interactive_filter.recommend(user_id, query, top_n)
    for idx in refined_recommendations:
        movie = movies_df[movies_df['movie_id'] == idx].iloc[0]
        print(f"{movie['movie_id']}:: {movie['title']}::{ '|'.join(movie['genres']) }")

    # Handle dynamic data changes
    # Example: Adding a new movie
    new_movie_event = {
        'action': 'add',
        'item_id': 4000,
        'item_features': {'genres': ['Animation', 'Adventure', 'Comedy']}
    }
    dynamic_filter.handle_data_change(new_movie_event)

    # Generate recommendations after adding a new movie
    logger.info(f"\nGenerating updated Top {top_n} recommendations for '{query}' after adding a new movie:")
    updated_recommendations = interactive_filter.recommend(user_id, query, top_n)
    for idx in updated_recommendations:
        # Check if the movie exists in the DataFrame
        if idx in movies_df['movie_id'].values:
            movie = movies_df[movies_df['movie_id'] == idx].iloc[0]
            print(f"{movie['movie_id']}:: {movie['title']}::{ '|'.join(movie['genres']) }")
        else:
            # If it's a new item, add it to the DataFrame
            new_movie = {
                'movie_id': idx,
                'title': 'New Animated Movie (2001)',  # Replace with actual title if available
                'genres': new_movie_event['item_features']['genres']
            }
            movies_df = movies_df.append(new_movie, ignore_index=True)
            print(f"{idx}:: {new_movie['title']}::{ '|'.join(new_movie['genres']) }")

if __name__ == "__main__":
    main()