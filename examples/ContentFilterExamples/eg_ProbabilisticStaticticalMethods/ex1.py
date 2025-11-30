import logging
from pathlib import Path
from typing import List
import pandas as pd

import corerec as cr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_movies(file_path: str) -> pd.DataFrame:
    """
    Load and parse the movies.dat file.

    Parameters:
    - file_path (str): Path to the movies.dat file.

    Returns:
    - pd.DataFrame: DataFrame containing movie information.
    """
    logger.info(f"Loading movies from {file_path}")
    movies = []
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue  # Skip malformed lines
            movie_id, title, genres = parts
            movies.append(
                {"movie_id": int(movie_id), "title": title, "genres": genres.replace("|", " ")}
            )
    df = pd.DataFrame(movies)
    logger.info(f"Loaded {len(df)} movies.")
    return df


def main():
    # Path to the movies.dat file
    # Try different possible paths
    import os
    possible_paths = [
        "src/SANDBOX/dataset/ml-1m/movies.dat",
        "sample_data/movies.dat",
        "../sample_data/movies.dat",
        "../../sample_data/movies.dat",
    ]
    
    movies_file = None
    for path in possible_paths:
        if os.path.exists(path):
            movies_file = Path(path)
            break
    
    if movies_file is None:
        logger.error("movies.dat not found. Please provide a valid path.")
        logger.error("Tried paths: " + str(possible_paths))
        return

    # Load movies data
    movies_df = load_movies(movies_file)

    # Preprocess genres
    movies_df["processed_genres"] = movies_df["genres"]

    # Initialize and fit the LSA model
    lsa = cr.LSA(n_components=min(20, len(movies_df["processed_genres"].tolist())))
    lsa.fit(movies_df["processed_genres"].tolist())

    # Example: Recommend movies similar to a given movie
    movie_title = "Father of the Bride Part II (1995)"
    if movie_title not in movies_df["title"].values:
        logger.error(f"Movie '{movie_title}' not found in the dataset.")
        return

    # Get the index of the movie
    movie_index = movies_df[movies_df["title"] == movie_title].index[0]

    # Generate recommendations
    top_n = 5
    recommendations = lsa.recommend(
        query=movies_df.at[movie_index, "processed_genres"], top_n=top_n
    )

    # Display recommendations
    logger.info(f"Top {top_n} recommendations for '{movie_title}':")
    for idx in recommendations:
        recommended_movie = movies_df.iloc[idx]
        print(
            f"{recommended_movie['movie_id']}:: {recommended_movie['title']}:: {recommended_movie['genres']}"
        )


if __name__ == "__main__":
    main()
