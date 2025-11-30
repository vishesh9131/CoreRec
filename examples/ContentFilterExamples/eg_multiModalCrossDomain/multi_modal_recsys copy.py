import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
# Import using cleaner module path
import corerec as cr


class MultiModalRecommender:
    def __init__(self):
        self.title_vectorizer = TfidfVectorizer(max_features=500)
        self.genre_binarizer = MultiLabelBinarizer()
        self.multi_modal_model = cr.MultiModal(self.title_vectorizer, self.genre_binarizer)

    def load_data(self, filepath):
        """
        Load the movies dataset.

        Args:
            filepath (str): Path to the movies.dat file.

        Returns:
            pandas.DataFrame: Loaded movies dataframe.
        """
        self.movies_df = pd.read_csv(
            filepath,
            sep="::",
            engine="python",
            header=None,
            names=["movie_id", "title", "genres"],
            encoding="ISO-8859-1",  # Changed encoding to handle special characters
        )
        self.movies_df["genres"] = self.movies_df["genres"].apply(lambda x: x.split("|"))
        return self.movies_df

    def preprocess_data(self):
        """
        Fit and transform the title and genre data.

        Returns:
            tuple: Tuple containing title features and genre features.
        """
        # Fit the vectorizers on the entire dataset
        title_features = self.title_vectorizer.fit_transform(self.movies_df["title"]).toarray()
        genre_features = self.genre_binarizer.fit_transform(self.movies_df["genres"])
        return title_features, genre_features

    def get_recommendations(self, movie_id, top_k=5):
        """
        Generate movie recommendations based on cosine similarity.

        Args:
            movie_id (int): ID of the movie to base recommendations on.
            top_k (int): Number of recommendations to generate.

        Returns:
            pandas.DataFrame: DataFrame containing top recommended movies.
        """
        # Find the index of the movie in the dataframe
        movie_idx = self.movies_df.index[self.movies_df["movie_id"] == movie_id].tolist()[0]

        # Preprocess data and get all features
        title_features, genre_features = self.preprocess_data()

        # Get all titles and genres as lists
        all_titles = self.movies_df["title"].tolist()
        all_genres = self.movies_df["genres"].tolist()

        # Get the specific movie's title and genres
        target_title = [self.movies_df.at[movie_idx, "title"]]
        target_genres = [self.movies_df.at[movie_idx, "genres"]]

        # Get combined features for all movies
        all_features = self.multi_modal_model.forward(all_titles, all_genres)

        # Get combined features for the target movie
        target_features = self.multi_modal_model.forward(target_title, target_genres)

        # Calculate cosine similarity between the target movie and all movies
        similarities = cosine_similarity(target_features, all_features).flatten()

        # Get indices of top_k similar movies excluding the target movie itself
        similar_indices = similarities.argsort()[-top_k - 1 : -1][::-1]

        return self.movies_df.iloc[similar_indices]


def main():
    # Initialize recommender
    recommender = MultiModalRecommender()

    # Load and preprocess data
    print("Loading data...")
    # Try different possible paths
    import os
    possible_paths = [
        "src/SANDBOX/dataset/ml-1m/movies.dat",
        "sample_data/movies.dat",
        "../sample_data/movies.dat",
        "../../sample_data/movies.dat",
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        print("Warning: movies.dat not found. Please provide a valid path.")
        print("Tried paths:", possible_paths)
        return
    
    movies_df = recommender.load_data(file_path)
    print(f"Loaded {len(movies_df)} movies")

    # Get recommendations for a specific movie
    movie_id = 78  # Example: Toy Story (1995)
    print(f"\nGenerating recommendations for movie ID {movie_id}...")
    recommendations = recommender.get_recommendations(movie_id)
    print("\nTop recommendations:")
    print(recommendations[["title", "genres"]])


if __name__ == "__main__":
    main()
