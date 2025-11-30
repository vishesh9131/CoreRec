import pandas as pd
import corerec as cr
from typing import Dict, List, Any
import os


def load_users(file_path: str) -> pd.DataFrame:
    column_names = ["user_id", "gender", "age", "occupation"]
    users = pd.read_csv(file_path, sep="|", names=column_names, encoding="latin-1")
    return users


def load_ratings(file_path: str) -> pd.DataFrame:
    column_names = ["user_id", "movie_id", "rating", "timestamp"]
    ratings = pd.read_csv(file_path, sep="\t", names=column_names, encoding="latin-1")
    return ratings


def load_movies(file_path: str) -> pd.DataFrame:
    # Load movie data from the specified file
    movies = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        encoding="latin-1",
        names=[
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )

    # Extract genres into a single 'genres' column
    genre_columns = movies.columns[6:]  # Assuming genres start from the 7th column
    movies["genres"] = movies[genre_columns].apply(lambda x: "|".join(x.index[x == 1]), axis=1)

    return movies[["movie_id", "title", "genres"]]


def build_user_interactions(ratings: pd.DataFrame) -> Dict[int, List[int]]:
    user_interactions = ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
    return user_interactions


def build_item_features(movies: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    item_features = {}
    for _, row in movies.iterrows():
        movie_id = row["movie_id"]
        genres = row["genres"]

        # Ensure genres is a string before splitting
        if isinstance(genres, str) and genres.strip():
            genres = genres.split("|")
        else:
            genres = []  # Handle cases where genres is not a string or is empty

        # Only add features if genres are present
        if genres:
            item_features[movie_id] = {genre: 1 for genre in genres}
        else:
            item_features[movie_id] = {}  # Ensure the movie ID is present even if no genres

    return item_features


def main():
    # Configuration
    data_path = "src/SANDBOX/dataset/ml-100k"
    context_config_path = os.path.join(data_path, "context_config.json")
    users_file = os.path.join(data_path, "u.user")
    ratings_file = os.path.join(data_path, "u.data")
    movies_file = os.path.join(data_path, "u.item")

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

    print("Movies DataFrame:")
    print(movies_df.head())  # Display the first few rows of the movies DataFrame

    # Build User Interactions and Item Features
    print("Building user interactions...")
    user_interactions = build_user_interactions(ratings_df)
    print(f"User Interactions: {user_interactions}")  # Debugging output

    print("Building item features...")
    item_features = build_item_features(movies_df)
    print(f"Item Features: {item_features}")  # Debugging output

    all_items = set(movies_df["movie_id"].tolist())

    # Initialize Recommenders
    print("Initializing recommenders...")
    user_recommender = cr.UserProfiling(user_attributes=users_df)
    context_recommender = cr.ContextAware(
        context_config_path=context_config_path, item_features=item_features
    )
    item_recommender = cr.ItemProfiling()

    # Fit Recommenders
    print("Fitting User Profiling Recommender...")
    user_recommender.fit(user_interactions)

    print("Fitting Context Aware Recommender...")
    context_recommender.fit(user_interactions)

    print("Fitting Item Profiling Recommender...")
    item_recommender.fit(user_interactions, item_features)

    # Example Recommendation
    user_id = 1  # Replace with desired user ID
    current_context = {"time_of_day": "evening", "location": "home"}

    print(f"Generating recommendations for User {user_id} with context {current_context}...")
    recommendations = context_recommender.recommend(
        user_id=user_id, context=current_context, top_n=10
    )

    # Check if recommendations are empty
    if not recommendations:
        print(f"No recommendations found for User {user_id} in context {current_context}.")
    else:
        # Fetch and display movie titles for recommended movie IDs
        recommended_movies = movies_df[movies_df["movie_id"].isin(recommendations)]
        print(f"Top 10 recommendations for User {user_id} in context {current_context}:")
        for _, row in recommended_movies.iterrows():
            print(f"- {row['title']}")


if __name__ == "__main__":
    main()
