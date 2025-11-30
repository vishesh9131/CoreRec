import pandas as pd
import os


def load_movielens_data(data_path: str = "src/SANDBOX/dataset/ml-1m"):
    """Load MovieLens ML-1M dataset with calculated average ratings"""
    # Read movies data
    movies_df = pd.read_csv(
        os.path.join(data_path, "movies.dat"),
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["movie_id", "title", "genres"],
        dtype={"movie_id": int, "title": str, "genres": str},
    )

    # Read ratings data
    ratings_df = pd.read_csv(
        os.path.join(data_path, "ratings.dat"),
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )

    # Calculate average rating and count for each movie
    rating_stats = ratings_df.groupby("movie_id").agg({"rating": ["mean", "count"]}).reset_index()

    # Flatten column names
    rating_stats.columns = ["movie_id", "avg_rating", "rating_count"]

    # Merge with movies_df
    movies_df = movies_df.merge(rating_stats, on="movie_id", how="left")

    # Fill NaN values for movies with no ratings
    movies_df["avg_rating"] = movies_df["avg_rating"].fillna(0.0)
    movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)

    # Add description
    movies_df["description"] = movies_df.apply(
        lambda x: (
            f"A {x['genres'].replace('|', ', ')} movie. "
            f"Average rating: {x['avg_rating']:.1f} from {x['rating_count']} ratings."
        ),
        axis=1,
    )

    return movies_df, ratings_df
