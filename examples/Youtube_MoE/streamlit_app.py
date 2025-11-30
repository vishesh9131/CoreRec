import streamlit as st
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Import your custom recommenders
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_recommender import (
    MatrixFactorizationRecommender,
)
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender
from corerec.engines.hybrid import HybridEngine
from data_loader import load_movielens_data

# Set page configuration first
st.set_page_config(page_title="Youtube MoE Implementation - Powered by CoreRec", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_feedback = {}
    st.session_state.feedback = pd.DataFrame(columns=["user_id", "movie_id", "rating"])


@st.cache_data
def load_movie_data():
    # Replace this with actual MovieLens data if needed
    movies_df, ratings_df = load_movielens_data()
    return movies_df, ratings_df


def load_saved_model(model_dir="examples/Youtube_MoE/models"):
    """Load the saved recommendation model and data"""
    try:
        # Check if all required files exist
        required_files = {
            "hybrid_engine.joblib": "Recommendation engine",
            "movies_df.pkl": "Movie database",
            "ratings_df.pkl": "User ratings",
        }

        missing_files = []
        for file, description in required_files.items():
            if not os.path.exists(os.path.join(model_dir, file)):
                missing_files.append(f"{description} ({file})")

        if missing_files:
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing_files)}\n"
                "Please run train_model.py first to generate these files."
            )

        # Load all components
        logging.info("Loading saved models and data...")
        hybrid_engine = joblib.load(os.path.join(model_dir, "hybrid_engine.joblib"))
        movies_df = pd.read_pickle(os.path.join(model_dir, "movies_df.pkl"))
        ratings_df = pd.read_pickle(os.path.join(model_dir, "ratings_df.pkl"))

        # Create movie ID mappings
        unique_movie_ids = movies_df["movie_id"].unique()
        movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}
        reverse_movie_id_map = {v: k for k, v in movie_id_map.items()}

        return hybrid_engine, movies_df, ratings_df, movie_id_map, reverse_movie_id_map

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Training new model...")
        from train_model import train_and_save_model

        return train_and_save_model()


def initialize_recommender():
    """Initialize the recommendation system"""
    if not os.path.exists("models"):
        os.makedirs("models")

    try:
        (
            hybrid_engine,
            movies_df,
            ratings_df,
            movie_id_map,
            reverse_movie_id_map,
        ) = load_saved_model()
        st.session_state.hybrid_engine = hybrid_engine
        st.session_state.movies_df = movies_df
        st.session_state.ratings_df = ratings_df
        st.session_state.movie_id_map = movie_id_map
        st.session_state.reverse_movie_id_map = reverse_movie_id_map
    except:
        st.stop()


def update_user_preferences(user_id: int, movie_id: int, liked: bool):
    """
    Update user preferences without retraining the entire model
    """
    # Update user feedback DataFrame
    rating_value = 5 if liked else 1
    new_feedback = pd.DataFrame(
        {"user_id": [user_id], "movie_id": [movie_id], "rating": [rating_value]}
    )
    st.session_state.feedback = pd.concat(
        [st.session_state.feedback, new_feedback], ignore_index=True
    )

    # Update user preferences for filtering
    if user_id not in st.session_state.user_feedback:
        st.session_state.user_feedback[user_id] = {"liked": set(), "disliked": set()}

    if liked:
        st.session_state.user_feedback[user_id]["liked"].add(movie_id)
    else:
        st.session_state.user_feedback[user_id]["disliked"].add(movie_id)


def render_movie_card(movie, user_id, idx):
    """Render a movie card with Tinder-like interface"""
    with st.container():
        # Center the card
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Card container with styling
            st.markdown(
                """
                <style>
                .movie-card {
                    background-color: black;
                    color: white;
                    border-radius: 15px;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin: 20px 0;
                    text-align: center;
                }
                .movie-title {
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .movie-rating {
                    color: #FFD700;
                    font-size: 18px;
                    margin-bottom: 10px;
                }
                .movie-genres {
                    color: #666;
                    margin-bottom: 15px;
                }
                .movie-description {
                    font-size: 16px;
                    line-height: 1.5;
                    margin-bottom: 20px;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Movie card content
            st.markdown(
                f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="movie-rating">⭐ {movie['avg_rating']:.1f} ({int(movie['rating_count'])} ratings)</div>
                    <div class="movie-genres">{movie['genres'].replace('|', ' • ')}</div>
                    <div class="movie-description">{movie['description']}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Action buttons
            col_left, col_right = st.columns(2)
            with col_left:
                if st.button("👎", key=f"dislike_{idx}", help="Skip this movie"):
                    with st.spinner("Recording feedback..."):
                        update_user_preferences(user_id, movie["movie_id"], liked=False)
                        time.sleep(0.3)
            with col_right:
                if st.button("❤️", key=f"like_{idx}", help="Add to favorites"):
                    with st.spinner("Recording feedback..."):
                        update_user_preferences(user_id, movie["movie_id"], liked=True)
                        time.sleep(0.3)


def get_next_recommendations(user_id: int, page_size: int = 1):
    """Get next batch of recommendations incorporating user feedback"""
    # Get lists of liked and disliked movies
    exclude_items = []
    if user_id in st.session_state.user_feedback:
        exclude_items = list(st.session_state.user_feedback[user_id]["liked"]) + list(
            st.session_state.user_feedback[user_id]["disliked"]
        )

    # Fetch recommendations from the hybrid engine with exclude_items
    recommendations = st.session_state.hybrid_engine.recommend(
        user_id,
        top_n=page_size * 5,  # Fetch more to account for filtering
        exclude_items=exclude_items,
    )

    if not recommendations:
        return []

    # Optionally, boost recommendations based on liked genres
    if user_id in st.session_state.user_feedback:
        liked_movies = st.session_state.user_feedback[user_id]["liked"]
        if liked_movies:
            liked_genres = set()
            for m_id in liked_movies:
                genres = st.session_state.movies_df.loc[
                    st.session_state.movies_df["movie_id"] == m_id, "genres"
                ].values
                if len(genres) > 0:
                    liked_genres.update(genres[0].split("|"))

            # Sort recommendations based on genre overlap
            recommendations = sorted(
                recommendations,
                key=lambda x: len(
                    set(
                        st.session_state.movies_df.loc[
                            st.session_state.movies_df["movie_id"] == x, "genres"
                        ]
                        .values[0]
                        .split("|")
                    )
                    & liked_genres
                ),
                reverse=True,
            )

    return recommendations[:page_size]


def retrain_model():
    """Retrain the model using accumulated feedback"""
    if st.session_state.feedback.empty:
        st.warning("No feedback available for retraining.")
        return

    # Append feedback to existing ratings_df
    updated_ratings_df = pd.concat(
        [st.session_state.ratings_df, st.session_state.feedback], ignore_index=True
    )

    # Update session state
    st.session_state.ratings_df = updated_ratings_df
    st.session_state.feedback = pd.DataFrame(
        columns=["user_id", "movie_id", "rating"]
    )  # Reset feedback

    # Retrain the model
    with st.spinner("Retraining the recommendation model..."):
        from train_model import train_and_save_model

        hybrid_engine, movies_df, ratings_df = train_and_save_model()

        # Reload the model into session state
        st.session_state.hybrid_engine = hybrid_engine
        st.session_state.movies_df = movies_df
        st.session_state.ratings_df = ratings_df

        # Update movie mappings
        unique_movie_ids = movies_df["movie_id"].unique()
        movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}
        reverse_movie_id_map = {v: k for k, v in movie_id_map.items()}

        st.session_state.movie_id_map = movie_id_map
        st.session_state.reverse_movie_id_map = reverse_movie_id_map

    st.success("Model retrained successfully!")


def main():
    # Header
    st.title("🎬 Youtube MoE Implementation - Powered by CoreRec")
    st.markdown("### Swipe right on movies you love!")

    # Sidebar with controls
    with st.sidebar:
        st.header("Your Profile")
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=6040,  # Adjust based on your dataset
            value=1,
            help="Your unique identifier",
        )

        # Show match stats
        if user_id in st.session_state.user_feedback:
            st.metric("❤️ Matches", len(st.session_state.user_feedback[user_id]["liked"]))
            st.metric("👎 Passed", len(st.session_state.user_feedback[user_id]["disliked"]))
        else:
            st.metric("❤️ Matches", 0)
            st.metric("👎 Passed", 0)

        st.markdown("---")

        # Retrain model button
        st.header("Model Training")
        if st.button("🛠️ Retrain Model with Feedback"):
            retrain_model()

    # Initialize recommender
    if not st.session_state.initialized:
        with st.spinner("Initializing recommendation system..."):
            initialize_recommender()
            st.session_state.initialized = True

    # Show one movie at a time
    recommendations = get_next_recommendations(user_id, page_size=1)
    if recommendations:
        movie_id = recommendations[0]
        movie = st.session_state.movies_df.loc[
            st.session_state.movies_df["movie_id"] == movie_id
        ].iloc[0]
        render_movie_card(movie, user_id, 0)
    else:
        st.info("No more movies to show! Try resetting your preferences or provide more feedback.")


if __name__ == "__main__":
    main()
