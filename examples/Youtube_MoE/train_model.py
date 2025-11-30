import os
import joblib
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from corerec.engines.unionizedFilterEngine.mf_base.matrix_factorization_recommender import (
    MatrixFactorizationRecommender,
)
from corerec.engines.contentFilterEngine.tfidf_recommender import TFIDFRecommender
from corerec.engines.hybrid import HybridEngine
from data_loader import load_movielens_data


def train_and_save_model(model_dir="examples/Youtube_MoE/models"):
    """Train and save the recommendation models"""
    logging.info("Loading MovieLens data...")
    movies_df, ratings_df = load_movielens_data()

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Create interaction matrix
    interaction_matrix = csr_matrix(
        (ratings_df["rating"].values, (ratings_df["user_id"].values, ratings_df["movie_id"].values))
    )

    # Train TF-IDF
    logging.info("Training TF-IDF model...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies_df["genres"] + " " + movies_df["description"])

    # Initialize and train recommenders
    tfidf_recommender = TFIDFRecommender(feature_matrix=tfidf_matrix)
    tfidf_recommender.fit(data=None)

    collaborative_recommender = MatrixFactorizationRecommender(
        num_factors=50,
        learning_rate=0.01,
        reg_user=0.02,
        reg_item=0.02,
        epochs=20,
        early_stopping_rounds=5,
        n_threads=4,
    )
    collaborative_recommender.fit(interaction_matrix)

    # Create hybrid engine
    hybrid_engine = HybridEngine(
        collaborative_engine=collaborative_recommender, content_engine=tfidf_recommender, alpha=0.5
    )

    # Save all components
    logging.info("Saving models and data...")
    joblib.dump(hybrid_engine, os.path.join(model_dir, "hybrid_engine.joblib"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
    movies_df.to_pickle(os.path.join(model_dir, "movies_df.pkl"))
    ratings_df.to_pickle(os.path.join(model_dir, "ratings_df.pkl"))

    logging.info("Model training and saving completed!")
    return hybrid_engine, movies_df, ratings_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_save_model()
