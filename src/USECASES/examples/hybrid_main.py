import os
from threadpoolctl import threadpool_limits
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from corerec.engines.hybrid import HybridEngine
from corerec.engines.content_based import ContentBasedFilteringEngine
from corerec.engines.collaborative.cr_unionizedFactory import UnionizedRecommenderFactory
from corerec.output.formatted_output import OutputFormatter


def main():
    # Set environment variable to suppress OpenBLAS warning
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Load data directly
    data_input = "src/SANDBOX/dataset/BollywoodMovieDetail.csv"  # Example data file path
    preprocessed_data = pd.read_csv(data_input)

    if preprocessed_data.empty:
        logging.error("Preprocessed data is empty. Exiting the pipeline.")
        return

    # Feature Extraction
    from corerec.preprocessing.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    print(preprocessed_data.head())
    tfidf_matrix, vectorizer = extractor.extract_tfidf(preprocessed_data, text_column="directors")

    # Use TF-IDF features for content-based recommendations
    item_features = tfidf_matrix.values  # Use .values to convert DataFrame to NumPy array

    # Simulate a simple interaction matrix
    # For demonstration, create a random interaction matrix
    num_users = 10
    num_items = len(preprocessed_data)
    interaction_matrix = np.random.randint(2, size=(num_users, num_items))

    # Convert interaction matrix to CSR format
    interaction_matrix_csr = csr_matrix(interaction_matrix)

    # Initialize Engines
    config = {
        "method": "matrix_factorization",  # or "user_based_cf", "svd", etc.
        "params": {"num_factors": 20, "learning_rate": 0.01, "regularization": 0.02, "epochs": 20},
    }
    collaborative_engine = UnionizedRecommenderFactory.get_recommender(config)
    content_engine = ContentBasedFilteringEngine(item_features=item_features)

    # Initialize Hybrid Recommendation Engine
    hybrid_engine = HybridEngine(
        collaborative_engine=collaborative_engine,
        content_engine=content_engine,
        alpha=0.5,  # Adjust alpha as needed
    )

    # Train the collaborative engine
    user_ids = list(range(num_users))
    item_ids = preprocessed_data["imdbId"].tolist()  # Use 'imdbId' instead of 'directorId'
    hybrid_engine.train(interaction_matrix_csr, user_ids, item_ids)

    # Generate Recommendations for a given user
    user_id = 0  # Example: Recommend for the first user
    recommendations = hybrid_engine.recommend(user_id, top_n=5)

    # Format Output
    formatter = OutputFormatter()
    formatted_recs = formatter.format_recommendations(
        recommendations, item_metadata=preprocessed_data[["imdbId", "title"]]
    )
    print(formatted_recs)

    # Ensure item_ids includes all items
    item_ids = preprocessed_data["imdbId"].tolist()

    # Check if all item_ids are present in the content-based engine
    missing_items = [item for item in item_ids if item not in content_engine.item_ids]
    if missing_items:
        logging.error(f"Missing items in content-based engine: {missing_items}")


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
