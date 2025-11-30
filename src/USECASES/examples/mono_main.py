import os
from threadpoolctl import threadpool_limits
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from corerec.engines.hybrid import HybridEngine
from corerec.engines.unionizedFilterEngine.cr_unionizedFactory import UnionizedRecommenderFactory
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

    # Simulate a simple interaction matrix
    # For demonstration, create a random interaction matrix
    num_users = 10
    num_items = len(preprocessed_data)
    interaction_matrix = np.random.randint(2, size=(num_users, num_items))

    # Convert interaction matrix to CSR format
    interaction_matrix_csr = csr_matrix(interaction_matrix)

    # Initialize Collaborative Engine
    config = {
        "method": "svd",  # or "matrix_factorization", "user_based_cf", etc.
        "params": {"num_factors": 20, "learning_rate": 0.01, "regularization": 0.02, "epochs": 20},
    }
    collaborative_engine = UnionizedRecommenderFactory.get_recommender(config)

    # Initialize Hybrid Recommendation Engine without content-based engine
    hybrid_engine = HybridEngine(
        collaborative_engine=collaborative_engine,
        content_engine=None,  # No content-based engine
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


if __name__ == "__main__":
    with threadpool_limits(limits=1, user_api="blas"):
        main()
