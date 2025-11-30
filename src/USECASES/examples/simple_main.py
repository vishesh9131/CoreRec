import os
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from corerec.engines.unionizedFilterEngine.cr_unionizedFactory import (
    UnionizedRecommenderFactory as RecommenderFactory,
)
from corerec.config.recommender_config import CONFIG


def main():
    logging.basicConfig(level=logging.INFO)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    data_input = "src/SANDBOX/dataset/BollywoodMovieDetail.csv"
    data = pd.read_csv(data_input)

    if data.empty:
        logging.error("Data is empty. Exiting the pipeline.")
        return

    num_users = 10
    num_items = len(data)
    interaction_matrix = np.random.randint(2, size=(num_users, num_items))
    interaction_matrix_csr = csr_matrix(interaction_matrix)
    user_ids = list(range(num_users))
    item_ids = data["imdbId"].tolist()

    cf_config = CONFIG["collaborative_filtering"]
    try:
        collaborative_recommender = RecommenderFactory.get_recommender(cf_config)
    except ValueError as e:
        logging.error(e)
        return

    with threadpool_limits(limits=1, user_api="blas"):
        try:
            collaborative_recommender.fit(interaction_matrix_csr, user_ids, item_ids)
        except Exception as e:
            logging.error(f"Failed to train recommender: {e}")
            return

    user_id = 0
    recommendations = collaborative_recommender.recommend(user_id, top_n=5)
    print(f"Recommendations for user {user_id}: {recommendations}")


if __name__ == "__main__":
    main()
