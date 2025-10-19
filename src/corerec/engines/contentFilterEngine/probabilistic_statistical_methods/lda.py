# lda implementation (Latent Dirichlet Allocation)
import logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Any

# Configure logging
logger = logging.getLogger(__name__)

class LDA:
    def __init__(self, n_components: int = 10, max_iter: int = 10):
        """
        Initialize the LDA model with the specified number of topics.

        Parameters:
        - n_components (int): Number of topics.
        - max_iter (int): Maximum number of iterations for the EM algorithm.
        """
        self.vectorizer = CountVectorizer(stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, random_state=42)
        logger.info(f"LDA initialized with {n_components} topics and {max_iter} max iterations.")

    def fit(self, documents: List[str]):
        """
        Fit the LDA model on the provided documents.

        Parameters:
        - documents (List[str]): List of documents to train the model.
        """
        logger.info("Fitting LDA model on documents.")
        count_matrix = self.vectorizer.fit_transform(documents)
        self.lda_model.fit(count_matrix)
        logger.info("LDA model training completed.")

    def transform(self, documents: List[str]) -> Any:
        """
        Transform documents into the LDA topic space.

        Parameters:
        - documents (List[str]): List of documents to transform.

        Returns:
        - Transformed document matrix in topic space.
        """
        logger.info("Transforming documents into LDA topic space.")
        count_matrix = self.vectorizer.transform(documents)
        return self.lda_model.transform(count_matrix)

    def recommend(self, query: str, top_n: int = 10) -> List[int]:
        """
        Recommend items based on the similarity of the query to the topics.

        Parameters:
        - query (str): The query text for which to generate recommendations.
        - top_n (int): Number of top recommendations to return.

        Returns:
        - List[int]: List of recommended item indices.
        """
        logger.info("Generating recommendations using LDA.")
        query_vec = self.transform([query])
        topic_distribution = self.lda_model.transform(self.vectorizer.transform([query]))
        similarity_scores = (topic_distribution @ self.lda_model.components_.T).flatten()
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        logger.info(f"Top {top_n} recommendations generated using LDA.")
        return top_indices.tolist()
