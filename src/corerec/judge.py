# Copyright 2023 The UnionizedFilterEngine Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import mean_squared_error, mean_absolute_error


class judge:
    """Class to compute evaluation metrics for recommender systems."""

    @staticmethod
    def precision_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Compute Precision@K.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        relevant_items: Set[int]
            Set of relevant item IDs.
        k: int
            Number of top recommendations to consider.

        Returns:
        --------
        float
            Precision@K score.
        """
        top_k = recommended_items[:k]
        relevant_count = len([item for item in top_k if item in relevant_items])
        return relevant_count / k

    @staticmethod
    def recall_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Compute Recall@K.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        relevant_items: Set[int]
            Set of relevant item IDs.
        k: int
            Number of top recommendations to consider.

        Returns:
        --------
        float
            Recall@K score.
        """
        top_k = recommended_items[:k]
        relevant_count = len([item for item in top_k if item in relevant_items])
        return relevant_count / len(relevant_items) if len(relevant_items) > 0 else 0.0

    @staticmethod
    def f1_score_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Compute F1-Score@K.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        relevant_items: Set[int]
            Set of relevant item IDs.
        k: int
            Number of top recommendations to consider.

        Returns:
        --------
        float
            F1-Score@K score.
        """
        precision = EvaluationMetrics.precision_at_k(recommended_items, relevant_items, k)
        recall = EvaluationMetrics.recall_at_k(recommended_items, relevant_items, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG)@K.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        relevant_items: Set[int]
            Set of relevant item IDs.
        k: int
            Number of top recommendations to consider.

        Returns:
        --------
        float
            NDCG@K score.
        """
        top_k = recommended_items[:k]
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant_items:
                dcg += 1 / np.log2(i + 2)  # log2(i + 2) because indexing starts at 0
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate_at_k(recommended_items: List[int], relevant_items: Set[int], k: int) -> float:
        """
        Compute Hit Rate@K.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        relevant_items: Set[int]
            Set of relevant item IDs.
        k: int
            Number of top recommendations to consider.

        Returns:
        --------
        float
            Hit Rate@K score.
        """
        top_k = recommended_items[:k]
        return 1.0 if any(item in relevant_items for item in top_k) else 0.0

    @staticmethod
    def mse(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """
        Compute Mean Squared Error (MSE).

        Parameters:
        -----------
        predicted_ratings: List[float]
            List of predicted ratings.
        actual_ratings: List[float]
            List of actual ratings.

        Returns:
        --------
        float
            MSE score.
        """
        return mean_squared_error(actual_ratings, predicted_ratings)

    @staticmethod
    def mae(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """
        Compute Mean Absolute Error (MAE).

        Parameters:
        -----------
        predicted_ratings: List[float]
            List of predicted ratings.
        actual_ratings: List[float]
            List of actual ratings.

        Returns:
        --------
        float
            MAE score.
        """
        return mean_absolute_error(actual_ratings, predicted_ratings)

    @staticmethod
    def rmse(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """
        Compute Root Mean Squared Error (RMSE).

        Parameters:
        -----------
        predicted_ratings: List[float]
            List of predicted ratings.
        actual_ratings: List[float]
            List of actual ratings.

        Returns:
        --------
        float
            RMSE score.
        """
        return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    @staticmethod
    def coverage(recommended_items: List[int], all_items: Set[int]) -> float:
        """
        Compute Coverage.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        all_items: Set[int]
            Set of all possible item IDs.

        Returns:
        --------
        float
            Coverage score.
        """
        unique_recommended = set(recommended_items)
        return len(unique_recommended) / len(all_items) if len(all_items) > 0 else 0.0

    @staticmethod
    def diversity(recommended_items: List[int], item_similarity_matrix: np.ndarray) -> float:
        """
        Compute Diversity.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        item_similarity_matrix: np.ndarray
            Pairwise similarity matrix for items.

        Returns:
        --------
        float
            Diversity score.
        """
        if len(recommended_items) < 2:
            return 0.0
        similarities = []
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                similarities.append(
                    item_similarity_matrix[recommended_items[i], recommended_items[j]]
                )
        return 1 - np.mean(similarities) if similarities else 0.0

    @staticmethod
    def novelty(recommended_items: List[int], item_popularity: Dict[int, float]) -> float:
        """
        Compute Novelty.

        Parameters:
        -----------
        recommended_items: List[int]
            List of recommended item IDs.
        item_popularity: Dict[int, float]
            Dictionary mapping item IDs to their popularity scores.

        Returns:
        --------
        float
            Novelty score.
        """
        if not recommended_items:
            return 0.0
        return np.mean([-np.log2(item_popularity.get(item, 1e-10)) for item in recommended_items])
