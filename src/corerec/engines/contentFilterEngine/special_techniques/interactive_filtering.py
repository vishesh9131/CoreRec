# interactive_filtering implementation

import logging
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

class InteractiveFilteringRecommender:
    def __init__(self, base_recommender: Any):
        """
        Initialize the InteractiveFilteringRecommender with a base recommender.

        Parameters:
        - base_recommender (Any): An instance of a base recommender (e.g., LSA, LDA).
        """
        self.base_recommender = base_recommender
        # Stores user feedback in the format {user_id: {item_id: feedback_score}}
        self.user_feedback: Dict[int, Dict[int, float]] = {}
        logger.info("InteractiveFilteringRecommender initialized with base recommender.")

    def collect_feedback(self, user_id: int, item_id: int, feedback_score: float):
        """
        Collect feedback from the user for a specific item.

        Parameters:
        - user_id (int): The ID of the user.
        - item_id (int): The ID of the item.
        - feedback_score (float): The feedback score (e.g., 1.0 for positive, -1.0 for negative).
        """
        if user_id not in self.user_feedback:
            self.user_feedback[user_id] = {}
        self.user_feedback[user_id][item_id] = feedback_score
        logger.info(f"Collected feedback from user {user_id} for item {item_id}: {feedback_score}")

        # Update the base recommender based on feedback
        self.update_recommender(user_id, item_id, feedback_score)

    def update_recommender(self, user_id: int, item_id: int, feedback_score: float):
        """
        Update the base recommender system based on user feedback.

        Parameters:
        - user_id (int): The ID of the user.
        - item_id (int): The ID of the item.
        - feedback_score (float): The feedback score.
        """
        try:
            # Example: Adjust user profile or model based on feedback
            # This is a placeholder for actual implementation
            logger.info(f"Updating base recommender for user {user_id} based on feedback.")
            if hasattr(self.base_recommender, 'update_user_profile'):
                self.base_recommender.update_user_profile(user_id, item_id, feedback_score)
            else:
                logger.warning("Base recommender does not support updating user profiles.")
        except Exception as e:
            logger.error(f"Error updating recommender based on feedback: {e}")

    def recommend(self, user_id: int, query: str, top_n: int = 10) -> List[int]:
        """
        Generate top-N item recommendations for a user, considering their feedback.

        Parameters:
        - user_id (int): The ID of the user.
        - query (str): The query text for generating recommendations.
        - top_n (int): Number of top recommendations to return.

        Returns:
        - List[int]: List of recommended item IDs.
        """
        logger.info(f"Generating recommendations for user {user_id} with query '{query}' using InteractiveFilteringRecommender.")
        # Get base recommendations
        base_recommendations = self.base_recommender.recommend(user_id, query, top_n=top_n * 2)

        if user_id in self.user_feedback:
            # Filter out items with negative feedback
            filtered_recommendations = [
                item_id for item_id in base_recommendations
                if self.user_feedback[user_id].get(item_id, 0) >= 0
            ]
            logger.info(f"Filtered recommendations for user {user_id}: {filtered_recommendations}")
            return filtered_recommendations[:top_n]
        else:
            return base_recommendations[:top_n]
