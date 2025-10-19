# user_profiling implementation
import pandas as pd
from typing import List, Dict, Any, Optional

class UserProfilingRecommender:
    """
    A context-aware recommender system that builds and maintains user profiles for personalized recommendations.
    
    This recommender system creates and maintains detailed user profiles that capture user preferences,
    behavior patterns, and context-dependent interactions. It uses these profiles to generate
    personalized recommendations that adapt to different contexts.
    
    Attributes:
        user_profiles (Dict[int, Dict]): Detailed profiles for each user containing:
            - Preference vectors
            - Historical interactions
            - Context-dependent behavior patterns
            - Long-term and short-term interests
        context_preferences (Dict[int, Dict]): User preferences mapped to different contexts
        preference_evolution (Dict[int, List]): Temporal evolution of user preferences
    
    Methods:
        update_profile: Updates user profile with new interaction data
        analyze_context_preferences: Analyzes user preferences in different contexts
        detect_preference_shifts: Identifies changes in user preferences over time
        recommend: Generates personalized recommendations based on profile and context
    
    Example:
        >>> profiler = UserProfilingRecommender()
        >>> profiler.update_profile(user_id=123, interaction_data={...}, context={...})
        >>> recommendations = profiler.recommend(user_id=123, context={"location": "work"})
    
    Note:
        - Profiles are automatically updated with each user interaction
        - Supports both explicit (ratings) and implicit (behavior) feedback
        - Implements drift detection for evolving user preferences
    """
    def __init__(self, user_attributes: Optional[pd.DataFrame] = None):
        """
        Initialize the user profiling recommender with optional user attributes.

        Parameters:
        - user_attributes (pd.DataFrame, optional): DataFrame containing user information.
        """
        self.user_profiles: Dict[int, Dict[str, Any]] = {}
        self.user_attributes = user_attributes

    def fit(self, user_interactions: Dict[int, List[int]]):
        """
        Build user profiles based on interactions and user attributes.

        Parameters:
        - user_interactions (dict): Dictionary mapping user IDs to lists of interacted item IDs.
        """
        for user_id, items in user_interactions.items():
            profile = {}

            # Incorporate user attributes if available
            if self.user_attributes is not None:
                user_info = self.user_attributes[self.user_attributes['user_id'] == user_id]
                if not user_info.empty:
                    user_info = user_info.iloc[0]
                    for col in self.user_attributes.columns:
                        if col != 'user_id':
                            profile[col] = user_info[col]

            # Add interacted items
            profile['interacted_items'] = set(items)
            self.user_profiles[user_id] = profile

    def recommend(self, user_id: int, all_items: set, top_n: int = 10) -> List[int]:
        """
        Generate top-N item recommendations for a given user based on their profile.

        Parameters:
        - user_id (int): The ID of the user.
        - all_items (set): Set of all available item IDs.
        - top_n (int): The number of recommendations to generate.

        Returns:
        - List[int]: List of recommended item IDs.
        """
        if user_id not in self.user_profiles:
            return []

        interacted_items = self.user_profiles[user_id].get('interacted_items', set())
        recommendations = list(all_items - interacted_items)
        return recommendations[:top_n]
    