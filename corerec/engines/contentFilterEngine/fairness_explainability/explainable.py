# explainable implementation
from typing import Dict, Any, Optional

class EXPLAINABLE:
    def __init__(self):
        """
        Initialize the explainable module.
        """
        self.explanations = {}

    def generate_explanation(self, user_id: int, item_id: int, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an explanation for why a particular item was recommended to a user.

        Parameters:
        - user_id (int): The ID of the user.
        - item_id (int): The ID of the recommended item.
        - context (dict, optional): The context in which the recommendation was made.

        Returns:
        - str: A textual explanation of the recommendation.
        """
        explanation = f"Item {item_id} was recommended to User {user_id} because "
        if context:
            explanation += f"of the context {context} and "
        explanation += "based on similar items and user preferences."
        self.explanations[(user_id, item_id)] = explanation
        return explanation

    def get_explanation(self, user_id: int, item_id: int) -> str:
        """
        Retrieve a previously generated explanation for a recommendation.

        Parameters:
        - user_id (int): The ID of the user.
        - item_id (int): The ID of the recommended item.

        Returns:
        - str: The explanation of the recommendation.
        """
        return self.explanations.get((user_id, item_id), "No explanation available.")
