"""
Custom Exception Hierarchy for CoreRec

This module provides custom exceptions for better error handling
and debugging throughout the CoreRec framework.

Author: Vishesh Yadav (sciencely98@gmail.com)
"""


class CoreRecException(Exception):
    """Base exception for all CoreRec errors."""

    pass


class ModelNotFittedError(CoreRecException):
    """
    Exception raised when attempting to use a model that hasn't been fitted.

    Example:
        >>> model = SomeRecommender()
        >>> model.predict(user_id=1, item_id=10)
        ModelNotFittedError: Model must be fitted before making predictions.
    """

    def __init__(self, message="Model must be fitted before making predictions. Call fit() first."):
        self.message = message
        super().__init__(self.message)


class InvalidDataError(CoreRecException):
    """
    Exception raised when input data is invalid or malformed.

    Example:
        >>> model.fit(data="invalid")
        InvalidDataError: Expected DataFrame or dict, got str
    """

    pass


class InvalidParameterError(CoreRecException):
    """
    Exception raised when model parameters are invalid.

    Example:
        >>> model = SomeRecommender(embedding_dim=-5)
        InvalidParameterError: embedding_dim must be positive, got -5
    """

    pass


class SaveLoadError(CoreRecException):
    """
    Exception raised when model save/load operations fail.

    Example:
        >>> model.save('/invalid/path/model.pkl')
        SaveLoadError: Cannot save model to /invalid/path/model.pkl
    """

    pass


class RecommendationError(CoreRecException):
    """
    Exception raised when recommendation generation fails.

    Example:
        >>> model.recommend(user_id=999999)
        RecommendationError: Unknown user_id: 999999
    """

    pass


class ConfigurationError(CoreRecException):
    """
    Exception raised when model configuration is invalid.

    Example:
        >>> model = SomeRecommender(config={'invalid': 'config'})
        ConfigurationError: Missing required config key: 'embedding_dim'
    """

    pass
