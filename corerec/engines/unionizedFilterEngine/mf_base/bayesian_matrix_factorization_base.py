# Bayesian Matrix Factorization
# IMPLEMENTATION IN PROGRESS - NOT READY FOR PRODUCTION USE

"""
WARNING: This feature is not yet implemented.

Bayesian Matrix Factorization is planned for a future release.
This is a placeholder for the upcoming implementation.

For working matrix factorization algorithms, please use:
    - ALSRecommender (Alternating Least Squares)
    - MatrixFactorizationBase (SGD-based MF)
    - Neural Matrix Factorization

Example:
    from corerec.engines.unionized.mf_base import ALSRecommender
    model = ALSRecommender(factors=50, regularization=0.01)
    model.fit(user_ids, item_ids, ratings)

Expected release: v0.6.0 or later
"""


class BayesianMatrixFactorizationBase:
    """
    Bayesian Matrix Factorization - NOT YET IMPLEMENTED
    
    This class is a placeholder for future implementation.
    Attempting to use it will raise NotImplementedError with guidance
    on alternative algorithms to use.
    
    Planned Features (when implemented):
        - Probabilistic matrix factorization with Bayesian inference
        - Automatic uncertainty estimation
        - Hyperparameter learning via MAP or full Bayesian inference
        - Support for missing data and implicit feedback
    
    Raises:
        NotImplementedError: Always raised when attempting to instantiate
    """
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "\n\nBayesianMatrixFactorization is not yet implemented.\n\n"
            "This feature is planned for CoreRec v0.6.0 or later.\n\n"
            "For matrix factorization, please use one of these working alternatives:\n\n"
            "1. ALSRecommender (Alternating Least Squares):\n"
            "   from corerec.engines.unionized.mf_base import ALSRecommender\n"
            "   model = ALSRecommender(factors=50, regularization=0.01)\n\n"
            "2. MatrixFactorizationBase (SGD-based):\n"
            "   from corerec.engines.unionized.mf_base import MatrixFactorizationBase\n"
            "   model = MatrixFactorizationBase(n_factors=50)\n\n"
            "3. Neural Matrix Factorization:\n"
            "   from corerec.engines.unionized.nn_base import NCF\n"
            "   model = NCF(embedding_dim=64)\n\n"
            "Track implementation progress: https://github.com/vishesh9131/CoreRec/issues"
        )