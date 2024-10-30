import numpy as np

class ALS:
    def __init__(self, num_factors=10, regularization=0.1, iterations=10):
        """
        Initialize the ALS model.
        
        Parameters:
        num_factors (int): Number of latent factors.
        regularization (float): Regularization parameter to prevent overfitting.
        iterations (int): Number of iterations to run the ALS algorithm.
        """
        self.num_factors = num_factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, ratings):
        """
        Fit the ALS model to the given ratings matrix.
        
        Parameters:
        ratings (ndarray): User-item ratings matrix.
        """
        num_users, num_items = ratings.shape
        self.user_factors = np.random.rand(num_users, self.num_factors)
        self.item_factors = np.random.rand(num_items, self.num_factors)

        for _ in range(self.iterations):
            for u in range(num_users):
                self.user_factors[u, :] = self._als_step(ratings[u, :], self.item_factors, self.user_factors[u, :])
            for i in range(num_items):
                self.item_factors[i, :] = self._als_step(ratings[:, i], self.user_factors, self.item_factors[i, :])

    def _als_step(self, ratings, fixed_vecs, solve_vec):
        """
        Perform one step of the ALS optimization.
        
        Parameters:
        ratings (ndarray): Ratings vector for a user or item.
        fixed_vecs (ndarray): Fixed vectors (item factors if updating user factors, and vice versa).
        solve_vec (ndarray): Vector to be updated (user factors if updating user, and vice versa).
        
        Returns:
        ndarray: Updated vector.
        """
        non_zero_indices = ratings.nonzero()[0]
        if len(non_zero_indices) == 0:
            return solve_vec

        fixed_vecs_non_zero = fixed_vecs[non_zero_indices, :]
        ratings_non_zero = ratings[non_zero_indices]

        YTY = fixed_vecs_non_zero.T.dot(fixed_vecs_non_zero)
        lambdaI = np.eye(YTY.shape[0]) * self.regularization
        YTCuPu = fixed_vecs_non_zero.T.dot(ratings_non_zero)

        return np.linalg.solve(YTY + lambdaI, YTCuPu)

    def predict(self, user, item):
        """
        Predict the rating for a given user and item.
        
        Parameters:
        user (int): User index.
        item (int): Item index.
        
        Returns:
        float: Predicted rating.
        """
        return self.user_factors[user, :].dot(self.item_factors[item, :].T)

    def recommend(self, user, num_items=10):
        """
        Recommend items for a given user.
        
        Parameters:
        user (int): User index.
        num_items (int): Number of items to recommend.
        
        Returns:
        ndarray: Indices of the recommended items.
        """
        scores = self.user_factors[user, :].dot(self.item_factors.T)
        return np.argsort(scores)[::-1][:num_items]