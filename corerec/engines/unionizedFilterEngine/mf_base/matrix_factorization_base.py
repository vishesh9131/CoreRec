# corerec/engines/unionizedFilterEngine/mf_base/matrix_factorization_base.py
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional
from corerec.engines.unionizedFilterEngine.base_recommender import BaseRecommender
from concurrent.futures import ThreadPoolExecutor

class MatrixFactorizationBase(BaseRecommender):
    """
    MatrixFactorizationBase: A Scalable and Efficient Recommender System
    Author: Vishesh Yadav
    
    Overview:
    ----------
    The `MatrixFactorizationBase` class implements a matrix factorization-based
    recommendation system using stochastic gradient descent (SGD) for optimization.
    It is designed to be efficient and scalable, suitable for large-scale datasets.

    Key Features:
    --------------
    - **Vectorized Operations**: Utilizes NumPy's vectorized operations to perform
      computations efficiently, reducing the need for explicit loops.
    
    - **Bias Terms**: Incorporates user and item bias terms to capture inherent
      biases, improving prediction accuracy.
    
    - **Sparse Matrix Handling**: Leverages SciPy's sparse matrix capabilities to
      efficiently manage large, sparse interaction matrices, minimizing memory usage.
    
    - **Stochastic Gradient Descent (SGD)**: Employs SGD for parameter updates,
      enabling faster convergence and scalability with large datasets.
    
    - **Parallelized Factor Updates**: Utilizes multi-threading to update user and
      item factors in parallel, taking advantage of multi-core processors.
    
    - **Early Stopping**: Includes an early stopping mechanism based on validation
      loss to prevent overfitting and reduce unnecessary epochs.
    
    - **Regularization**: Provides separate regularization parameters for user and
      item factors, allowing fine control over model complexity.
    
    - **Xavier Initialization**: Uses Xavier initialization for factor initialization,
      ensuring better convergence properties.

    Parameters:
    -------------
    - `num_factors` (int): Number of latent factors for users and items.
    - `learning_rate` (float): Learning rate for SGD updates.
    - `reg_user` (float): Regularization parameter for user factors and biases.
    - `reg_item` (float): Regularization parameter for item factors and biases.
    - `epochs` (int): Number of training epochs.
    - `early_stopping_rounds` (Optional[int]): Number of epochs with no improvement
      on validation loss to trigger early stopping.
    - `n_threads` (int): Number of threads for parallel processing.

    Methods:
    ---------
    - `initialize_factors(num_users, num_items)`: Initializes user and item factors
      and biases using Xavier initialization.
    
    - `compute_loss(interaction_matrix)`: Computes the loss (MSE + regularization)
      for the given interaction matrix.
    
    - `fit(interaction_matrix, validation_matrix)`: Trains the model using the
      interaction matrix, with optional validation for early stopping.
    
    - `_sgd_step(interaction_matrix)`: Performs a single SGD step to update user
      and item factors and biases.

    Usage:
    -------
    This class is intended for use in recommendation systems where scalability and
    efficiency are critical. It is particularly well-suited for large datasets with
    sparse interactions, such as user-item ratings in collaborative filtering tasks.
    """
    def __init__(self, num_factors: int = 20, learning_rate: float = 0.01, 
                 reg_user: float = 0.02, reg_item: float = 0.02, 
                 epochs: int = 20, early_stopping_rounds: Optional[int] = None, 
                 n_threads: int = 4):
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.n_threads = n_threads
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = 0.0

    def initialize_factors(self, num_users: int, num_items: int):
        # Xavier Initialization for better convergence
        limit = np.sqrt(6 / (num_users + self.num_factors))
        self.user_factors = np.random.uniform(-limit, limit, (num_users, self.num_factors))
        self.item_factors = np.random.uniform(-limit, limit, (num_items, self.num_factors))
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_bias = 0.0

    def compute_loss(self, interaction_matrix: csr_matrix) -> float:
        # Vectorized computation of predictions
        predictions = self.user_factors.dot(self.item_factors.T) + self.user_bias[:, np.newaxis] + self.item_bias[np.newaxis, :] + self.global_bias
        errors = interaction_matrix.toarray() - predictions
        mse = np.sum(errors ** 2) / interaction_matrix.nnz
        reg = (self.reg_user * np.sum(self.user_factors ** 2) +
               self.reg_item * np.sum(self.item_factors ** 2) +
               self.reg_user * np.sum(self.user_bias ** 2) +
               self.reg_item * np.sum(self.item_bias ** 2))
        return mse + reg

    def fit(self, interaction_matrix: csr_matrix, validation_matrix: Optional[csr_matrix] = None):
        num_users, num_items = interaction_matrix.shape
        self.initialize_factors(num_users, num_items)
        self.global_bias = interaction_matrix.data.mean()
        
        best_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(self.epochs):
            self._sgd_step(interaction_matrix)
            loss = self.compute_loss(interaction_matrix)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

            if validation_matrix is not None:
                val_loss = self.compute_loss(validation_matrix)
                print(f"Validation Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if self.early_stopping_rounds and no_improve_epochs >= self.early_stopping_rounds:
                        print("Early stopping triggered.")
                        break

    def _sgd_step(self, interaction_matrix: csr_matrix):
        def update_user(user):
            interactions = interaction_matrix[user].indices
            for item in interactions:
                prediction = np.dot(self.user_factors[user], self.item_factors[item]) + self.user_bias[user] + self.item_bias[item] + self.global_bias
                error = interaction_matrix[user, item] - prediction

                # Update biases
                self.user_bias[user] += self.learning_rate * (error - self.reg_user * self.user_bias[user])
                self.item_bias[item] += self.learning_rate * (error - self.reg_item * self.item_bias[item])

                # Update factors
                user_factors_old = self.user_factors[user].copy()
                self.user_factors[user] += self.learning_rate * (error * self.item_factors[item] - self.reg_user * self.user_factors[user])
                self.item_factors[item] += self.learning_rate * (error * user_factors_old - self.reg_item * self.item_factors[item])

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            executor.map(update_user, range(interaction_matrix.shape[0]))