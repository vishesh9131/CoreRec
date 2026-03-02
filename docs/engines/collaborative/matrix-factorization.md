# Matrix Factorization

Matrix Factorization (MF) techniques are the cornerstone of collaborative filtering. They work by decomposing the user-item interaction matrix into lower-dimensional user and item latent factors.

## Algorithms

### 1. Fast Matrix Factorization (SVD-like)
`MatrixFactorizationRecommender` is a standard implementation of matrix factorization using Gradient Descent. It learns user and item embeddings to minimize the reconstruction error of the ratings.

**Best for**: Explicit feedback datasets (e.g., star ratings 1-5).

::: corerec.engines.collaborative.mf_base.matrix_factorization_recommender.MatrixFactorizationRecommender
    options:
      show_root_heading: true
      show_source: true

---

### 2. Alternating Least Squares (ALS)
`ALSRecommender` uses the Alternating Least Squares optimization method. It is particularly effective for large-scale implicit feedback datasets (e.g., clicks, views) because it can parallelize computation and handle unobserved data efficiently.

**Best for**: Implicit feedback, large-scale data.

::: corerec.engines.collaborative.mf_base.als_recommender.ALSRecommender
    options:
      show_root_heading: true
      show_source: true

## Mathematical Details

For a user $u$ and item $i$, the predicted score $\hat{r}_{ui}$ is calculated as:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i
$$

Where:
*   $\mu$: Global bias
*   $b_u$: User bias
*   $b_i$: Item bias
*   $\mathbf{p}_u$: User latent vector
*   $\mathbf{q}_i$: Item latent vector
