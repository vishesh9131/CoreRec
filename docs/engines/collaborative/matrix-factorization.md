# Matrix Factorization

Matrix Factorization (MF) techniques are the cornerstone of collaborative filtering. They work by decomposing the user-item interaction matrix into lower-dimensional user and item latent factors.

## Algorithms

### Alternating Least Squares (ALS)
`ALS` uses the Alternating Least Squares optimization method. It is particularly effective for large-scale implicit feedback datasets (e.g., clicks, views) because it can parallelize computation and handle unobserved data efficiently.

**Best for**: Implicit feedback, large-scale data.

::: corerec.engines.matrix_factorization.ALS
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
