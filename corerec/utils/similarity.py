"""
Similarity Computation Utilities

Vectorized implementations of various similarity metrics for
item-item and user-user similarity matrices. Designed to work
with sparse co-occurrence matrices efficiently.

Most of these take a co-occurrence matrix as input rather than
raw interaction data - the caller handles matrix construction.
"""

import numpy as np
from scipy import sparse


def jaccard(cooccurrence: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Jaccard similarity from co-occurrence matrix.
    
    J(i,j) = |users who rated both| / |users who rated either|
           = C_ij / (C_ii + C_jj - C_ij)
    
    Args:
        cooccurrence: sparse matrix where C_ij = number of users
                     who interacted with both item i and j
    
    Returns:
        sparse similarity matrix
    """
    diag = cooccurrence.diagonal()
    
    # for each pair: union = pop_i + pop_j - intersection
    # doing this vectorized is a bit tricky with sparse
    co_arr = cooccurrence.toarray() if sparse.issparse(cooccurrence) else cooccurrence
    
    row_counts = diag.reshape(-1, 1)  # column vector
    col_counts = diag.reshape(1, -1)  # row vector
    
    union = row_counts + col_counts - co_arr
    
    # avoid div by zero
    union = np.maximum(union, 1e-10)
    
    similarity = co_arr / union
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def cosine_similarity(cooccurrence: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Cosine similarity from co-occurrence matrix.
    
    cos(i,j) = C_ij / sqrt(C_ii * C_jj)
    
    Args:
        cooccurrence: sparse co-occurrence matrix
    
    Returns:
        sparse similarity matrix
    """
    diag = cooccurrence.diagonal()
    
    # norms are sqrt of diagonal (self-cooccurrence = count)
    norms = np.sqrt(diag)
    norms = np.maximum(norms, 1e-10)  # no div by 0
    
    co_arr = cooccurrence.toarray() if sparse.issparse(cooccurrence) else cooccurrence
    
    # outer product of norms gives denominator
    denom = np.outer(norms, norms)
    
    similarity = co_arr / denom
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def lift(cooccurrence: sparse.spmatrix, n_users: int) -> sparse.csr_matrix:
    """
    Lift similarity - ratio of observed to expected co-occurrence.
    
    lift(i,j) = P(i,j) / (P(i) * P(j))
              = (C_ij/n) / ((C_ii/n) * (C_jj/n))
              = C_ij * n / (C_ii * C_jj)
    
    Args:
        cooccurrence: sparse co-occurrence matrix
        n_users: total number of users (for probability calc)
    
    Returns:
        sparse similarity matrix
    """
    diag = cooccurrence.diagonal()
    
    co_arr = cooccurrence.toarray() if sparse.issparse(cooccurrence) else cooccurrence
    
    # P(i) * P(j) = (C_ii/n) * (C_jj/n) = C_ii * C_jj / n^2
    # so lift = C_ij / (C_ii * C_jj / n) = C_ij * n / (C_ii * C_jj)
    denom = np.outer(diag, diag).astype(float)
    denom = np.maximum(denom, 1e-10)
    
    similarity = (co_arr * n_users) / denom
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def inclusion_index(cooccurrence: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Inclusion index - asymmetric similarity based on subset relationship.
    
    inclusion(i,j) = C_ij / min(C_ii, C_jj)
    
    High when one item's users are largely a subset of another's.
    
    Args:
        cooccurrence: sparse co-occurrence matrix
    
    Returns:
        sparse similarity matrix
    """
    diag = cooccurrence.diagonal()
    
    co_arr = cooccurrence.toarray() if sparse.issparse(cooccurrence) else cooccurrence
    n = len(diag)
    
    # min(C_ii, C_jj) for all pairs - need to broadcast
    row_counts = diag.reshape(-1, 1)
    col_counts = diag.reshape(1, -1)
    min_counts = np.minimum(row_counts, col_counts)
    min_counts = np.maximum(min_counts, 1e-10)
    
    similarity = co_arr / min_counts
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def mutual_information(cooccurrence: sparse.spmatrix, n_users: int) -> sparse.csr_matrix:
    """
    Pointwise mutual information between items.
    
    PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
             = log(C_ij * n / (C_ii * C_jj))
    
    Measures how much more likely items co-occur than by chance.
    
    Args:
        cooccurrence: sparse co-occurrence matrix
        n_users: total number of users
    
    Returns:
        sparse similarity matrix (can have negative values)
    """
    diag = cooccurrence.diagonal()
    
    co_arr = cooccurrence.toarray().astype(float)
    
    # avoid log(0) by masking zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(diag, diag).astype(float)
        
        # PMI = log(C_ij * n / (C_ii * C_jj))
        ratio = (co_arr * n_users) / np.maximum(denom, 1e-10)
        similarity = np.log(np.maximum(ratio, 1e-10))
        
        # where cooccurrence is 0, PMI is -inf, set to 0
        similarity[co_arr == 0] = 0.0
    
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def lexicographers_mutual_information(
    cooccurrence: sparse.spmatrix, 
    n_users: int
) -> sparse.csr_matrix:
    """
    Lexicographer's Mutual Information - PMI weighted by joint probability.
    
    LMI(i,j) = P(i,j) * PMI(i,j)
             = (C_ij/n) * log(C_ij * n / (C_ii * C_jj))
    
    Combines frequency with information content. More stable than raw PMI
    because rare but informative pairs get down-weighted.
    
    Args:
        cooccurrence: sparse co-occurrence matrix
        n_users: total number of users
    
    Returns:
        sparse similarity matrix
    """
    diag = cooccurrence.diagonal()
    
    co_arr = cooccurrence.toarray().astype(float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(diag, diag).astype(float)
        
        # P(i,j)
        joint_prob = co_arr / n_users
        
        # PMI part
        ratio = (co_arr * n_users) / np.maximum(denom, 1e-10)
        pmi = np.log(np.maximum(ratio, 1e-10))
        pmi[co_arr == 0] = 0.0
        
        # LMI = P(i,j) * PMI
        similarity = joint_prob * pmi
    
    np.fill_diagonal(similarity, 0.0)
    
    return sparse.csr_matrix(similarity)


def exponential_decay(timestamps: np.ndarray, reference_time: int, half_life: float) -> np.ndarray:
    """
    Apply exponential time decay to values.
    
    decay = 2^(-(reference_time - timestamp) / half_life)
    
    After half_life seconds, the weight drops to 0.5.
    This is more intuitive than arbitrary decay coefficients.
    
    Args:
        timestamps: array of timestamps (seconds)
        reference_time: current/reference time (seconds)
        half_life: time in seconds for weight to decay by half
    
    Returns:
        array of decay weights in [0, 1]
    """
    time_diff = reference_time - timestamps
    
    # 2^(-t/half_life) = exp(-t * ln(2) / half_life)
    decay = np.power(2.0, -time_diff / max(half_life, 1e-10))
    
    return decay


def get_top_k_scored_items(
    scores: np.ndarray, 
    top_k: int, 
    sort_top_k: bool = True
) -> tuple:
    """
    Get top-k items from score matrix efficiently.
    
    Args:
        scores: 2D array of shape (n_users, n_items) with scores
        top_k: number of items to return per user
        sort_top_k: whether to sort the top k by score (descending)
    
    Returns:
        tuple of (item_indices, item_scores) both shape (n_users, top_k)
    """
    n_users, n_items = scores.shape
    top_k = min(top_k, n_items)
    
    # argpartition is O(n) vs O(n log n) for full sort
    # gives us top k but not sorted
    partitioned_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
    
    # gather the actual scores for those indices
    row_idx = np.arange(n_users)[:, np.newaxis]
    top_scores = scores[row_idx, partitioned_idx]
    
    if sort_top_k:
        # sort within the top k
        sorted_order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(partitioned_idx, sorted_order, axis=1)
        top_scores = np.take_along_axis(top_scores, sorted_order, axis=1)
    else:
        top_indices = partitioned_idx
    
    return top_indices, top_scores


def rescale(
    values: np.ndarray,
    target_min: float,
    target_max: float,
    source_min: np.ndarray,
    source_max: np.ndarray
) -> np.ndarray:
    """
    Rescale values from source range to target range.
    
    Maps [source_min, source_max] -> [target_min, target_max] linearly.
    source_min/max can be per-row for different scaling per user.
    
    Args:
        values: array to rescale
        target_min: minimum of target range
        target_max: maximum of target range
        source_min: minimum of source range (can be per-row)
        source_max: maximum of source range (can be per-row)
    
    Returns:
        rescaled values
    """
    source_range = source_max - source_min
    target_range = target_max - target_min
    
    # avoid div by zero for constant source
    source_range = np.where(source_range == 0, 1.0, source_range)
    
    normalized = (values - source_min) / source_range
    rescaled = normalized * target_range + target_min
    
    return rescaled
