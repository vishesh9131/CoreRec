"""
Simple Algorithm for Recommendation (SAR)

A fast neighborhood-based collaborative filter using item-item similarity.
Works well on sparse implicit feedback data without needing gradient descent.

The basic idea:
1. Build a user-item affinity matrix (possibly time-weighted)
2. Compute item-item similarity from co-occurrence patterns  
3. Score items for a user by combining their history with similarity

This impl supports multiple similarity metrics, time decay, thresholding,
and score normalization. It can handle DataFrame or list-based input.

Reference: Microsoft Recommenders SAR implementation
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Any, Union
from scipy import sparse
import os
import pickle

from corerec.api.base_recommender import BaseRecommender
from corerec.api.exceptions import ModelNotFittedError, InvalidDataError
from corerec.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL, 
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
    SIM_COOCCURRENCE,
    SIM_COSINE,
    SIM_JACCARD,
    SIM_LIFT,
    SIM_INCLUSION_INDEX,
    SIM_MUTUAL_INFORMATION,
    SIM_LEXICOGRAPHERS_MI,
    SUPPORTED_SIMILARITY_TYPES,
)
from corerec.utils.similarity import (
    jaccard,
    cosine_similarity,
    lift,
    inclusion_index,
    mutual_information,
    lexicographers_mutual_information,
    exponential_decay,
    get_top_k_scored_items,
    rescale,
)

logger = logging.getLogger(__name__)


class SAR(BaseRecommender):
    """
    Simple Algorithm for Recommendation.
    
    A fast item-based collaborative filter that:
    - Uses co-occurrence to estimate item similarity
    - Applies time decay to weight recent interactions higher
    - Can normalize predictions to original rating scale
    - Supports multiple similarity metrics
    
    Good for:
    - Quick baselines without deep learning
    - Implicit feedback (views, clicks, purchases)
    - Situations where interpretability matters
    
    Not ideal for:
    - Very sparse data with few interactions per item
    - Cold start (needs some user history)
    - When item/user features are important
    
    Example:
        >>> model = SAR(similarity_type='jaccard')
        >>> model.fit(train_df)
        >>> recs = model.recommend_k_items(test_df, top_k=10)
    """

    def __init__(
        self,
        col_user: str = DEFAULT_USER_COL,
        col_item: str = DEFAULT_ITEM_COL,
        col_rating: str = DEFAULT_RATING_COL,
        col_timestamp: str = DEFAULT_TIMESTAMP_COL,
        col_prediction: str = DEFAULT_PREDICTION_COL,
        similarity_type: str = SIM_JACCARD,
        time_decay_coefficient: float = 30.0,
        time_now: Optional[int] = None,
        timedecay_formula: bool = False,
        threshold: int = 1,
        normalize: bool = False,
    ):
        """
        Initialize SAR model.
        
        Args:
            col_user: name of user column in input data
            col_item: name of item column
            col_rating: name of rating/weight column
            col_timestamp: name of timestamp column (needed if timedecay_formula=True)
            col_prediction: name for prediction column in output
            similarity_type: one of 'cooccurrence', 'cosine', 'jaccard', 'lift',
                           'inclusion_index', 'mutual_information', 'lexicographers_mi'
            time_decay_coefficient: half-life in days for time decay (default 30 days)
            time_now: reference time for decay calc (defaults to max timestamp in data)
            timedecay_formula: whether to apply time decay weighting
            threshold: minimum co-occurrence count to keep (filters noise)
            normalize: whether to rescale predictions to original rating range
        """
        super().__init__(name="SAR")
        
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        
        # validate similarity type
        if similarity_type not in SUPPORTED_SIMILARITY_TYPES:
            raise ValueError(
                f"similarity_type must be one of {SUPPORTED_SIMILARITY_TYPES}, "
                f"got '{similarity_type}'"
            )
        self.similarity_type = similarity_type
        
        # time decay params - convert days to seconds internally
        self.time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        
        # threshold for co-occurrence filtering
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        
        # normalization
        self.normalize = normalize
        self._col_unity_rating = "_unity_rating"
        self.unity_user_affinity = None
        self.rating_min = None
        self.rating_max = None
        
        # internal index columns
        self._col_user_id = "_user_idx"
        self._col_item_id = "_item_idx"
        
        # model state - populated by fit()
        self.user_affinity = None  # sparse user-item matrix
        self.item_similarity = None  # item-item similarity
        self.item_frequencies = None  # count of users per item
        self.user_frequencies = None  # count of items per user
        
        # index mappings
        self.user2index: Dict = {}
        self.index2user: Dict = {}
        self.item2index: Dict = {}
        self.index2item: Dict = {}
        
        self.n_users = 0
        self.n_items = 0
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def _set_index(self, df: pd.DataFrame) -> None:
        """
        Build continuous index mappings from raw user/item IDs.
        This saves memory vs using raw IDs as sparse matrix indices.
        """
        # using dict(enumerate(...)) to get 0-indexed mappings
        unique_items = df[self.col_item].unique()
        unique_users = df[self.col_user].unique()
        
        self.index2item = dict(enumerate(unique_items))
        self.index2user = dict(enumerate(unique_users))
        
        # invert for fast lookup
        self.item2index = {v: k for k, v in self.index2item.items()}
        self.user2index = {v: k for k, v in self.index2user.items()}
        
        self.n_items = len(self.item2index)
        self.n_users = len(self.user2index)
        
        # also set base class attributes
        self.num_users = self.n_users
        self.num_items = self.n_items
        self.uid_map = self.user2index
        self.iid_map = self.item2index
    
    # =========================================================================
    # AFFINITY MATRIX
    # =========================================================================
    
    def _compute_affinity_matrix(
        self, 
        df: pd.DataFrame, 
        rating_col: str
    ) -> sparse.csr_matrix:
        """
        Build sparse user-item affinity matrix.
        
        Uses COO format for construction (good for building),
        then converts to CSR (good for row slicing & mat mult).
        """
        rows = df[self._col_user_id].values
        cols = df[self._col_item_id].values
        data = df[rating_col].values
        
        affinity = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float64
        )
        
        return affinity.tocsr()
    
    # =========================================================================
    # TIME DECAY
    # =========================================================================
    
    def _compute_time_decay(
        self, 
        df: pd.DataFrame, 
        decay_column: str
    ) -> pd.DataFrame:
        """
        Apply exponential time decay to ratings/weights.
        
        More recent interactions get higher weight. The half-life param
        controls how quickly old interactions fade - after half_life days,
        an interaction's weight is 0.5x its original value.
        """
        # use latest timestamp as reference if not specified
        if self.time_now is None:
            self.time_now = df[self.col_timestamp].max()
        
        # compute decay factors
        decay_weights = exponential_decay(
            timestamps=df[self.col_timestamp].values,
            reference_time=self.time_now,
            half_life=self.time_decay_half_life
        )
        
        # apply decay
        df = df.copy()
        df[decay_column] = df[decay_column] * decay_weights
        
        # aggregate by user-item pair (in case of dups after grouping)
        # sum makes sense - if user interacted multiple times, combine weights
        df = df.groupby([self.col_user, self.col_item], as_index=False).sum()
        
        return df
    
    # =========================================================================
    # CO-OCCURRENCE MATRIX
    # =========================================================================
    
    def _compute_cooccurrence_matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """
        Compute item co-occurrence matrix.
        
        C[i,j] = number of users who interacted with both item i and j.
        This is U^T @ U where U is a binary user-item matrix.
        
        Applies threshold to filter out low-count pairs (noise).
        """
        # binary matrix - just presence/absence
        rows = df[self._col_user_id].values
        cols = df[self._col_item_id].values
        ones = np.ones(len(rows), dtype=np.float64)
        
        user_item_binary = sparse.coo_matrix(
            (ones, (rows, cols)),
            shape=(self.n_users, self.n_items)
        ).tocsr()
        
        # co-occurrence is basically a dot product
        item_cooccurrence = user_item_binary.T.dot(user_item_binary)
        
        # threshold - zero out low counts to reduce noise
        if self.threshold > 1:
            item_cooccurrence = item_cooccurrence.multiply(
                item_cooccurrence >= self.threshold
            )
        
        return item_cooccurrence.tocsr()
    
    # =========================================================================
    # SIMILARITY COMPUTATION
    # =========================================================================
    
    def _compute_similarity(self, cooccurrence: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Compute item similarity from co-occurrence matrix.
        
        Different metrics capture different notions of relatedness:
        - jaccard: good general purpose, handles popularity bias
        - cosine: classic, assumes items are vectors
        - lift: probabilistic, good for market basket
        - cooccurrence: raw counts, biased toward popular items
        - inclusion_index: asymmetric, finds subsets
        - mutual_information: information theoretic
        """
        logger.info(f"Computing {self.similarity_type} similarity")
        
        if self.similarity_type == SIM_COOCCURRENCE:
            # raw co-occurrence as similarity (simple but popularity-biased)
            return cooccurrence
        
        elif self.similarity_type == SIM_JACCARD:
            return jaccard(cooccurrence)
        
        elif self.similarity_type == SIM_COSINE:
            return cosine_similarity(cooccurrence)
        
        elif self.similarity_type == SIM_LIFT:
            return lift(cooccurrence, self.n_users)
        
        elif self.similarity_type == SIM_INCLUSION_INDEX:
            return inclusion_index(cooccurrence)
        
        elif self.similarity_type == SIM_MUTUAL_INFORMATION:
            return mutual_information(cooccurrence, self.n_users)
        
        elif self.similarity_type == SIM_LEXICOGRAPHERS_MI:
            return lexicographers_mutual_information(cooccurrence, self.n_users)
        
        else:
            # shouldn't hit this due to __init__ validation
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
    
    # =========================================================================
    # FIT
    # =========================================================================
    
    def fit(self, df: pd.DataFrame) -> "SAR":
        """
        Train the SAR model on interaction data.
        
        Args:
            df: DataFrame with user, item, rating columns.
                Must not have duplicate (user, item) pairs.
                If timedecay_formula=True, needs timestamp column too.
        
        Returns:
            self (for method chaining)
        
        Raises:
            InvalidDataError: if input has issues
        """
        # figure out which cols we need
        required_cols = [self.col_user, self.col_item, self.col_rating]
        if self.time_decay_flag:
            required_cols.append(self.col_timestamp)
        
        # basic validation
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise InvalidDataError(f"Missing required columns: {missing}")
        
        # check for duplicates
        if df[required_cols].duplicated().any():
            raise InvalidDataError(
                "Input contains duplicate (user, item) pairs. "
                "Aggregate or dedupe before calling fit()."
            )
        
        if not np.issubdtype(df[self.col_rating].dtype, np.number):
            raise InvalidDataError("Rating column must be numeric")
        
        # build index mappings
        if not self.user2index:  # allow pre-setting indices
            self._set_index(df)
        
        logger.info(f"Training SAR on {self.n_users} users, {self.n_items} items")
        
        # work with a copy to avoid modifying input
        work_df = df[required_cols].copy()
        
        # apply time decay if requested
        if self.time_decay_flag:
            logger.info("Applying time decay")
            work_df = self._compute_time_decay(work_df, self.col_rating)
        
        # add index columns
        work_df[self._col_user_id] = work_df[self.col_user].map(self.user2index)
        work_df[self._col_item_id] = work_df[self.col_item].map(self.item2index)
        
        # handle normalization setup
        if self.normalize:
            self.rating_min = work_df[self.col_rating].min()
            self.rating_max = work_df[self.col_rating].max()
            
            logger.info("Setting up normalization factors")
            work_df[self._col_unity_rating] = 1.0
            
            if self.time_decay_flag:
                # need decayed unity ratings for proper normalization
                work_df = self._compute_time_decay(work_df, self._col_unity_rating)
            
            self.unity_user_affinity = self._compute_affinity_matrix(
                work_df, self._col_unity_rating
            )
        
        # main affinity matrix
        logger.info("Building user affinity matrix")
        self.user_affinity = self._compute_affinity_matrix(work_df, self.col_rating)
        
        # co-occurrence and similarity
        logger.info("Computing item co-occurrence")
        item_cooccurrence = self._compute_cooccurrence_matrix(work_df)
        
        # item frequencies from diagonal
        self.item_frequencies = np.asarray(item_cooccurrence.diagonal()).flatten()
        
        logger.info("Computing item similarity")
        self.item_similarity = self._compute_similarity(item_cooccurrence)
        
        # cleanup
        del item_cooccurrence
        del work_df

        self.is_fitted = True
        logger.info("SAR training complete")
        
        return self

    # =========================================================================
    # SCORING
    # =========================================================================
    
    def score(
        self, 
        test: pd.DataFrame, 
        remove_seen: bool = False
    ) -> np.ndarray:
        """
        Score all items for users in the test set.
        
        Args:
            test: DataFrame with user column (items are scored for all users in test)
            remove_seen: set scores of seen items to -inf
        
        Returns:
            2D array of shape (n_test_users, n_items) with scores
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        # map test users to indices
        test_users = test[self.col_user].unique()
        user_indices = []
        for u in test_users:
            idx = self.user2index.get(u, None)
            if idx is None:
                raise ValueError(
                    f"User '{u}' not in training set. SAR can't score unknown users."
                )
            user_indices.append(idx)
        
        user_indices = np.array(user_indices)
        
        # score = affinity @ similarity
        # each user's history weighted by item similarities
        logger.info("Computing recommendation scores")
        scores = self.user_affinity[user_indices, :].dot(self.item_similarity)
        
        # convert to dense if still sparse
        if sparse.issparse(scores):
            scores = scores.toarray()
        
        # normalization to original rating scale
        if self.normalize:
            logger.info("Normalizing scores")
            counts = self.unity_user_affinity[user_indices, :].dot(self.item_similarity)
            if sparse.issparse(counts):
                counts = counts.toarray()
            
            # per-user min/max for rescaling
            user_min = counts.min(axis=1, keepdims=True) * self.rating_min
            user_max = counts.max(axis=1, keepdims=True) * self.rating_max
            
            # tile to match score shape
            user_min = np.tile(user_min, (1, scores.shape[1]))
            user_max = np.tile(user_max, (1, scores.shape[1]))
            
            scores = rescale(scores, self.rating_min, self.rating_max, user_min, user_max)
        
        # remove seen items if requested
        if remove_seen:
            logger.info("Removing seen items from scores")
            seen_mask = self.user_affinity[user_indices, :].toarray()
            scores = np.where(seen_mask > 0, -np.inf, scores)
        
        return scores
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    
    def recommend_k_items(
        self,
        test: pd.DataFrame,
        top_k: int = 10,
        sort_top_k: bool = True,
        remove_seen: bool = False,
    ) -> pd.DataFrame:
        """
        Get top-k recommendations for users in test set.
        
        Args:
            test: DataFrame with user column
            top_k: number of items to recommend per user
            sort_top_k: whether to sort results by score
            remove_seen: exclude items user interacted with in training
        
        Returns:
            DataFrame with columns [user, item, prediction]
        """
        scores = self.score(test, remove_seen=remove_seen)
        
        top_items, top_scores = get_top_k_scored_items(
            scores, top_k=top_k, sort_top_k=sort_top_k
        )
        
        # build result dataframe
        test_users = test[self.col_user].unique()
        
        # actual k might be less than requested if fewer items exist
        actual_k = top_items.shape[1]
        
        result = pd.DataFrame({
            self.col_user: np.repeat(test_users, actual_k),
            self.col_item: [self.index2item[i] for i in top_items.flatten()],
            self.col_prediction: top_scores.flatten()
        })
        
        # filter out invalid scores (from -inf)
        result = result.replace(-np.inf, np.nan).dropna()
        
        return result
    
    def recommend(
        self,
        user_id: Any,
        top_k: int = 10,
        exclude_items: Optional[List[Any]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Get top-k recommendations for a single user.
        
        This is the BaseRecommender interface method. For batch recommendations,
        use recommend_k_items() which is more efficient.
        
        Args:
            user_id: user to recommend for
            top_k: number of items
            exclude_items: additional items to exclude (beyond seen items)
        
        Returns:
            list of recommended item IDs
        """
        if not self.is_fitted:
            raise ModelNotFittedError()

        if user_id not in self.user2index:
            return []  # cold user, can't recommend

        user_idx = self.user2index[user_id]

        # get user's affinity vector and compute scores
        user_affinity = self.user_affinity[user_idx, :]
        scores = user_affinity.dot(self.item_similarity)

        if sparse.issparse(scores):
            scores = scores.toarray().flatten()
        else:
            scores = np.asarray(scores).flatten()

        # mask seen items
        seen = user_affinity.toarray().flatten() > 0
        scores[seen] = -np.inf

        # mask additional exclusions
        if exclude_items:
            for item in exclude_items:
                if item in self.item2index:
                    scores[self.item2index[item]] = -np.inf
        
        # get top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # filter out -inf scores and convert to item IDs
        result = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                result.append(self.index2item[idx])
        
        return result
    
    def predict(
        self, 
        user_id: Any, 
        item_id: Any, 
        **kwargs
    ) -> float:
        """
        Predict score for a single user-item pair.
        
        Args:
            user_id: user identifier
            item_id: item identifier
        
        Returns:
            predicted affinity score
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        if user_id not in self.user2index:
            return 0.0  # unknown user
        if item_id not in self.item2index:
            return 0.0  # unknown item
        
        user_idx = self.user2index[user_id]
        item_idx = self.item2index[item_id]
        
        # score for this item = affinity @ similarity[:, item]
        user_aff = self.user_affinity[user_idx, :]
        item_sim = self.item_similarity[:, item_idx]
        
        score = user_aff.dot(item_sim)
        
        if sparse.issparse(score):
            score = score.toarray().item()
        elif hasattr(score, 'item'):
            score = score.item()
        
        return float(score)
    
    def predict_batch(self, test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict scores for specific user-item pairs in test.
        
        Args:
            test: DataFrame with user and item columns
        
        Returns:
            DataFrame with user, item, prediction columns
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        # get all scores for users in test
        scores = self.score(test)
        
        # map to per-row scores
        test_users = test[self.col_user].unique()
        user_to_row = {u: i for i, u in enumerate(test_users)}
        
        # look up scores for each (user, item) pair
        user_indices = test[self.col_user].map(user_to_row).values
        item_indices = test[self.col_item].map(
            lambda x: self.item2index.get(x, -1)
        ).values
        
        predictions = []
        for ui, ii in zip(user_indices, item_indices):
            if ii >= 0:
                predictions.append(scores[ui, ii])
            else:
                # unknown item
                predictions.append(0.0)
        
        result = pd.DataFrame({
            self.col_user: test[self.col_user].values,
            self.col_item: test[self.col_item].values,
            self.col_prediction: predictions
        })
        
        return result
    
    # =========================================================================
    # ADDITIONAL METHODS
    # =========================================================================
    
    def get_popularity_based_topk(
        self,
        top_k: int = 10,
        sort_top_k: bool = True,
        items: bool = True
    ) -> pd.DataFrame:
        """
        Get most popular items or users.
        
        Useful as a fallback for cold users or as a simple baseline.
        
        Args:
            top_k: number of items/users to return
            sort_top_k: whether to sort by popularity
            items: if True return items, if False return users
        
        Returns:
            DataFrame with item/user and prediction (popularity count)
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        if items:
            frequencies = self.item_frequencies
            col = self.col_item
            idx_map = self.index2item
        else:
            # compute user frequencies if not cached
            if self.user_frequencies is None:
                self.user_frequencies = np.asarray(
                    self.user_affinity.getnnz(axis=1)
                ).flatten()
            frequencies = self.user_frequencies
            col = self.col_user
            idx_map = self.index2user
        
        # reshape for get_top_k
        freq_2d = frequencies.reshape(1, -1)
        
        top_indices, top_scores = get_top_k_scored_items(
            freq_2d, top_k=top_k, sort_top_k=sort_top_k
        )
        
        result = pd.DataFrame({
            col: [idx_map[i] for i in top_indices.flatten()],
            self.col_prediction: top_scores.flatten()
        })
        
        return result
    
    def get_item_based_topk(
        self,
        items: pd.DataFrame,
        top_k: int = 10,
        sort_top_k: bool = True
    ) -> pd.DataFrame:
        """
        Get recommendations based on seed items.
        
        Useful for cold users - provide some items they like and
        get similar item recommendations without needing training data.
        
        Args:
            items: DataFrame with item column (and optionally user, rating)
            top_k: number of recommendations
            sort_top_k: whether to sort by score
        
        Returns:
            DataFrame with user (if provided), item, prediction
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        # map items to indices, skip unknown
        item_ids = items[self.col_item].map(
            lambda x: self.item2index.get(x, np.nan)
        ).values
        
        # ratings if provided, else all 1s
        if self.col_rating in items.columns:
            ratings = items[self.col_rating].values
        else:
            ratings = np.ones(len(item_ids))
        
        # users if provided, else treat as single user
        has_users = self.col_user in items.columns
        if has_users:
            test_users = items[self.col_user]
            unique_users = test_users.unique()
            user_map = {u: i for i, u in enumerate(unique_users)}
            user_ids = test_users.map(user_map).values
            n_pseudo_users = len(unique_users)
        else:
            test_users = pd.Series(["_seed_user"] * len(item_ids))
            unique_users = ["_seed_user"]
            user_ids = np.zeros(len(item_ids), dtype=int)
            n_pseudo_users = 1
        
        # filter out unknown items
        valid = ~np.isnan(item_ids)
        item_ids = item_ids[valid].astype(int)
        ratings = ratings[valid]
        user_ids = user_ids[valid]
        
        if len(item_ids) == 0:
            return pd.DataFrame(columns=[self.col_user, self.col_item, self.col_prediction])
        
        # build pseudo affinity from seed items
        pseudo_affinity = sparse.coo_matrix(
            (ratings, (user_ids, item_ids)),
            shape=(n_pseudo_users, self.n_items)
        ).tocsr()
        
        # score via similarity
        scores = pseudo_affinity.dot(self.item_similarity)
        if sparse.issparse(scores):
            scores = scores.toarray()
        
        # mask seed items so we dont recommend them
        for ui, ii in zip(user_ids, item_ids):
            scores[ui, ii] = -np.inf
        
        top_items, top_scores = get_top_k_scored_items(
            scores, top_k=top_k, sort_top_k=sort_top_k
        )
        
        # actual k might be less than requested
        actual_k = top_items.shape[1]
        
        result = pd.DataFrame({
            self.col_user: np.repeat(unique_users, actual_k),
            self.col_item: [self.index2item[i] for i in top_items.flatten()],
            self.col_prediction: top_scores.flatten()
        })
        
        # filter invalid
        result = result.replace(-np.inf, np.nan).dropna()
        
        # drop user col if we made it up
        if not has_users:
            result = result.drop(columns=[self.col_user])
        
        return result
    
    def get_topk_most_similar_users(
        self,
        user: Any,
        top_k: int = 10,
        sort_top_k: bool = True
    ) -> pd.DataFrame:
        """
        Find users most similar to the given user.
        
        Similarity is based on affinity vectors (what items they interacted with).
        
        Args:
            user: user ID to find similar users for
            top_k: number of similar users
            sort_top_k: whether to sort by similarity
        
        Returns:
            DataFrame with user and prediction (similarity score)
        """
        if not self.is_fitted:
            raise ModelNotFittedError()
        
        if user not in self.user2index:
            raise ValueError(f"User '{user}' not in training set")
        
        user_idx = self.user2index[user]
        
        # user-user similarity via dot product of affinity vectors
        user_vec = self.user_affinity[user_idx, :]
        all_users = self.user_affinity
        
        similarities = user_vec.dot(all_users.T)
        if sparse.issparse(similarities):
            similarities = similarities.toarray()
        
        similarities = similarities.reshape(1, -1)
        
        # dont return the user itself
        similarities[0, user_idx] = -np.inf
        
        top_users, top_scores = get_top_k_scored_items(
            similarities, top_k=top_k, sort_top_k=sort_top_k
        )
        
        result = pd.DataFrame({
            self.col_user: [self.index2user[i] for i in top_users.flatten()],
            self.col_prediction: top_scores.flatten()
        })
        
        return result.replace(-np.inf, np.nan).dropna()
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    
    def save(self, path: str, **kwargs) -> None:
        """
        Save model to disk.
        
        Args:
            path: file path for the saved model
        """
        # ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        state = {
            # config
            "col_user": self.col_user,
            "col_item": self.col_item,
            "col_rating": self.col_rating,
            "col_timestamp": self.col_timestamp,
            "col_prediction": self.col_prediction,
            "similarity_type": self.similarity_type,
            "time_decay_half_life": self.time_decay_half_life,
            "time_decay_flag": self.time_decay_flag,
            "time_now": self.time_now,
            "threshold": self.threshold,
            "normalize": self.normalize,
            "rating_min": self.rating_min,
            "rating_max": self.rating_max,
            # indices
            "user2index": self.user2index,
            "index2user": self.index2user,
            "item2index": self.item2index,
            "index2item": self.index2item,
            "n_users": self.n_users,
            "n_items": self.n_items,
            # model data
            "user_affinity": self.user_affinity,
            "item_similarity": self.item_similarity,
            "item_frequencies": self.item_frequencies,
            "user_frequencies": self.user_frequencies,
            "unity_user_affinity": self.unity_user_affinity,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "SAR":
        """
        Load model from disk.
        
        Args:
            path: file path to load from
        
        Returns:
            loaded SAR model
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # reconstruct model
        model = cls(
            col_user=state["col_user"],
            col_item=state["col_item"],
            col_rating=state["col_rating"],
            col_timestamp=state["col_timestamp"],
            col_prediction=state["col_prediction"],
            similarity_type=state["similarity_type"],
            # convert half_life back to days for constructor
            time_decay_coefficient=state["time_decay_half_life"] / (24 * 60 * 60),
            timedecay_formula=state["time_decay_flag"],
            time_now=state["time_now"],
            threshold=state["threshold"],
            normalize=state["normalize"],
        )
        
        # restore state
        model.rating_min = state["rating_min"]
        model.rating_max = state["rating_max"]
        model.user2index = state["user2index"]
        model.index2user = state["index2user"]
        model.item2index = state["item2index"]
        model.index2item = state["index2item"]
        model.n_users = state["n_users"]
        model.n_items = state["n_items"]
        model.num_users = state["n_users"]
        model.num_items = state["n_items"]
        model.uid_map = state["user2index"]
        model.iid_map = state["item2index"]
        model.user_affinity = state["user_affinity"]
        model.item_similarity = state["item_similarity"]
        model.item_frequencies = state["item_frequencies"]
        model.user_frequencies = state["user_frequencies"]
        model.unity_user_affinity = state["unity_user_affinity"]
        model.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        
        return model
    
    # =========================================================================
    # LEGACY INTERFACE SUPPORT
    # =========================================================================
    
    def fit_from_lists(
        self,
        user_ids: List[Any],
        item_ids: List[Any],
        ratings: List[float],
        timestamps: Optional[List[int]] = None
    ) -> "SAR":
        """
        Fit model from separate lists instead of DataFrame.
        
        This is for compatibility with older code that passes lists.
        Internally converts to DataFrame and calls fit().
        
        Args:
            user_ids: list of user IDs
            item_ids: list of item IDs
            ratings: list of ratings/weights
            timestamps: optional list of timestamps
        
        Returns:
            self
        """
        data = {
            self.col_user: user_ids,
            self.col_item: item_ids,
            self.col_rating: ratings,
        }
        
        if timestamps is not None:
            data[self.col_timestamp] = timestamps
        
        df = pd.DataFrame(data)
        return self.fit(df)
