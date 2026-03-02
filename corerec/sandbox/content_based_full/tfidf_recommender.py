import numpy as np
from corerec.api.base_recommender import BaseRecommender
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRecommender(BaseRecommender):
    """
    TF-IDF based content recommender.
    
    Uses TF-IDF vectorization to find similar items based on text content.
    Can recommend items by text query or by item similarity.
    
    Supports two initialization modes:
    1. Old API: TFIDFRecommender(feature_matrix) - for backward compatibility
    2. New API: TFIDFRecommender() then fit(items, docs)
    """
    
    def __init__(self, feature_matrix=None, name: Optional[str] = None, verbose: bool = False):
        super().__init__(name=name or "TFIDFRecommender", trainable=True, verbose=verbose)
        self.vectorizer = None
        self.tfidf_matrix = None
        self.item_to_index = {}
        self.index_to_item = {}
        self.docs = {}
        self.is_fitted = False
        self._similarity_matrix = None
        self.feature_matrix = None  # For old API
        
        # Support old API: if feature_matrix is provided, use it directly
        if feature_matrix is not None:
            if hasattr(feature_matrix, "toarray"):
                self.feature_matrix = feature_matrix.toarray()
            else:
                self.feature_matrix = np.array(feature_matrix)
            self._similarity_matrix = self._compute_similarity_matrix()
            self.is_fitted = True
            self.num_items = self.feature_matrix.shape[0]
    
    def _compute_similarity_matrix(self):
        """Compute cosine similarity matrix from feature matrix."""
        if self.feature_matrix is None:
            return None
        norm = np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        normalized_features = self.feature_matrix / norm
        similarity = np.dot(normalized_features, normalized_features.T)
        return similarity
    
    @property
    def similarity_matrix(self):
        """Get or compute the similarity matrix."""
        if self._similarity_matrix is not None:
            return self._similarity_matrix
        elif self.feature_matrix is not None:
            self._similarity_matrix = self._compute_similarity_matrix()
            return self._similarity_matrix
        elif self.tfidf_matrix is not None:
            # Compute from TF-IDF matrix if available
            self.feature_matrix = self.tfidf_matrix.toarray() if hasattr(self.tfidf_matrix, 'toarray') else np.array(self.tfidf_matrix)
            self._similarity_matrix = self._compute_similarity_matrix()
            return self._similarity_matrix
        else:
            raise ValueError("No feature matrix available. Fit the model first or provide feature_matrix to constructor.")
    
    def fit(self, items: List[Any], docs: Dict[Any, str], **kwargs) -> "TFIDFRecommender":
        """
        Fit the TF-IDF model on item documents.
        
        Args:
            items: List of item IDs
            docs: Dictionary mapping item IDs to text documents
            **kwargs: Additional arguments (unused)
            
        Returns:
            Self for method chaining
        """
        if not items:
            raise ValueError("items must be a non-empty list")
        if not docs:
            raise ValueError("docs must be a non-empty dictionary")
        
        # Build item to index mapping
        self.item_to_index = {item: idx for idx, item in enumerate(items)}
        self.index_to_item = {idx: item for item, idx in self.item_to_index.items()}
        self.docs = docs
        
        # Extract texts in order of items
        texts = [docs.get(item, "") for item in items]
        
        # Fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        self.is_fitted = True
        self.num_items = len(items)
        
        if self.verbose:
            print(f"{self.name} fitted on {len(items)} items")
        
        return self
    
    def predict(self, user_id: Any, item_id: Any, **kwargs) -> float:
        """
        Predict relevance score for a user-item pair.
        
        For TF-IDF, this returns a default score since we don't have user preferences.
        Consider using recommend_by_text or recommend instead.
        
        Args:
            user_id: User identifier (unused for content-based)
            item_id: Item identifier
            **kwargs: Additional arguments (unused)
            
        Returns:
            Default score of 0.5 (neutral)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if item_id not in self.item_to_index:
            return 0.0
        
        # Return neutral score since we don't have user preferences
        return 0.5
    
    def recommend(
        self, 
        user_id_or_indices: Any = None, 
        top_k: int = 10, 
        top_n: int = None,
        exclude_items: Optional[List[Any]] = None,
        **kwargs
    ) -> Union[List[Any], np.ndarray]:
        """
        Generate recommendations.
        
        Supports two modes:
        1. Old API: recommend(item_indices, top_n) - returns numpy array of indices
        2. New API: recommend(user_id, top_k) - returns list of item IDs
        
        Args:
            user_id_or_indices: User ID (new API) or list of item indices (old API)
            top_k: Number of recommendations (new API, alias for top_n)
            top_n: Number of recommendations (old API, takes precedence)
            exclude_items: Items to exclude (new API only)
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of recommended item IDs (new API) or numpy array of indices (old API)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        
        # Determine which API is being used
        top_n = top_n if top_n is not None else top_k
        
        # Old API: if user_id_or_indices is a list/array, treat as item indices
        if isinstance(user_id_or_indices, (list, np.ndarray)) or (hasattr(user_id_or_indices, '__iter__') and not isinstance(user_id_or_indices, str)):
            # Old API: recommend based on item indices using similarity matrix
            sim_matrix = self.similarity_matrix  # This will compute if needed
            if sim_matrix is None:
                raise ValueError("Similarity matrix not available. Use new API or provide feature_matrix in constructor.")
            
            item_indices = list(user_id_or_indices)
            combined_scores = np.zeros(sim_matrix.shape[1])
            for idx in item_indices:
                if 0 <= idx < len(combined_scores):
                    combined_scores += sim_matrix[idx]
            
            top_items = combined_scores.argsort()[-top_n:][::-1]
            return top_items
        
        # New API: recommend based on user (content-based without user history)
        exclude_items = exclude_items or []
        exclude_set = set(exclude_items)
        
        # If we have similarity matrix (old API mode), use it
        if self.similarity_matrix is not None:
            # Return items sorted by average similarity (simple heuristic)
            avg_similarities = self.similarity_matrix.mean(axis=0)
            top_indices = avg_similarities.argsort()[-top_n:][::-1]
            return top_indices.tolist()
        
        # Return items sorted by document length (simple heuristic for new API)
        items_with_scores = [
            (item, len(self.docs.get(item, "")))
            for item in self.item_to_index.keys()
            if item not in exclude_set
        ]
        items_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in items_with_scores[:top_k]]
    
    def recommend_by_text(self, query_text: str, top_n: int = 10) -> List[Any]:
        """
        Recommend items based on text query using TF-IDF similarity.
        
        Args:
            query_text: Text query to find similar items
            top_n: Number of recommendations to return
            
        Returns:
            List of recommended item IDs sorted by relevance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before recommendation")
        
        if not query_text or not query_text.strip():
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query_text])
        
        # Compute cosine similarity with all items
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top N items
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Return top items even if similarity is low (content-based can have low scores)
        recommendations = [
            self.index_to_item[idx] 
            for idx in top_indices 
            if idx in self.index_to_item
        ]
        
        return recommendations
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
            **kwargs: Additional arguments (unused)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            print(f"{self.name} saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "TFIDFRecommender":
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
            **kwargs: Additional arguments (unused)
            
        Returns:
            Loaded TFIDFRecommender instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, TFIDFRecommender):
            raise ValueError(f"Loaded object is not a TFIDFRecommender instance")
        
        if model.verbose:
            print(f"{model.name} loaded from {path}")
        
        return model
