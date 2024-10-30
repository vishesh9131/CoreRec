# feature_selection implementation
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

class FEATURE_SELECTION:
    def __init__(self, k=10, method='chi2'):
        """
        Initialize feature selection.
        
        Args:
            k (int): Number of top features to select
            method (str): Feature selection method ('chi2', 'variance', 'correlation')
        """
        self.k = k
        self.method = method
        self.selected_features = None
        self.feature_scores = None
        self.scaler = MinMaxScaler()

    def fit_transform(self, X, y=None):
        """
        Fit the feature selector and transform the data.
        
        Args:
            X: Input features
            y: Target variables (optional for some methods)
        """
        if self.method == 'chi2':
            # Scale features to non-negative for chi2
            X_scaled = self.scaler.fit_transform(X)
            selector = SelectKBest(chi2, k=self.k)
            X_selected = selector.fit_transform(X_scaled, y)
            self.feature_scores = selector.scores_
            self.selected_features = selector.get_support()
            return X_selected
            
        elif self.method == 'variance':
            # Select features based on variance
            variances = np.var(X, axis=0)
            top_k_idx = np.argsort(variances)[-self.k:]
            self.selected_features = np.zeros(X.shape[1], dtype=bool)
            self.selected_features[top_k_idx] = True
            self.feature_scores = variances
            return X[:, top_k_idx]
            
        elif self.method == 'correlation':
            # Select features based on correlation with target
            correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            top_k_idx = np.argsort(np.abs(correlations))[-self.k:]
            self.selected_features = np.zeros(X.shape[1], dtype=bool)
            self.selected_features[top_k_idx] = True
            self.feature_scores = correlations
            return X[:, top_k_idx]

    def transform(self, X):
        """Transform new data using selected features."""
        if self.selected_features is None:
            raise ValueError("Fit the selector first using fit_transform()")
        return X[:, self.selected_features]

    def get_feature_importance(self):
        """Return feature importance scores."""
        if self.feature_scores is None:
            raise ValueError("Fit the selector first using fit_transform()")
        return self.feature_scores
