# noise_handling implementation
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

class NOISE_HANDLING:
    def __init__(self, method='isolation_forest', contamination=0.1):
        """
        Initialize noise handling.
        
        Args:
            method (str): Noise detection method ('isolation_forest', 'zscore', 'iqr')
            contamination (float): Expected proportion of outliers in the dataset
        """
        self.method = method
        self.contamination = contamination
        self.outlier_detector = None
        self.scaler = RobustScaler()

    def fit_transform(self, X):
        """
        Detect and handle noisy samples in the data.
        
        Args:
            X: Input data
        """
        if self.method == 'isolation_forest':
            self.outlier_detector = IsolationForest(contamination=self.contamination)
            is_inlier = self.outlier_detector.fit_predict(X) == 1
            return X[is_inlier], is_inlier

        elif self.method == 'zscore':
            X_scaled = self.scaler.fit_transform(X)
            z_scores = np.abs(X_scaled)
            is_inlier = np.all(z_scores < 3, axis=1)
            return X[is_inlier], is_inlier

        elif self.method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            is_inlier = np.all((X > (Q1 - 1.5 * IQR)) & (X < (Q3 + 1.5 * IQR)), axis=1)
            return X[is_inlier], is_inlier

    def transform(self, X):
        """Apply noise detection to new data."""
        if self.method == 'isolation_forest':
            if self.outlier_detector is None:
                raise ValueError("Fit the detector first using fit_transform()")
            return X[self.outlier_detector.predict(X) == 1]
        else:
            return self.fit_transform(X)[0]
