"""
Data Transformers

Common data transformations for recommendation data.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from corerec.pipelines.data_pipeline import DataTransformer


class MissingValueHandler(DataTransformer):
    """
    Handle missing values in data.

    Strategies: mean, median, mode, drop, fill_value

    Example:
        handler = MissingValueHandler(strategy='mean')
        clean_data = handler.transform(data)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, strategy: str = "mean", fill_value: Any = None):
        """
        Initialize missing value handler.

        Args:
            strategy: Strategy ('mean', 'median', 'mode', 'drop', 'fill_value')
            fill_value: Value to fill if strategy='fill_value'

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values = {}

    def fit(self, data: pd.DataFrame) -> "MissingValueHandler":
        """Learn fill values from data."""
        if self.strategy == "mean":
            self.fill_values = data.mean().to_dict()
        elif self.strategy == "median":
            self.fill_values = data.median().to_dict()
        elif self.strategy == "mode":
            self.fill_values = data.mode().iloc[0].to_dict()

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value handling."""
        data = data.copy()

        if self.strategy == "drop":
            return data.dropna()
        elif self.strategy == "fill_value":
            return data.fillna(self.fill_value)
        else:
            for col, fill_val in self.fill_values.items():
                if col in data.columns:
                    data[col] = data[col].fillna(fill_val)
            return data


class CategoryEncoder(DataTransformer):
    """
    Encode categorical features to numeric.

    Example:
        encoder = CategoryEncoder()
        encoded_data = encoder.fit_transform(data)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize category encoder.

        Args:
            columns: Specific columns to encode (None = auto-detect)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.columns = columns
        self.encodings: Dict[str, Dict] = {}

    def fit(self, data: pd.DataFrame) -> "CategoryEncoder":
        """Learn category mappings."""
        cols_to_encode = self.columns or data.select_dtypes(include=["object", "category"]).columns

        for col in cols_to_encode:
            if col in data.columns:
                unique_values = data[col].unique()
                self.encodings[col] = {val: idx for idx, val in enumerate(unique_values)}

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply category encoding."""
        data = data.copy()

        for col, encoding in self.encodings.items():
            if col in data.columns:
                data[col] = data[col].map(encoding)

        return data


class FeatureScaler(DataTransformer):
    """
    Scale numerical features.

    Methods: standard, minmax, robust

    Example:
        scaler = FeatureScaler(method='standard')
        scaled_data = scaler.fit_transform(data)

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None):
        """
        Initialize feature scaler.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            columns: Columns to scale (None = all numeric)

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.method = method
        self.columns = columns
        self.params: Dict[str, Dict] = {}

    def fit(self, data: pd.DataFrame) -> "FeatureScaler":
        """Learn scaling parameters."""
        cols_to_scale = self.columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_scale:
            if col in data.columns:
                if self.method == "standard":
                    self.params[col] = {"mean": data[col].mean(), "std": data[col].std()}
                elif self.method == "minmax":
                    self.params[col] = {"min": data[col].min(), "max": data[col].max()}
                elif self.method == "robust":
                    self.params[col] = {
                        "median": data[col].median(),
                        "iqr": data[col].quantile(0.75) - data[col].quantile(0.25),
                    }

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        data = data.copy()

        for col, params in self.params.items():
            if col in data.columns:
                if self.method == "standard":
                    data[col] = (data[col] - params["mean"]) / (params["std"] + 1e-8)
                elif self.method == "minmax":
                    range_val = params["max"] - params["min"]
                    data[col] = (data[col] - params["min"]) / (range_val + 1e-8)
                elif self.method == "robust":
                    data[col] = (data[col] - params["median"]) / (params["iqr"] + 1e-8)

        return data


class DataValidator(DataTransformer):
    """
    Validate data quality.

    Checks for common issues and raises errors or warnings.

    Example:
        validator = DataValidator(required_columns=['user_id', 'item_id'])
        validator.transform(data)  # Raises error if validation fails

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(
        self, required_columns: Optional[List[str]] = None, max_missing_ratio: float = 0.3
    ):
        """
        Initialize data validator.

        Args:
            required_columns: Columns that must exist
            max_missing_ratio: Maximum allowed ratio of missing values

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.required_columns = required_columns or []
        self.max_missing_ratio = max_missing_ratio

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return unchanged if valid."""
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check missing values
        for col in data.columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > self.max_missing_ratio:
                raise ValueError(
                    f"Column '{col}' has {missing_ratio:.1%} missing values "
                    f"(max allowed: {self.max_missing_ratio:.1%})"
                )

        # Check for duplicates
        if data.duplicated().any():
            print(f"Warning: Found {data.duplicated().sum()} duplicate rows")

        return data
