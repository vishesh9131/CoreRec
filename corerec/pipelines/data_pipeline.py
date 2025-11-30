"""
Data Pipeline for Recommendation Data

Composable pipeline for data transformation and preprocessing.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import List, Callable, Any
from abc import ABC, abstractmethod
import pandas as pd


class DataTransformer(ABC):
    """
    Base class for data transformers.

    All transformers should inherit from this and implement transform().

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data.

        Args:
            data: Input DataFrame

        Returns:
            Transformed DataFrame

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        pass

    def fit(self, data: pd.DataFrame) -> "DataTransformer":
        """
        Fit transformer to data (optional).

        Override this if your transformer needs to learn from data.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        return self

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class DataPipeline:
    """
    Composable data transformation pipeline.

    Chains multiple transformers together for clean data processing.

    Architecture:

    Raw Data → [Transformer 1] → [Transformer 2] → ... → Clean Data
         ↓           ↓                ↓                     ↓
    Missing Values  Encoding      Scaling            Ready for Model

    Example:
        from corerec.pipelines import DataPipeline
        from corerec.pipelines import MissingValueHandler, CategoryEncoder

        pipeline = DataPipeline()
        pipeline.add(MissingValueHandler('mean'))
        pipeline.add(CategoryEncoder())

        # Transform data
        clean_data = pipeline.transform(raw_data)

        # Or fit then transform
        pipeline.fit(train_data)
        clean_train = pipeline.transform(train_data)
        clean_test = pipeline.transform(test_data)  # Uses fitted params

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self):
        """
        Initialize empty pipeline.

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.transformers: List[DataTransformer] = []
        self._is_fitted = False

    def add(self, transformer: DataTransformer) -> "DataPipeline":
        """
        Add a transformer to the pipeline.

        Args:
            transformer: Data transformer to add

        Returns:
            self for method chaining

        Example:
            pipeline.add(MissingValueHandler()).add(CategoryEncoder())

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.transformers.append(transformer)
        return self

    def fit(self, data: pd.DataFrame) -> "DataPipeline":
        """
        Fit all transformers to data.

        Args:
            data: Training data to fit transformers on

        Returns:
            self for method chaining

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        current_data = data

        for transformer in self.transformers:
            transformer.fit(current_data)
            current_data = transformer.transform(current_data)

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers to data.

        Args:
            data: Data to transform

        Returns:
            Transformed data

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        current_data = data

        for transformer in self.transformers:
            current_data = transformer.transform(current_data)

        return current_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            data: Data to fit and transform

        Returns:
            Transformed data

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.fit(data)
        return self.transform(data)

    def get_transformers(self) -> List[DataTransformer]:
        """Get list of transformers in pipeline."""
        return self.transformers.copy()

    def is_fitted(self) -> bool:
        """Check if pipeline is fitted."""
        return self._is_fitted
