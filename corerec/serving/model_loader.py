"""
Model Loader for Production

Handles model loading with caching and versioning.

Author: Vishesh Yadav (mail: sciencely98@gmail.com)
"""

from typing import Optional, Dict, Any
from pathlib import Path
import logging


class ModelLoader:
    """
    Production model loader with caching.

    Loads models efficiently with caching to avoid repeated loading.

    Example:
        loader = ModelLoader()
        model = loader.load('models/ncf_v1.pkl')

        # Subsequent loads use cache
        model2 = loader.load('models/ncf_v1.pkl')  # From cache!

    Author: Vishesh Yadav (mail: sciencely98@gmail.com)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for model cache

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        self.cache: Dict[str, Any] = {}
        self.cache_dir = cache_dir
        self.logger = logging.getLogger("ModelLoader")

    def load(self, model_path: str, use_cache: bool = True) -> Any:
        """
        Load model from path.

        Args:
            model_path: Path to model file
            use_cache: Whether to use cached model

        Returns:
            Loaded model

        Author: Vishesh Yadav (mail: sciencely98@gmail.com)
        """
        # Check cache first
        if use_cache and model_path in self.cache:
            self.logger.info(f"Loading model from cache: {model_path}")
            return self.cache[model_path]

        # Load model
        self.logger.info(f"Loading model from disk: {model_path}")

        # Try serialization framework first
        try:
            from corerec.serialization import load_from_file

            model = load_from_file(model_path)
        except Exception as e:
            # Fallback to pickle
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)

        # Cache if requested
        if use_cache:
            self.cache[model_path] = model

        return model

    def clear_cache(self):
        """Clear the model cache."""
        self.cache.clear()
        self.logger.info("Model cache cleared")
