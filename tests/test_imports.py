"""
Test imports for refactored CoreRec modules.

Validates that key imports work correctly after refactoring.
"""

import unittest


class TestImports(unittest.TestCase):
    """Test that key imports work after refactoring."""

    def test_base_recommender_import(self):
        """Test Base Recommender import."""
        from corerec.api.base_recommender import BaseRecommender

        self.assertIsNotNone(BaseRecommender)

    def test_exceptions_import(self):
        """Test exceptions import."""
        from corerec.api.exceptions import (
            CoreRecException,
            ModelNotFittedError,
            InvalidParameterError,
        )

        self.assertIsNotNone(CoreRecException)
        self.assertIsNotNone(ModelNotFittedError)
        self.assertIsNotNone(InvalidParameterError)

    def test_mixins_import(self):
        """Test mixins import."""
        from corerec.api.mixins import ModelPersistenceMixin, BatchProcessingMixin, ValidationMixin

        self.assertIsNotNone(ModelPersistenceMixin)
        self.assertIsNotNone(BatchProcessingMixin)
        self.assertIsNotNone(ValidationMixin)

    def test_dcn_import(self):
        """Test DCN model import."""
        from corerec.engines.dcn import DCN

        self.assertIsNotNone(DCN)

        # Test that DCN inherits from BaseRecommender
        from corerec.api.base_recommender import BaseRecommender

        self.assertTrue(issubclass(DCN, BaseRecommender))

    def test_dcn_initialization(self):
        """Test DCN can be initialized."""
        from corerec.engines.dcn import DCN

        model = DCN(epochs=1, verbose=False)
        self.assertEqual(model.name, "DCN")
        self.assertFalse(model.is_fitted)
        self.assertEqual(model.epochs, 1)

    def test_engines_module(self):
        """Test engines module import."""
        from corerec import engines

        self.assertIsNotNone(engines)
        self.assertTrue(hasattr(engines, "DCN"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
