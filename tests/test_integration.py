import pytest
import numpy as np

from corerec.engines.unionizedFilterEngine.nn_base.FM_base import FM_base
from corerec.engines.unionizedFilterEngine.nn_base.FFM_base import FFM_base
from corerec.engines.unionizedFilterEngine.nn_base.Fibinet_base import Fibinet_base
from corerec.engines.unionizedFilterEngine.nn_base.FLEN_base import FLEN_base
from corerec.engines.unionizedFilterEngine.nn_base.FGCNN_base import FGCNN_base


class TestIntegration:
    @pytest.fixture
    def sample_data(self):
        # Generate common sample data format for most models
        data = []
        for i in range(100):
            data.append(
                {
                    "user_id": f"user_{i % 10}",
                    "item_id": f"item_{i % 20}",
                    "category": f"cat_{i % 5}",
                    "label": float(np.random.randint(0, 2)),
                }
            )
        return data

    def test_models_compatibility(self, sample_data):
        """Test that models can be trained on the same data format."""
        models = [
            FM_base(num_epochs=1, verbose=False),
            FFM_base(num_epochs=1, verbose=False),
            Fibinet_base(num_epochs=1, verbose=False),
            FLEN_base(num_epochs=1, verbose=False),
            FGCNN_base(
                channels=[32],
                kernel_heights=[3],
                pooling_sizes=[2],
                recombine_kernels=[2],
                num_epochs=1,
                verbose=False,
            ),
        ]

        user_features = {"user_id": sample_data[0]["user_id"]}
        item_pool = [{"item_id": f"item_{i}", "category": f"cat_{i % 5}"}
                     for i in range(5)]

        for model in models:
            model.fit(sample_data)
            assert model.is_fitted

            # Test prediction
            prediction = model.predict(sample_data[0])
            assert 0 <= prediction <= 1

            # Test recommendation
            recommendations = model.recommend(
                user_features, item_pool=item_pool, top_n=3)
            assert len(recommendations) == 3
