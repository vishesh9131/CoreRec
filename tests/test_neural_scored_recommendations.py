from unittest.mock import patch

import numpy as np
import pytest

from tests.test_model_contract import MODELS, _build


NEURAL = [m for m in MODELS if m[0] in {'twotower', 'lightgcn', 'hstu', 'sasrec', 'multvae', 'multidae'}]


@pytest.mark.parametrize('model_id,module_path,cls_name,kwargs', NEURAL, ids=[m[0] for m in NEURAL])
def test_neural_scored_rankings_survive_persistence(model_id, module_path, cls_name, kwargs, tmp_path):
    model = _build(module_path, cls_name, kwargs)
    model.fit(['01', '01', '02', '02', '03', '03'], ['a', 'b', 'b', 'c', 'c', 'd'])
    path = tmp_path / 'model'
    model.save(path)
    for candidate in (model, type(model).load(path)):
        ids = candidate.recommend('01', top_k=4)
        with patch.object(candidate, 'predict', side_effect=AssertionError('Extra pairwise inference')):
            scored = candidate.recommend('01', top_k=4, return_scores=True)
        assert [item for item, score in scored] == ids
        assert set(ids) == {'c', 'd'}
        for item, score in scored:
            assert isinstance(score, float)
            assert np.isfinite(score)
            assert score == pytest.approx(candidate.predict('01', item), rel=1e-5, abs=1e-6)
        assert set(candidate.recommend('01', top_k=4, exclude_seen=False)) == {'a', 'b', 'c', 'd'}
        assert candidate.recommend('unknown', return_scores=True) == []
        assert candidate.recommend('01', top_k=4, exclude_items=['c', 'd'], return_scores=True) == []


def test_sasrec_scores_include_popularity_adjustment():
    from corerec.engines import SASRec

    model = SASRec(hidden_units=8, num_blocks=1, epochs=1, max_seq_length=4,
                   item_popularity_bias=True, device="cpu")
    model.fit(['01', '01', '02', '02'], ['a', 'b', 'b', 'c'])
    adjusted = model.recommend('01', top_k=3, exclude_seen=False, return_scores=True)
    model.item_popularity_bias = False
    raw = dict(model.recommend('01', top_k=3, exclude_seen=False, return_scores=True))
    for item, score in adjusted:
        assert score == pytest.approx(raw[item] - model.item_popularity[model.item_to_index[item] - 1],
                                      rel=1e-5, abs=1e-6)
    assert [score for item, score in adjusted] == sorted([score for item, score in adjusted], reverse=True)
