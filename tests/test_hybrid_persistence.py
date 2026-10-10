import pytest
import torch

from corerec.api.exceptions import SaveLoadError
from corerec.hybrid import RetrievalThenRerank
from corerec.ranking import PointwiseRanker


class Retriever(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([[.2, .8, .5]]))

    def forward(self, batch):
        return self.scores


class Reranker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([[.7, .1, .9]]))

    def forward(self, batch):
        return self.scores.gather(1, batch['candidate_indices'])


@pytest.mark.parametrize('training', [True, False])
def test_hybrid_safe_bundle_roundtrip(tmp_path, monkeypatch, training):
    monkeypatch.chdir(tmp_path)
    model = RetrievalThenRerank('two-stage', {'num_candidates': 2}, Retriever(), Reranker())
    with torch.no_grad():
        model.retriever.scores.mul_(2)
        model.reranker.scores.add_(.4)
    model.train(training)
    expected = model.recommend({}, {}, top_k=2)
    path = model.save('model.pt')
    assert path == 'model'
    assert not (tmp_path / 'model.pt').exists()
    loaded = RetrievalThenRerank.load(path, retriever=Retriever(), reranker=Reranker())
    assert loaded.name == model.name
    assert loaded.training == training
    assert loaded.config == model.config
    assert loaded.recommend({}, {}, top_k=2) == expected
    for key, weights in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], weights)
    with pytest.raises(ValueError, match='original neural architectures'):
        RetrievalThenRerank.load(path)


def test_hybrid_rejects_candidate_ranker():
    with pytest.raises(TypeError, match='RecommendationPipeline'):
        RetrievalThenRerank('hybrid', {}, Retriever(), PointwiseRanker())


def test_hybrid_legacy_manifest_reports_missing_weights(tmp_path):
    path = tmp_path / 'legacy.pt'
    torch.save({'name': 'old', 'config': {}, 'retriever_path': 'directory'}, path)
    with pytest.warns(UserWarning, match='arbitrary code'):
        with pytest.raises(SaveLoadError, match='no combined weights'):
            RetrievalThenRerank.load(path, allow_pickle=True)


def test_hybrid_trusted_combined_legacy_weights(tmp_path):
    original = RetrievalThenRerank('old', {'num_candidates': 2}, Retriever(), Reranker())
    path = tmp_path / 'combined.pt'
    torch.save({'model_name': original.name, 'config': original.config,
                'model_state_dict': original.state_dict()}, path)
    with pytest.warns(UserWarning, match='arbitrary code'):
        loaded = RetrievalThenRerank.load(path, retriever=Retriever(),
                                         reranker=Reranker(), allow_pickle=True)
    assert loaded.recommend({}, {}, top_k=2) == original.recommend({}, {}, top_k=2)
