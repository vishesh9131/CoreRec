import pytest
import torch

from corerec.hybrid import RetrievalThenRerank
from tests.test_hybrid_persistence import Retriever, Reranker


@pytest.mark.parametrize('values', [[-3., -2., -1.], [0., 0., 0.]])
@pytest.mark.parametrize('combined', [False, True])
def test_hybrid_recommend_only_retrieved_candidates(values, combined):
    retriever, reranker = Retriever(), Reranker()
    with torch.no_grad():
        reranker.scores.copy_(torch.tensor([values]))
    model = RetrievalThenRerank('hybrid', {'num_candidates': 2,
                                         'use_combined_score': combined}, retriever, reranker)
    recommendations = model.recommend({}, {}, top_k=10)
    assert {item for item, score in recommendations} == {1, 2}
    assert len(recommendations) == 2
    if not combined and values[0] < 0:
        assert recommendations == [(2, -1.), (1, -2.)]
    # Training retains dense, finite scores and differentiable component weights.
    scores = model({})
    assert scores.shape == (1, 3)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert reranker.scores.grad is not None


def test_hybrid_rerank_all_keeps_all_items():
    model = RetrievalThenRerank('hybrid', {'num_candidates': 1, 'rerank_all': True},
                               Retriever(), Reranker())
    assert {item for item, score in model.recommend({}, {}, top_k=10)} == {0, 1, 2}


def test_recommend_uses_one_retrieval_pass():
    retriever = Retriever()
    calls = []
    handle = retriever.register_forward_hook(lambda *args: calls.append(1))
    model = RetrievalThenRerank('hybrid', {'num_candidates': 2}, retriever, Reranker())
    model.recommend({}, {}, top_k=1)
    handle.remove()
    assert calls == [1]
