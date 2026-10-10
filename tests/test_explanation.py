"""corerec.explanation (#79: ~23% covered). Ids 0 are real ids."""

import pytest

from corerec.explanation import FeatureExplainer, GenerativeExplainer, HistoryExplainer

ITEMS = {7: {"genre": "sci-fi", "brand": "Acme"}, 8: {"genre": "drama"}}


@pytest.mark.parametrize("user", [0, 42])
def test_feature_explainer_uses_the_matching_feature(user):
    exp = FeatureExplainer(item_features=ITEMS, user_preferences={user: {"genre": ["sci-fi"]}})
    e = exp.explain(7, {"user_id": user})
    assert e.explanation_type == "feature_genre"
    assert e.text == "Matches your interest in sci-fi"
    assert exp.explain(8, {"user_id": user}).explanation_type == "generic"


def test_custom_templates_without_default_fall_back_instead_of_raising():
    """The docstring's own templates have no 'default' key; a match on brand raised KeyError."""
    exp = FeatureExplainer(item_features=ITEMS, user_preferences={1: {"brand": "Acme"}},
                           templates={"category": "Because you like {value}"})
    assert exp.explain(7, {"user_id": 1}).text == "Based on your preferences"


def test_feature_match_rules():
    exp = FeatureExplainer()
    assert exp._features_match("a", "a")
    assert exp._features_match("b", {"a", "b"})
    assert exp._features_match("Sci-Fi Thriller", "sci-fi")
    assert not exp._features_match(3, "3")


@pytest.mark.parametrize("user,item", [(0, 0), (5, 9)])
def test_history_explainer_names_the_most_similar_recent_item(user, item):
    sim = {(1, item): 0.2, (1, 3): 0.9}
    exp = HistoryExplainer(user_history={user: [item, 3]},
                           item_similarity=lambda a, b: sim.get((a, b), 0.0),
                           item_names={3: "Blade Runner"})
    e = exp.explain(1, {"user_id": user})
    assert e.text == "Because you liked Blade Runner" and e.supporting_items == [3]
    assert e.confidence == pytest.approx(0.9)


def test_history_explainer_without_similarity_uses_the_latest_item_even_if_it_is_0():
    e = HistoryExplainer(user_history={1: [5, 0]}).explain(9, {"user_id": 1})
    assert e.text == "Because you liked 0" and e.supporting_items == [0]


def test_history_explainer_unknown_user_is_generic():
    assert HistoryExplainer(user_history={1: [2]}).explain(9, {"user_id": 3}).explanation_type == "generic"


def test_generative_explainer_prompts_with_context_and_trims_the_answer():
    seen = []

    def llm(prompt):
        seen.append(prompt)
        return '"' + "x" * 200 + '"'

    exp = GenerativeExplainer(llm_fn=llm, item_context_fn=lambda i: {"title": "Dune"},
                              user_context_fn=lambda u: {"likes": "sci-fi"}, max_length=50)
    e = exp.explain(7, {"user_id": 0, "source": "collab"})
    assert "Item: title: Dune" in seen[0] and "User preferences: likes: sci-fi" in seen[0]
    assert "Recommendation source: collab" in seen[0]
    assert len(e.text) == 50 and e.text.endswith("...") and not e.text.startswith('"')


def test_generative_explainer_falls_back_without_an_llm_or_on_error():
    assert GenerativeExplainer().explain(1, {}).explanation_type == "generic"

    def broken(prompt):
        raise RuntimeError("rate limited")
    assert GenerativeExplainer(llm_fn=broken).explain(1, {}).text == "Recommended based on your preferences"


def test_explain_batch_returns_one_per_item():
    exp = HistoryExplainer(user_history={1: [2]})
    assert [e.item_id for e in exp.explain_batch([3, 4], {"user_id": 1})] == [3, 4]
