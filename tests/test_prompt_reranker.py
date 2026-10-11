"""corerec.hybrid.prompt_reranker (#79: 20% covered).

No network: the LLM reply comes from a stub, and the OpenAI/Anthropic clients
talk to a fake aiohttp session that records the request.
"""

import pytest

pytest.importorskip("aiohttp")

import corerec.hybrid.prompt_reranker as pr  # noqa: E402
from corerec.hybrid.prompt_reranker import PromptReranker, PromptTemplate  # noqa: E402

ITEMS = [{"description": d} for d in ("jazz", "rock", "folk")]


def _reranker(reply, **config):
    r = PromptReranker("r", {"llm_config": {"api_key": "test"}, **config})
    r.prompts = []

    async def generate_async(prompt, **kw):
        r.prompts.append(prompt)
        return reply

    r.llm_client.generate_async = generate_async
    return r


def test_template_finds_placeholders_with_text_around_them():
    """Text after a placeholder became part of its name, so every format() raised."""
    t = PromptTemplate("Prefs: {user_preferences}\nItems:\n{items}\nPick one {{literally}}.")
    assert t.placeholders == ["user_preferences", "items"]
    assert t.format(user_preferences="x", items="y") == "Prefs: x\nItems:\ny\nPick one {literally}."


def test_template_names_the_missing_placeholder():
    with pytest.raises(ValueError, match="items"):
        PromptTemplate("{user_preferences} {items}").format(user_preferences="x")


def test_default_template_reranks():
    r = _reranker("2")
    assert r.rerank("likes rock", "evening", ITEMS, ["1", "2", "3"])[0] == ("2", 1.0)
    assert "2: rock" in r.prompts[0] and "likes rock" in r.prompts[0] and "evening" in r.prompts[0]


def test_an_id_inside_a_longer_id_is_not_picked():
    """The reply "12" used to give 1 and 2 a score of 1.0 too, and 1 ranked first."""
    r = _reranker("12")
    assert r.rerank("p", "c", ITEMS, ["1", "2", "12"]) == [("12", 1.0), ("1", 0.0), ("2", 0.0)]


def test_integer_ids_are_matched():
    """`1 in "..."` raised TypeError, which was logged and every item scored 0."""
    r = _reranker("Best item: 12")
    assert r.rerank("p", "c", ITEMS, [1, 2, 12]) == [(12, 1.0), (1, 0.0), (2, 0.0)]


def test_candidates_are_capped_at_max_candidates():
    r = _reranker("1", max_candidates=2)
    assert [i for i, _ in r.rerank("p", "c", ITEMS, [1, 2, 3])] == [1, 2]
    assert "3: folk" not in r.prompts[0]


def test_no_match_keeps_the_original_order():
    r = _reranker("none of these")
    assert r.rerank("p", "c", ITEMS, ["a", "b", "c"]) == [("a", 0.0), ("b", 0.0), ("c", 0.0)]


def test_batch_rerank_runs_every_request():
    r = _reranker("b")
    out = r.batch_rerank(["p1", "p2"], ["c1", "c2"], [ITEMS, ITEMS], [["a", "b", "c"], ["b", "x", "y"]])
    assert [o[0] for o in out] == [("b", 1.0), ("b", 1.0)]
    assert len(r.prompts) == 2


def test_forward_scores_one_value_per_request():
    r = _reranker("12")
    r.llm_client.generate = lambda prompt: "12"
    batch = {"user_preferences": ["p", "q"], "context": ["c", "c"],
             "items": [ITEMS, ITEMS], "item_ids": [[1, 2, 12], [1, 2, 3]]}
    assert r(batch).tolist() == [1.0, 0.0]


def test_unknown_llm_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported LLM type"):
        PromptReranker("r", {"llm_type": "palm"})


def test_training_is_a_no_op():
    r = _reranker("x")
    assert r.train_step({}, None) == {"loss": 0.0}
    assert r.validate_step({}) == {"val_loss": 0.0}


class _FakeSession:
    """Stands in for aiohttp.ClientSession; `reply` is (status, json body)."""

    reply = (200, {})
    sent = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json, headers):
        _FakeSession.sent.append((url, json, headers))
        status, body = _FakeSession.reply

        class _Resp:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def json(self):
                return body

            async def text(self):
                return str(body)

        resp = _Resp()
        resp.status = status
        return resp


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setattr(pr.aiohttp, "ClientSession", _FakeSession)
    _FakeSession.sent = []
    return _FakeSession


def test_openai_client_sends_a_chat_request_and_reads_the_reply(fake_http):
    fake_http.reply = (200, {"choices": [{"message": {"content": " 12 \n"}}]})
    client = pr.OpenAIClient(api_key="k", model_name="m")
    assert client.generate("pick one") == "12"
    url, body, headers = fake_http.sent[0]
    assert url.endswith("/chat/completions") and headers["Authorization"] == "Bearer k"
    assert body["model"] == "m" and body["messages"][-1] == {"role": "user", "content": "pick one"}


def test_anthropic_client_sends_a_messages_request_and_reads_the_reply(fake_http):
    fake_http.reply = (200, {"content": [{"text": "7"}]})
    client = pr.AnthropicClient(api_key="k", model_name="m")
    assert client.batch_generate(["a", "b"]) == ["7", "7"]
    url, body, headers = fake_http.sent[0]
    assert url.endswith("/v1/messages") and headers["x-api-key"] == "k"
    assert body["system"] and body["messages"] == [{"role": "user", "content": "a"}]


@pytest.mark.parametrize("client_cls", [pr.OpenAIClient, pr.AnthropicClient])
def test_api_errors_give_an_empty_reply(fake_http, client_cls):
    fake_http.reply = (429, {"error": "rate limited"})
    assert client_cls(api_key="k").generate("x") == ""
    fake_http.reply = (200, {"unexpected": True})
    assert client_cls(api_key="k").generate("x") == ""
