"""The API reference pages' examples run (#225).

pipeline, retrieval, multimodal and constants opened with examples that
called the API with arguments it doesn't take; test_docs_imports.py passed
because the imports resolved. Every python block of these pages runs here,
in order, in a temp directory.
"""

import re
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[1] / "docs" / "source" / "api"
# pages whose examples are complete programs; the others show fragments
# (`model.fit(...)` on a model the reader already has)
RUNNABLE = ["constants.md", "multimodal.md", "pipeline.md", "ranking.md", "reranking.md", "retrieval.md"]


@pytest.mark.parametrize("page", RUNNABLE)
def test_api_reference_example_runs(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = re.findall(r"```python\n(.*?)```", (API / page).read_text(), re.S)
    assert blocks, f"{page} has no python example"
    ns = {}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{page}[block {i}]", "exec"), ns)
        except Exception as e:
            pytest.fail(f"docs/source/api/{page} block {i}: {type(e).__name__}: {e}\n{block}")
