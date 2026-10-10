"""Every model constructor call in docs/source uses parameters the model has.

Tutorials passed ALS(n_factors=..., learning_rate=...) and DeepFM(deep_layers=...),
which either trained on defaults or raised on the first line (#86). This parses
each python block and builds the model with the same keyword arguments, so the
model's own constructor decides, aliases (ALS epochs=) included.
"""

import ast
import re
from pathlib import Path

import pytest

import corerec.engines as engines

SOURCE = Path(__file__).resolve().parents[1] / "docs" / "source"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def _calls():
    for page in sorted(SOURCE.rglob("*.md")):
        for block in BLOCK.findall(page.read_text(errors="ignore")):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue  # fragments with "..." placeholders
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                if name in engines.MODELS and node.keywords:
                    yield f"{page.relative_to(SOURCE)}:{node.lineno}", name, node.keywords


def _literal_kwargs(keywords):
    out = {}
    for kw in keywords:
        if kw.arg is None:
            continue  # **config
        try:
            out[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            pass  # a variable: its name can't be checked by building the model
    return out


CALLS = list(_calls())


@pytest.mark.parametrize("where,name,keywords", CALLS, ids=[c[0] for c in CALLS])
def test_docs_build_models_with_real_parameters(where, name, keywords):
    kwargs = _literal_kwargs(keywords)
    kwargs.pop("device", None)  # 'cuda' in a doc shouldn't fail on a CPU runner
    try:
        getattr(engines, name)(**kwargs)
    except TypeError as e:
        pytest.fail(f"{where}: {name}({', '.join(kwargs)}) -> {e}")
