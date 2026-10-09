# Contributing to CoreRec

Thanks for helping. This page gets you from a fresh clone to a merged pull
request. Every command here is what CI runs, so if it passes locally it should
pass on the PR.

## Find something to work on

- Issues labelled [`good first issue`](https://github.com/vishesh9131/CoreRec/labels/good%20first%20issue)
  are scoped to one or two files and say which test proves the fix.
- Comment on the issue before starting so two people don't do the same work.
- For anything larger (a new model, an API change), open an issue first and
  describe the change. It saves a rewrite later.

## Set up

Python 3.10 or newer. On Apple Silicon use a native arm64 Python
(for example Miniforge arm64); an x86_64 Python runs under Rosetta and is
several times slower.

```bash
git clone https://github.com/<your-username>/CoreRec.git
cd CoreRec
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,serving,onnx]"
```

Check the install:

```bash
python -m pytest tests/test_model_contract.py -q
```

## Make a change

```bash
git checkout -b fix/short-description
```

Where things live:

| Path | What |
|---|---|
| `corerec/engines/` | The production models (`corerec models` lists them) |
| `corerec/nn/` | Building blocks and `Recommender` for custom torch models |
| `corerec/serving/` | `ModelServer`, feedback log, retrain, CLI artifacts |
| `corerec/export.py` | ONNX export |
| `corerec/evaluation/` | Metrics and evaluators |
| `corerec/sandbox/` | Experimental models: not tested in CI, not production |
| `tests/` | One pytest suite; CI runs all of it |
| `docs/source/` | Sphinx docs (Markdown via MyST) |

## Test

Run the suite the way CI does:

```bash
python -m pytest tests/ --tb=short --strict-markers -m "not docs_build" \
    --cov=corerec --cov-fail-under=40
```

The slow MovieLens-100K accuracy floors in `tests/test_benchmark_floors.py`
skip themselves unless the dataset is present.

What a fix needs:

- **A test that fails without the fix.** Put it next to related tests, and
  say in its docstring what used to go wrong.
- **Production models keep the shared contract.** `tests/test_model_contract.py`
  runs every model in `corerec.engines.MODELS` through `fit`, `recommend`,
  `exclude_items`, save/load and `ModelLoader`. A model that can't meet it goes
  in `KNOWN_DIVERGENT` with a reason; don't loosen the assertions.
- **No silent failures.** Don't swallow exceptions with `except Exception: pass`
  or turn an import error into `X = None`. Raise, or log and count it.

Lint (CI runs this exact check):

```bash
ruff check . --select=E9,F63,F7,F82 --no-fix
```

## Docs

If you change behaviour a user sees, update the page in `docs/source/` and
`CHANGELOG.md` (under `[Unreleased]`). Code in docs must run as written. Build
locally with:

```bash
pip install sphinx sphinx-book-theme sphinx-copybutton sphinx-design myst-parser
python -m sphinx -b html docs/source docs/build/html
```

## Open the pull request

- One change per PR. Link the issue (`Fixes #123`).
- Describe what was wrong, how you fixed it, and how you tested it. Numbers
  help: before/after timings, NDCG, memory.
- Keep commits as your own work under your own GitHub account.
- CI must be green. A maintainer reviews, may ask for changes, then merges.

## Questions

Open an issue with the `question` label, or email vishesh@corerec.tech.
