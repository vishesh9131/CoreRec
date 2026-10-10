# Installation Guide

## Requirements

- Python 3.10 or higher
- PyTorch 2.0 or higher
- NumPy, Pandas, SciPy

## Install from PyPI

```bash
pip install corerec
```

## Install from Source

```bash
git clone https://github.com/vishesh9131/CoreRec.git
cd CoreRec
pip install -e .
```

## Google Colab / Jupyter

Install into the running kernel with `%pip`. For the MovieLens example, include
`datasets`; the base install does not include `cr_learn`:

```text
%pip install --upgrade "corerec[datasets]"
```

To use the cloned source in Colab, run these in separate cells:

```text
!git clone --depth 1 https://github.com/vishesh9131/CoreRec.git
```

```text
%pip install "/content/CoreRec[datasets]"
```

Use a regular install here: an editable install (`-e`) creates a `.pth` file/import
hook that an already-running kernel may not have loaded. If you already installed
with `-e`, restart the session/kernel, then rerun the import and training cells.
`!pip show corerec` succeeding only confirms that shell pip sees the package.
If imports still fail, compare `sys.executable` with the interpreter used to
install; `%pip` targets the current kernel. Restart after upgrading dependencies
that were already imported, too.

```python
import corerec
from corerec.engines import DCN
from cr_learn import ml_1m
print(corerec.__version__)
```

The [Colab quickstart notebook](https://github.com/vishesh9131/CoreRec/blob/main/examples/colab_quickstart.ipynb)
trains on a small MovieLens sample. Run HTTP serving separately in a terminal:
install `corerec[serving]` first, then use `ModelServer(model).start()` in a script.
This blocking server call uses Uvicorn's event loop and is not part of the notebook
training cell; a Colab port is also not your computer's `localhost`.

## Optional Extras

```bash
# Serving (FastAPI REST API)
pip install "corerec[serving]"

# Tutorial datasets (cr_learn)
pip install "corerec[datasets]"

# Development + tests
pip install "corerec[dev]"

# Everything
pip install "corerec[all]"
```

See also: {doc}`api_versioning` (API stability policy) and {doc}`torch_nn_vendored` (internal PyTorch modules).

## Install cr_learn (for tutorials)

Tutorial examples use the `cr_learn` dataset package:

```bash
pip install cr_learn
```

The first run of `ml_1m.load()` downloads MovieLens 1M (~25 MB) to your local cache.

## Environment Notes

- **NumPy / PyTorch**: If you see NumPy compatibility warnings with PyTorch, use a matched pair, e.g. `pip install 'numpy<2'` with older PyTorch builds, or upgrade PyTorch to a NumPy 2–compatible release.
- **GPU**: Deep learning models auto-detect CUDA when available; CPU works for small examples.

## Verify Installation

```python
import corerec
print(corerec.__version__)

# Test import
from corerec.engines.dcn import DCN
model = DCN()
print("Installation successful!")
```

## Troubleshooting

### CUDA Issues
If you encounter CUDA errors:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### ImportError
If you get import errors after installation:
```bash
pip install --upgrade corerec
```
