# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 1.10 or higher
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
print("✅ Installation successful!")
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
