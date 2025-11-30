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

## Install with Optional Dependencies

### For Deep Learning Models
```bash
pip install corerec[deeplearning]
```

### For Graph Models
```bash
pip install corerec[graph]
```

### For All Features
```bash
pip install corerec[all]
```

## Install cr_learn (for tutorials)

```bash
pip install cr_learn
```

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
