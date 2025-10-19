# Installation

This guide will help you install CoreRec and its dependencies on your system.

## Requirements

CoreRec requires Python 3.8 or higher. Before installing CoreRec, make sure you have Python installed:

```bash
python --version
# or
python3 --version
```

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The easiest way to install CoreRec is using pip:

```bash
pip install --upgrade corerec
```

This will install the latest stable version of CoreRec along with all required dependencies.

### Method 2: Install from Source

For the latest development version, install directly from GitHub:

```bash
# Clone the repository
git clone https://github.com/vishesh9131/CoreRec.git
cd CoreRec

# Install in development mode
pip install -e .
```

### Method 3: Install Specific Version

To install a specific version of CoreRec:

```bash
pip install corerec==1.0.0
```

## Dependencies

CoreRec depends on the following packages, which will be installed automatically:

### Core Dependencies

- **PyTorch** (>=1.10.0): Deep learning framework
- **NumPy** (>=1.20.0): Numerical computing
- **Pandas** (>=1.3.0): Data manipulation
- **SciPy** (>=1.7.0): Scientific computing
- **scikit-learn** (>=1.0.0): Machine learning utilities

### Optional Dependencies

For additional functionality, you may want to install:

```bash
# For visualization
pip install matplotlib seaborn plotly

# For graph operations
pip install networkx

# For advanced deep learning models
pip install torch-geometric

# For distributed training
pip install horovod

# For experiment tracking
pip install wandb tensorboard

# For serving models
pip install fastapi uvicorn
```

## Verify Installation

After installation, verify that CoreRec is correctly installed:

```python
import corerec
print(corerec.__version__)
```

Or run a quick test:

```python
from corerec.engines.dcn import DCN
print("CoreRec installed successfully!")
```

## GPU Support

CoreRec supports both CPU and GPU computation. For GPU support, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

## Platform-Specific Instructions

=== "Linux"

    ```bash
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install python3-pip
    pip3 install --upgrade corerec
    ```

=== "macOS"

    ```bash
    # Using Homebrew
    brew install python3
    pip3 install --upgrade corerec
    ```

=== "Windows"

    ```bash
    # Using pip
    python -m pip install --upgrade pip
    pip install --upgrade corerec
    ```

## Virtual Environment (Recommended)

It's recommended to install CoreRec in a virtual environment:

=== "venv"

    ```bash
    # Create virtual environment
    python -m venv corerec_env
    
    # Activate (Linux/macOS)
    source corerec_env/bin/activate
    
    # Activate (Windows)
    corerec_env\Scripts\activate
    
    # Install CoreRec
    pip install corerec
    ```

=== "conda"

    ```bash
    # Create conda environment
    conda create -n corerec_env python=3.10
    
    # Activate environment
    conda activate corerec_env
    
    # Install CoreRec
    pip install corerec
    ```

## Docker Installation

You can also use CoreRec with Docker:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install corerec

COPY . /app

CMD ["python", "your_script.py"]
```

Build and run:

```bash
docker build -t corerec-app .
docker run corerec-app
```

## Troubleshooting

### Common Issues

#### Import Error

If you encounter import errors:

```bash
pip install --upgrade --force-reinstall corerec
```

#### CUDA Out of Memory

If you get CUDA out of memory errors:

```python
# Use CPU instead
model = DCN(device='cpu')

# Or use a smaller batch size
model = DCN(batch_size=16)
```

#### Missing Dependencies

Install all optional dependencies:

```bash
pip install corerec[all]
```

### Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/vishesh9131/CoreRec/issues)
2. Search existing issues or create a new one
3. Contact: sciencely98@gmail.com

## Next Steps

Now that CoreRec is installed, you can:

- Follow the [Quick Start Guide](quickstart.md) to build your first recommender
- Explore the [Architecture Overview](architecture.md) to understand CoreRec's design
- Check out [Examples](../examples/index.md) for real-world use cases

---

!!! success "Installation Complete!"
    You're all set! Head over to the [Quick Start Guide](quickstart.md) to build your first recommendation system.


