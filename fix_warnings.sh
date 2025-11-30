#!/bin/bash
# Script to fix common warnings in CoreRec

echo "Fixing CoreRec warnings..."
echo ""

# 1. Fix NumPy version (most important)
echo "1. Downgrading NumPy to <2.0.0 for PyTorch compatibility..."
pip install "numpy<2.0.0" --upgrade

# 2. Install ipywidgets for Jupyter
echo ""
echo "2. Installing ipywidgets for Jupyter progress bars..."
pip install ipywidgets

# 3. Verify installations
echo ""
echo "3. Verifying installations..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo ""
echo "Done! Restart your Jupyter kernel to apply changes."

