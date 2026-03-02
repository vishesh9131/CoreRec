# Contributing to CoreRec

First off, thank you for considering contributing to CoreRec! It's people like you that make CoreRec such a great tool for the recommendation systems community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## 🤝 Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inspiring community for all. Please be respectful and constructive.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Familiarity with recommendation systems (helpful but not required)

### Finding an Issue to Work On

1. Check the [Issues](https://github.com/vishesh9131/CoreRec/issues) page
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to let others know you're working on it

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/CoreRec.git
cd CoreRec
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Install CoreRec in editable mode
pip install -e .
```

### 3. Set Up Pre-commit Hooks (Optional but Recommended)

```bash
pre-commit install
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 🔧 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version, CoreRec version)
   - Code samples if applicable

### Suggesting Enhancements

1. Check if the enhancement has been suggested
2. Create an issue describing:
   - Use case and motivation
   - Proposed API or implementation
   - Examples of how it would be used

### Adding New Algorithms

1. Place the algorithm in the appropriate engine:
   - `corerec/engines/collaborative/` for collaborative filtering
   - `corerec/engines/content_based/` for content-based
   - `corerec/engines/` for deep learning models

2. Ensure it inherits from `BaseRecommender`

3. Implement required methods:
   - `fit()`
   - `predict()`
   - `recommend()`

4. Add comprehensive docstrings

5. Include unit tests

6. Add example usage in `examples/`

## 📝 Coding Standards

### Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters (not 79)
- Use Black for code formatting
- Use isort for import sorting

```bash
# Format your code
black corerec/
isort corerec/

# Check linting
flake8 corerec/
pylint corerec/
```

### Type Hints

All public functions should have type hints:

```python
def recommend(
    self, 
    user_id: int, 
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Generate recommendations for a user.
    
    Args:
        user_id: User identifier
        top_k: Number of recommendations to return
        
    Returns:
        List of (item_id, score) tuples
    """
    pass
```

### Documentation

- All public classes, methods, and functions need docstrings
- Follow NumPy/Google docstring format
- Include examples in docstrings where helpful

```python
def fit(self, data: pd.DataFrame) -> 'BaseRecommender':
    """
    Train the recommendation model.
    
    Args:
        data: Training data with columns ['user_id', 'item_id', 'rating']
        
    Returns:
        Self for method chaining
        
    Example:
        >>> model = MyRecommender()
        >>> model.fit(train_data).save('model.pkl')
    """
    pass
```

### Import Organization

```python
# Standard library imports
import os
from typing import List, Dict

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from corerec.api.base_recommender import BaseRecommender
from corerec.utils import logging
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=corerec --cov-report=html

# Run specific test file
pytest tests/test_dcn.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in `tests/` directory
- Mirror the source structure
- Use descriptive test names
- Aim for >80% code coverage

```python
import pytest
from corerec.engines import DCN

class TestDCN:
    @pytest.fixture
    def sample_data(self):
        return {
            'user_ids': [1, 2, 3],
            'item_ids': [1, 2, 3],
            'ratings': [5.0, 4.0, 3.0]
        }
    
    def test_dcn_initialization(self):
        model = DCN(embedding_dim=64)
        assert model.embedding_dim == 64
    
    def test_dcn_fit(self, sample_data):
        model = DCN()
        model.fit(**sample_data)
        assert model.is_fitted
```

## 📚 Documentation

### Building Documentation

```bash
cd docs/
mkdocs serve
# Visit http://localhost:8000
```

### Adding Documentation

- Update `docs/` for user-facing documentation
- Ensure docstrings are comprehensive for API docs
- Add examples to `examples/` directory

## 🔄 Pull Request Process

### Before Submitting

1. ✅ Run all tests and ensure they pass
2. ✅ Run linters (black, flake8, mypy)
3. ✅ Update documentation if needed
4. ✅ Add tests for new features
5. ✅ Update CHANGELOG.md

### Submitting PR

1. Push to your fork
2. Create a pull request against `main` branch
3. Fill out the PR template:
   - Description of changes
   - Related issue number
   - Type of change (bug fix, feature, docs, etc.)
   - Checklist confirmation

4. Wait for review
5. Address feedback
6. Once approved, maintainers will merge

### PR Guidelines

- Keep PRs focused (one feature/fix per PR)
- Write clear commit messages
- Reference related issues
- Update documentation
- Add tests

### Commit Message Format

```
type(scope): Short description

Longer description if needed

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(engines): Add BERT-based content recommender

Implement a new content-based recommender using BERT embeddings
for better semantic understanding of item descriptions.

Fixes #456
```

## ❓ Questions?

- Open an issue with the `question` label
- Email: vishesh@corerec.tech
- Join our discussions on GitHub

## 🙏 Thank You!

Your contributions help make CoreRec better for everyone in the recommendation systems community!

---

**Happy Coding! 🚀**

