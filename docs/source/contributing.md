# Contributing to CoreRec

Thank you for your interest in contributing to CoreRec! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Familiarity with recommendation systems (helpful but not required)

### Finding an Issue to Work On

1. Check the [Issues](https://github.com/vishesh9131/CoreRec/issues) page
2. Look for issues labeled `good first issue` or `help wanted`
3. Comment on the issue to let others know you're working on it

## Development Setup

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### Suggesting Features

1. Check if the feature has already been suggested
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Potential implementation approach

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Write tests** for new functionality
5. **Update documentation** if needed
6. **Run tests**: `pytest`
7. **Run linters**: `flake8`, `black --check`
8. **Commit changes**: `git commit -m "Add feature X"`
9. **Push to your fork**: `git push origin feature/my-feature`
10. **Create a Pull Request**

## Coding Standards

### Code Style

- Follow PEP 8 style guide
- Use `black` for code formatting
- Maximum line length: 100 characters
- Use type hints where possible

### Documentation

- Write docstrings for all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings when helpful

### Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Use `pytest` for testing

## Pull Request Process

1. **Update CHANGELOG.md** with your changes
2. **Ensure all tests pass**
3. **Update documentation** if needed
4. **Request review** from maintainers
5. **Address review comments**
6. **Wait for approval** before merging

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Contact the maintainers
- Check existing documentation

Thank you for contributing to CoreRec!

