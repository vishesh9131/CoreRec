# Testing Guide

CoreRec includes a comprehensive test suite to ensure reliability and correctness of all algorithms and components.

## Test Organization

Tests live flat in `tests/`, one file per concern rather than one per engine:

```
tests/
├── test_all_production_models.py   # CI gate: import, fit, predict, recommend, save/load
├── test_model_contract.py          # every interaction model meets the same API
├── test_production_contract.py     # unified API, model by model
├── test_mf_models.py               # ALS, Item2Vec
├── test_classic_vae_graph_models.py # ItemKNN, UserKNN, EASE, SLIM, VAEs
├── test_hstu.py
├── test_nn.py                      # corerec.nn building blocks
├── test_onnx_export.py
├── test_safe_persistence.py        # safe bundle save/load
├── test_serving_smoke.py           # ModelServer
├── test_feedback_loop.py           # feedback, online metrics, drift
├── test_pipeline_integration.py    # retrieval -> ranking pipelines
├── test_docs_imports.py            # every import in docs/ resolves
├── test_scale.py                   # regressions that only show at 100k users
├── contentFilterEngine/
│   └── tfidf_recommender_test.py
└── conftest.py
```

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run what CI runs (skips the slow docs build)
pytest tests/ -m "not docs_build"
```

`pyproject.toml` turns coverage on for every run (`--cov=corerec`, fails
under 40%). Add `--no-cov` when you run a single file and don't want the
coverage gate to fail it.

### Run Specific Test Categories

```bash
# Production gate: every production model, end to end
pytest tests/test_all_production_models.py

# Shared API contract
pytest tests/test_model_contract.py

# Content filter tests
pytest tests/contentFilterEngine/

# Serving and integration
pytest tests/test_serving_smoke.py tests/test_pipeline_integration.py
```

### Run Individual Test Files

```bash
# Test one model across the contract
pytest tests/test_model_contract.py -k dcn --no-cov
pytest tests/test_model_contract.py -k deepfm --no-cov

# Only the docs build tests, or everything but them
pytest tests/ -m "docs_build"
pytest tests/ -m "not docs_build"
```

### Run Quick Smoke Tests

```bash
# Quick checks that everything imports and the public API is in place
pytest tests/test_imports.py tests/test_public_api.py --no-cov
pytest tests/test_serving_smoke.py --no-cov
```

## Test Types

### 1. Unit Tests

Test individual components and methods:

```python
# tests/test_als_example.py
import pytest
from corerec.engines import ALS

def test_als_initialization():
    """Test ALS model initialization"""
    model = ALS(factors=20, iterations=10)
    assert model.factors == 20
    assert model.iterations == 10

def test_als_fit():
    """Test ALS training"""
    model = ALS(factors=10, iterations=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    assert model.is_fitted

def test_als_predict():
    """Test ALS prediction"""
    model = ALS(factors=10, iterations=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    score = model.predict(user_id=1, item_id=1)
    assert isinstance(score, float)

def test_als_recommend():
    """Test ALS recommendations"""
    model = ALS(factors=10, iterations=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    recs = model.recommend(user_id=1, top_k=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
```

ALS is implicit-feedback: `predict()` is a preference score, not a rating
on the 1-5 scale, so don't assert a range on it.

### 2. Integration Tests

Test complete workflows:

```python
# tests/test_integration_example.py
import numpy as np
import pytest
from corerec.engines import DCN
from corerec.evaluation import evaluate

def test_complete_workflow():
    """Test complete train-evaluate workflow"""
    # Prepare data: 30 users, 60 items, 600 interactions
    rng = np.random.default_rng(0)
    user_ids = rng.integers(0, 30, 600).tolist()
    item_ids = rng.integers(0, 60, 600).tolist()
    ratings = [1.0] * len(user_ids)
    
    # Split data
    train_size = int(0.8 * len(user_ids))
    train = list(zip(user_ids, item_ids, ratings))[:train_size]
    test = list(zip(user_ids, item_ids, ratings))[train_size:]
    
    # Train model
    model = DCN(embedding_dim=16, num_cross_layers=2, epochs=5)
    model.fit(*map(list, zip(*train)))
    
    # Evaluate: ranking metrics through model.recommend(), seen items excluded
    metrics = evaluate(model, test, train_interactions=train, k=10)
    
    assert "NDCG@10" in metrics
    assert "Recall@10" in metrics
    assert 0.0 <= metrics["NDCG@10"] <= 1.0
    
    # Get recommendations
    recs = model.recommend(user_id=user_ids[0], top_k=10)
    assert len(recs) == 10
```

### 3. Smoke Tests

Quick sanity checks:

```python
# tests/test_smoke_example.py
"""
Smoke tests for deep learning models.
Quick checks to ensure models can be imported and run.
"""

def test_dcn_smoke():
    """Quick smoke test for DCN"""
    from corerec.engines import DCN
    
    model = DCN(embedding_dim=8, num_cross_layers=1, epochs=1)
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    
    model.fit(users, items, ratings)
    recs = model.recommend(1, top_k=3)
    assert len(recs) > 0

def test_deepfm_smoke():
    """Quick smoke test for DeepFM"""
    from corerec.engines import DeepFM
    
    model = DeepFM(embedding_dim=8, hidden_layers=[8], epochs=1)
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    
    model.fit(users, items, ratings)
    recs = model.recommend(1, top_k=3)
    assert len(recs) > 0
```

Let a failure raise. A `try/except` that prints and carries on makes the
test pass whatever happens.

### 4. Import Tests

Verify all imports work:

```python
# tests/test_import_example.py
"""Test imports for the classic collaborative filtering models"""

def test_als_import():
    from corerec.engines import ALS

def test_ease_import():
    from corerec.engines import EASE

def test_every_registered_model_imports():
    import corerec.engines as engines
    for name in engines.list_models():
        getattr(engines, name)
```

## Writing Tests

### Test Structure

Follow this template for new tests:

```python
import pytest
from corerec.engines.your_model import YourModel

class TestYourModel:
    """Test suite for YourModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data"""
        return {
            'user_ids': [1, 1, 2, 2, 3],
            'item_ids': [1, 2, 1, 3, 2],
            'ratings': [5.0, 4.0, 4.0, 5.0, 3.0]
        }
    
    def test_initialization(self):
        """Test model initialization"""
        model = YourModel(param1=10, param2=20)
        assert model.param1 == 10
        assert model.param2 == 20
    
    def test_fit(self, sample_data):
        """Test model training"""
        model = YourModel()
        model.fit(**sample_data)
        assert model.is_fitted
    
    def test_predict(self, sample_data):
        """Test prediction"""
        model = YourModel()
        model.fit(**sample_data)
        score = model.predict(user_id=1, item_id=1)
        assert isinstance(score, float)
    
    def test_recommend(self, sample_data):
        """Test recommendations"""
        model = YourModel()
        model.fit(**sample_data)
        recs = model.recommend(user_id=1, top_k=2)
        assert isinstance(recs, list)
        assert len(recs) <= 2
    
    def test_save_load(self, sample_data, tmp_path):
        """Test model persistence"""
        model = YourModel()
        model.fit(**sample_data)
        
        # Save
        save_path = tmp_path / "model.pkl"
        model.save(str(save_path))
        
        # Load
        loaded_model = YourModel.load(str(save_path))
        assert loaded_model.is_fitted
        
        # Test loaded model works
        recs = loaded_model.recommend(user_id=1, top_k=2)
        assert len(recs) <= 2
```

### Using Fixtures

```python
import pytest

@pytest.fixture
def sample_interactions():
    """Provide sample user-item interactions"""
    return {
        'user_ids': [i % 20 for i in range(100)],
        'item_ids': [i % 50 for i in range(100)],
        'ratings': [float(i % 5 + 1) for i in range(100)]
    }

@pytest.fixture
def trained_model(sample_interactions):
    """Provide a trained model"""
    from corerec.engines import DCN
    model = DCN(embedding_dim=16, epochs=5)
    model.fit(**sample_interactions)
    return model

def test_with_trained_model(trained_model):
    """Test using trained model fixture"""
    recs = trained_model.recommend(user_id=1, top_k=10)
    assert len(recs) == 10
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("factors,iterations", [
    (10, 5),
    (20, 10),
    (50, 20)
])
def test_als_with_different_params(factors, iterations):
    """Test ALS with different hyperparameters"""
    from corerec.engines import ALS
    
    model = ALS(factors=factors, iterations=iterations)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    assert model.is_fitted
```

## Test Coverage

Check test coverage:

```bash
# Generate coverage report
pytest tests/ --cov=corerec --cov-report=html

# View report
open htmlcov/index.html
```

## Continuous Integration

CoreRec uses GitHub Actions for CI. The model tests job in
`.github/workflows/ci.yml` looks like this (trimmed):

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install -e ".[dev,serving,onnx]"
    - name: Run tests
      run: |
        python -m pytest tests/ -v --tb=short --strict-markers \
          -m "not docs_build" --cov=corerec --cov-fail-under=40
```

## Test Examples

### Complete Test Example

```python
# tests/test_complete_example.py
import pytest
import numpy as np
from corerec.engines import DCN

class TestDCNComplete:
    """Complete test suite for DCN"""
    
    @pytest.fixture
    def model(self):
        return DCN(
            embedding_dim=16,
            num_cross_layers=2,
            deep_layers=[32, 16],
            epochs=5,
            batch_size=32
        )
    
    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        n = 100
        return {
            'user_ids': rng.integers(1, 20, n).tolist(),
            'item_ids': rng.integers(1, 50, n).tolist(),
            'ratings': rng.uniform(1, 5, n).tolist()
        }
    
    def test_initialization(self, model):
        assert model.embedding_dim == 16
        assert model.num_cross_layers == 2
        assert not model.is_fitted
    
    def test_fit(self, model, data):
        model.fit(**data)
        assert model.is_fitted
    
    def test_predict(self, model, data):
        model.fit(**data)
        score = model.predict(user_id=data['user_ids'][0], item_id=data['item_ids'][0])
        assert isinstance(score, (int, float))
    
    def test_recommend(self, model, data):
        model.fit(**data)
        recs = model.recommend(user_id=data['user_ids'][0], top_k=5)
        assert len(recs) <= 5
    
    def test_batch_predict(self, model, data):
        model.fit(**data)
        pairs = list(zip(data['user_ids'][:3], data['item_ids'][:3]))
        scores = model.batch_predict(pairs)
        assert len(scores) == 3
    
    def test_batch_recommend(self, model, data):
        model.fit(**data)
        users = sorted(set(data['user_ids']))[:3]
        recs = model.batch_recommend(users, top_k=5)
        assert len(recs) == 3
        for user_recs in recs.values():
            assert len(user_recs) <= 5
```

## Running the Test Suite

There is no separate test runner; use pytest directly:

```bash
# What CI runs
python -m pytest tests/ -m "not docs_build"

# Just the production gate, fastest useful check before a PR
python -m pytest tests/test_all_production_models.py tests/test_model_contract.py --no-cov
```

## Next Steps

