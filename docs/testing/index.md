# Testing Guide

CoreRec includes a comprehensive test suite to ensure reliability and correctness of all algorithms and components.

## Test Organization

CoreRec's tests are organized by engine and component:

```
tests/
├── unionizedFilterEngine/          # Collaborative filtering tests
│   ├── algorithms_smoke_test.py    # Quick smoke tests
│   ├── mf_base_import_test.py      # Matrix factorization tests
│   ├── nn_base_import_test.py      # Neural network tests
│   ├── graph_based_base_import_test.py
│   ├── attention_mechanism_base_import_test.py
│   ├── bayesian_method_base_import_test.py
│   ├── sequential_model_base_import_test.py
│   └── variational_encoder_base_import_test.py
│
├── contentFilterEngine/            # Content-based filtering tests
│   ├── all_algorithms_test.py
│   ├── context_personalization_tests.py
│   ├── embedding_rep_learning_tests.py
│   ├── fairness_explainability_tests.py
│   ├── graph_based_algorithms_tests.py
│   ├── hybrid_ensemble_methods_tests.py
│   ├── learning_paradigms_tests.py
│   ├── nn_based_algorithms_tests.py
│   ├── probabilistic_statistical_methods_tests.py
│   ├── special_techniques_tests.py
│   └── traditional_ml_algorithms_tests.py
│
├── engines_models_smoke_test.py    # Deep learning models tests
├── test_integration.py             # Integration tests
└── test_*.py                       # Individual model tests
```

## Running Tests

### Run All Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=corerec --cov-report=html
```

### Run Specific Test Categories

```bash
# Run unionized filter tests
pytest tests/unionizedFilterEngine/

# Run content filter tests
pytest tests/contentFilterEngine/

# Run engine smoke tests
pytest tests/engines_models_smoke_test.py

# Run integration tests
pytest tests/test_integration.py
```

### Run Individual Test Files

```bash
# Test specific algorithm
pytest tests/test_DCN_base.py
pytest tests/test_DeepFM_base.py
pytest tests/test_GNN_base.py

# Test with specific markers
pytest tests/ -m "slow"
pytest tests/ -m "not slow"
```

### Run Quick Smoke Tests

```bash
# Quick smoke tests for all algorithms
python tests/unionizedFilterEngine/algorithms_smoke_test.py
python tests/contentFilterEngine/all_algorithms_test.py
python tests/engines_models_smoke_test.py
```

## Test Types

### 1. Unit Tests

Test individual components and methods:

```python
# tests/test_SVD_base.py
import pytest
from corerec.engines.unionizedFilterEngine.mf_base.SVD_base import SVD

def test_svd_initialization():
    """Test SVD model initialization"""
    model = SVD(n_factors=20, n_epochs=10)
    assert model.n_factors == 20
    assert model.n_epochs == 10

def test_svd_fit():
    """Test SVD training"""
    model = SVD(n_factors=10, n_epochs=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    assert model.is_fitted

def test_svd_predict():
    """Test SVD prediction"""
    model = SVD(n_factors=10, n_epochs=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    score = model.predict(user_id=1, item_id=1)
    assert isinstance(score, float)
    assert 0 <= score <= 5

def test_svd_recommend():
    """Test SVD recommendations"""
    model = SVD(n_factors=10, n_epochs=5)
    user_ids = [1, 1, 2, 2, 3]
    item_ids = [1, 2, 1, 3, 2]
    ratings = [5.0, 4.0, 4.0, 5.0, 3.0]
    
    model.fit(user_ids, item_ids, ratings)
    recs = model.recommend(user_id=1, top_k=2)
    assert isinstance(recs, list)
    assert len(recs) <= 2
```

### 2. Integration Tests

Test complete workflows:

```python
# tests/test_integration.py
import pytest
from corerec.engines.dcn import DCN
from corerec.evaluation import evaluate_model

def test_complete_workflow():
    """Test complete train-evaluate workflow"""
    # Prepare data
    user_ids = list(range(1, 100))
    item_ids = list(range(1, 50))
    ratings = [float(i % 5 + 1) for i in range(len(user_ids))]
    
    # Split data
    train_size = int(0.8 * len(user_ids))
    train_users = user_ids[:train_size]
    train_items = item_ids[:train_size]
    train_ratings = ratings[:train_size]
    
    test_users = user_ids[train_size:]
    test_items = item_ids[train_size:]
    test_ratings = ratings[train_size:]
    
    # Train model
    model = DCN(embedding_dim=16, num_cross_layers=2, epochs=5)
    model.fit(train_users, train_items, train_ratings)
    
    # Evaluate
    metrics = evaluate_model(
        model, test_users, test_items, test_ratings,
        metrics=['rmse', 'mae']
    )
    
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert metrics['rmse'] > 0
    
    # Get recommendations
    recs = model.recommend(user_id=1, top_k=10)
    assert len(recs) == 10
```

### 3. Smoke Tests

Quick sanity checks:

```python
# tests/engines_models_smoke_test.py
"""
Smoke tests for all deep learning models.
Quick checks to ensure models can be imported and run.
"""

def test_dcn_smoke():
    """Quick smoke test for DCN"""
    from corerec.engines.dcn import DCN
    
    model = DCN(embedding_dim=8, num_cross_layers=1, epochs=1)
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    
    try:
        model.fit(users, items, ratings)
        recs = model.recommend(1, top_n=3)
        assert len(recs) > 0
        print("✓ DCN smoke test passed")
    except Exception as e:
        print(f"✗ DCN smoke test failed: {e}")

def test_deepfm_smoke():
    """Quick smoke test for DeepFM"""
    from corerec.engines.deepfm import DeepFM
    
    model = DeepFM(embedding_dim=8, hidden_layers=[8], epochs=1)
    users = [1, 2, 1, 3]
    items = [10, 10, 20, 30]
    ratings = [1, 0, 1, 0]
    
    try:
        model.fit(users, items, ratings)
        recs = model.recommend(1, top_n=3)
        assert len(recs) > 0
        print("✓ DeepFM smoke test passed")
    except Exception as e:
        print(f"✗ DeepFM smoke test failed: {e}")
```

### 4. Import Tests

Verify all imports work:

```python
# tests/unionizedFilterEngine/mf_base_import_test.py
"""Test imports for matrix factorization algorithms"""

def test_svd_import():
    try:
        from corerec.engines.unionizedFilterEngine.mf_base.SVD_base import SVD
        print("✓ SVD import successful")
    except ImportError as e:
        print(f"✗ SVD import failed: {e}")

def test_als_import():
    try:
        from corerec.engines.unionizedFilterEngine.mf_base.ALS_base import ALS
        print("✓ ALS import successful")
    except ImportError as e:
        print(f"✗ ALS import failed: {e}")

def test_nmf_import():
    try:
        from corerec.engines.unionizedFilterEngine.mf_base.nmf_base import NMF
        print("✓ NMF import successful")
    except ImportError as e:
        print(f"✗ NMF import failed: {e}")
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
        'user_ids': list(range(1, 101)),
        'item_ids': list(range(1, 51)),
        'ratings': [float(i % 5 + 1) for i in range(100)]
    }

@pytest.fixture
def trained_model(sample_interactions):
    """Provide a trained model"""
    from corerec.engines.dcn import DCN
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

@pytest.mark.parametrize("n_factors,n_epochs", [
    (10, 5),
    (20, 10),
    (50, 20)
])
def test_svd_with_different_params(n_factors, n_epochs):
    """Test SVD with different hyperparameters"""
    from corerec.engines.unionizedFilterEngine.mf_base.SVD_base import SVD
    
    model = SVD(n_factors=n_factors, n_epochs=n_epochs)
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

CoreRec uses GitHub Actions for CI:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    - name: Run tests
      run: pytest tests/ --cov=corerec
```

## Test Examples

### Complete Test Example

```python
# tests/test_complete_example.py
import pytest
import numpy as np
from corerec.engines.dcn import DCN

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
        np.random.seed(42)
        n = 100
        return {
            'user_ids': np.random.randint(1, 20, n).tolist(),
            'item_ids': np.random.randint(1, 50, n).tolist(),
            'ratings': np.random.uniform(1, 5, n).tolist()
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
        score = model.predict(user_id=1, item_id=1)
        assert isinstance(score, (int, float))
    
    def test_recommend(self, model, data):
        model.fit(**data)
        recs = model.recommend(user_id=1, top_k=5)
        assert len(recs) <= 5
    
    def test_batch_predict(self, model, data):
        model.fit(**data)
        pairs = [(1, 1), (2, 2), (3, 3)]
        scores = model.batch_predict(pairs)
        assert len(scores) == 3
    
    def test_batch_recommend(self, model, data):
        model.fit(**data)
        users = [1, 2, 3]
        recs = model.batch_recommend(users, top_k=5)
        assert len(recs) == 3
        for user_recs in recs.values():
            assert len(user_recs) <= 5
```

## Running the Test Suite

Use the provided test runner:

```bash
# Run all algorithm tests
python examples/run_all_algo_tests_example.py

# Custom test runner
python corerec/run_algo_tests.py
```

## Next Steps

- Learn about [Unit Tests](unit-tests.md) in detail
- Explore [Integration Tests](integration-tests.md)
- Understand [Smoke Tests](smoke-tests.md)
- See [Running Tests](running-tests.md) for advanced options
- Check [Contributing Guidelines](../contributing/testing-guidelines.md) for test requirements


