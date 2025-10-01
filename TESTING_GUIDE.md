# ULTRATHINK Testing Guide

Comprehensive guide for testing the ULTRATHINK framework.

## Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Test Structure

```
tests/
├── unit/                     # Unit tests (fast, isolated)
│   ├── test_models/
│   │   ├── test_architecture.py
│   │   └── test_moe.py
│   ├── test_training/
│   │   └── test_optimizer.py
│   └── test_data/
│       └── test_datasets.py
├── integration/              # Integration tests (end-to-end)
│   └── test_forward_pass.py
├── conftest.py              # Shared fixtures
└── smoke_test.py            # Quick smoke test
```

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
pytest -m unit            # Unit tests only
pytest -m integration     # Integration tests only
pytest -m "not slow"      # Skip slow tests
pytest -m gpu             # GPU-only tests
```

### By Module
```bash
pytest tests/unit/test_models/
pytest tests/unit/test_models/test_architecture.py
pytest tests/unit/test_models/test_architecture.py::TestRMSNorm
```

### Specific Test
```bash
pytest tests/unit/test_models/test_architecture.py::TestRMSNorm::test_forward_pass -v
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto          # Use all CPU cores
pytest -n 4             # Use 4 workers
```

## Test Coverage

### Generate Coverage Report
```bash
# HTML report (recommended)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=src --cov-report=term-missing

# XML report (for CI/CD)
pytest --cov=src --cov-report=xml
```

### Coverage Goals
- **Overall**: 80%+ coverage
- **Critical paths**: 95%+ coverage
- **New code**: 90%+ coverage

### Current Coverage
- Architecture components: ~85%
- MoE system: ~75%
- Training utilities: ~70%
- Data loaders: ~60%

## Writing Tests

### Test Naming Convention
```python
# File: test_<module>.py
# Class: Test<ClassName>
# Method: test_<what_it_tests>

def test_rmsnorm_forward_pass():
    """Test RMSNorm forward pass produces correct output shape"""
    pass
```

### Using Fixtures
```python
def test_model_forward(small_model_config, sample_batch, device):
    """Test model forward pass with fixtures"""
    model = UltraThinkModel(small_model_config).to(device)
    output = model(**sample_batch)
    assert output['loss'] is not None
```

### Parametrized Tests
```python
@pytest.mark.parametrize("batch_size,seq_len", [
    (1, 128),
    (2, 256),
    (4, 512),
])
def test_various_shapes(batch_size, seq_len):
    """Test model with various input shapes"""
    pass
```

### Testing Exceptions
```python
def test_invalid_config():
    """Test that invalid config raises ValueError"""
    with pytest.raises(ValueError, match="Invalid configuration"):
        config = ModelConfig(n_embd=-1)
```

### Mocking External Dependencies
```python
from unittest.mock import patch, MagicMock

@patch('wandb.init')
def test_training_without_wandb(mock_wandb):
    """Test training works without W&B"""
    mock_wandb.return_value = MagicMock()
    # Test code here
```

## Common Test Patterns

### Model Tests
```python
def test_model_initialization():
    """Test model initializes correctly"""
    config = ModelConfig()
    model = UltraThinkModel(config)
    assert model is not None
    assert len(list(model.parameters())) > 0

def test_model_forward_pass():
    """Test model forward pass"""
    model = create_test_model()
    x = torch.randn(2, 16, 768)
    output = model(x)
    assert output.shape == (2, 16, 50257)  # vocab_size

def test_model_backward_pass():
    """Test gradients flow correctly"""
    model = create_test_model()
    x = torch.randn(2, 16, 768, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
```

### Training Tests
```python
def test_optimizer_step():
    """Test optimizer updates parameters"""
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters())
    
    param_before = model.weight.clone()
    
    x = torch.randn(4, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    param_after = model.weight
    assert not torch.equal(param_before, param_after)
```

### Data Tests
```python
def test_dataset_loading():
    """Test dataset loads correctly"""
    config = DatasetConfig(name="wikitext")
    dataset = load_dataset(config)
    assert len(dataset) > 0
    
    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
```

## Continuous Integration

### GitHub Actions
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Debugging Tests

### Run with Print Statements
```bash
pytest -s  # Don't capture stdout
```

### Run with Debugger
```bash
pytest --pdb  # Drop into debugger on failure
```

### Verbose Traceback
```bash
pytest --tb=long  # Full traceback
pytest --tb=short  # Short traceback
pytest --tb=line  # One line per failure
```

### Show Local Variables
```bash
pytest -l  # Show local variables in tracebacks
```

## Performance Testing

### Benchmark Tests
```python
import time

def test_model_performance(benchmark):
    """Benchmark model forward pass"""
    model = create_test_model()
    x = torch.randn(2, 512, 768)
    
    result = benchmark(model, x)
    # Asserts on timing
```

### Memory Profiling
```python
import tracemalloc

def test_memory_usage():
    """Test memory usage stays within limits"""
    tracemalloc.start()
    
    model = create_large_model()
    x = torch.randn(8, 2048, 1024)
    output = model(x)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 8 * 1024**3  # Less than 8GB
```

## Test Fixtures

### Available Fixtures
```python
# From conftest.py
device              # torch.device (cuda/cpu)
small_model_config  # Small ModelConfig for testing
tiny_ultrathink_config  # Tiny UltraThinkConfig
sample_batch        # Sample training batch
temp_dir           # Temporary directory
```

### Creating Custom Fixtures
```python
@pytest.fixture
def trained_model():
    """Fixture providing a pre-trained model"""
    model = load_model("path/to/checkpoint")
    return model

def test_inference(trained_model):
    """Test inference with trained model"""
    output = trained_model.generate(prompt)
    assert len(output) > 0
```

## Best Practices

### DO ✅
- Write tests for new features
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests fast and independent
- Mock expensive operations
- Use fixtures for common setup
- Test both success and failure paths

### DON'T ❌
- Write tests that depend on external services
- Use hardcoded paths or credentials
- Write tests that depend on execution order
- Test implementation details (test behavior)
- Commit failing tests
- Skip tests without good reason

## Troubleshooting

### Tests Pass Locally but Fail in CI
- Check Python version differences
- Check for filesystem path issues
- Look for timing/race conditions
- Verify all dependencies are installed

### Slow Tests
- Use pytest-xdist for parallelization
- Mock slow operations
- Use smaller test data
- Profile with `pytest --durations=10`

### Flaky Tests
- Add retries for network operations
- Fix race conditions
- Increase timeouts
- Make tests deterministic (set seeds)

### Import Errors
- Check PYTHONPATH
- Verify package installation
- Look for circular imports

## Next Steps

1. **Run tests**: `pytest -v`
2. **Check coverage**: `pytest --cov=src`
3. **Fix failing tests**: Debug and iterate
4. **Add new tests**: For uncovered code
5. **Integrate CI**: Set up automated testing

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
