# Contributing to ULTRATHINK

Thank you for your interest in contributing to ULTRATHINK! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to avoid duplicates
2. **Use the issue templates** when available
3. **Provide detailed information** including:
   - Python version and OS
   - PyTorch version
   - Full error traceback
   - Steps to reproduce
   - Expected vs actual behavior

### Suggesting Features

We welcome feature suggestions! Please:

1. **Check existing feature requests** first
2. **Describe the use case** and motivation
3. **Provide implementation ideas** if possible
4. **Consider backwards compatibility**

### Pull Requests

#### Before You Start

1. **Fork the repository** and create a feature branch
2. **Check existing PRs** to avoid duplicate work
3. **Discuss major changes** in an issue first

#### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ultrathink.git
cd ultrathink

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development tools
pip install pre-commit pytest pytest-cov black flake8 mypy

# Setup pre-commit hooks
pre-commit install
```

#### Code Standards

We maintain high code quality standards:

**Code Formatting**
```bash
# Format code with Black
black src/ tests/ scripts/

# Check formatting
black --check src/ tests/ scripts/
```

**Linting**
```bash
# Run flake8
flake8 src/ tests/ scripts/

# Run mypy for type checking
mypy src/
```

**Testing**
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run smoke test
python -m tests.smoke_test
```

#### Code Style Guidelines

1. **Follow PEP 8** with Black formatting
2. **Use type hints** for all functions and methods
3. **Write docstrings** for public APIs (Google style)
4. **Keep functions focused** and reasonably sized
5. **Use meaningful variable names**
6. **Add comments for complex logic**

Example:
```python
def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10
) -> Dict[str, float]:
    """Train a PyTorch model.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer for training
        device: Device to train on
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If epochs is not positive
    """
    if epochs <= 0:
        raise ValueError("Epochs must be positive")
        
    # Training implementation...
    return {"loss": final_loss, "accuracy": final_acc}
```

#### Commit Guidelines

**Commit Message Format**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add flash attention support

fix(training): resolve gradient accumulation bug

docs(readme): update installation instructions

test(models): add unit tests for MoE routing
```

#### Pull Request Process

1. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above

3. **Write or update tests** for your changes

4. **Update documentation** if needed

5. **Run the full test suite**
   ```bash
   python -m pytest tests/
   python -m tests.smoke_test
   ```

6. **Run pre-commit checks**
   ```bash
   pre-commit run --all-files
   ```

7. **Push to your fork** and create a pull request

8. **Fill out the PR template** completely

9. **Respond to review feedback** promptly

#### PR Review Process

- All PRs require at least one review
- Automated checks must pass
- Documentation must be updated for user-facing changes
- Breaking changes require discussion and migration guide

## 🏗️ Architecture Guidelines

### Adding New Models

When adding new model components:

1. **Follow the existing patterns** in `src/models/`
2. **Inherit from appropriate base classes**
3. **Add comprehensive docstrings**
4. **Include configuration classes**
5. **Add unit tests**
6. **Update integration tests**

### Adding New Training Features

For training enhancements:

1. **Consider backwards compatibility**
2. **Add configuration options**
3. **Include proper logging**
4. **Add evaluation metrics**
5. **Document hyperparameter effects**

### Adding New Data Loaders

For data pipeline additions:

1. **Support streaming when possible**
2. **Include quality filtering options**
3. **Add proper error handling**
4. **Support multiple formats**
5. **Include data validation**

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Use descriptive test names**
   ```python
   def test_transformer_block_forward_pass_with_attention_mask():
   ```

2. **Test edge cases and error conditions**
   ```python
   def test_model_raises_error_with_invalid_vocab_size():
   ```

3. **Use fixtures for common setup**
   ```python
   @pytest.fixture
   def small_model_config():
       return ModelConfig(vocab_size=1000, n_embd=128, n_layer=2)
   ```

4. **Mock external dependencies**
   ```python
   @patch('wandb.init')
   def test_training_without_wandb(mock_wandb):
   ```

### Performance Tests

Include performance benchmarks for:
- Model forward/backward pass timing
- Memory usage patterns
- Throughput measurements

## 📚 Documentation

### Code Documentation

- **Docstrings**: Use Google style for all public APIs
- **Type hints**: Required for all function signatures
- **Comments**: Explain complex algorithms and business logic

### User Documentation

- **README updates**: For user-facing changes
- **Configuration docs**: For new parameters
- **Examples**: Include usage examples
- **Migration guides**: For breaking changes

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backwards compatible)
- Patch: Bug fixes

### Release Checklist

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create release PR
6. Tag release after merge
7. Update GitHub release notes

## 🎯 Priority Areas

We're particularly interested in contributions to:

1. **Performance optimizations**
   - Memory efficiency improvements
   - Training speed optimizations
   - Inference acceleration

2. **New model architectures**
   - Novel attention mechanisms
   - Advanced MoE strategies
   - Multi-modal improvements

3. **Training improvements**
   - Better data loading
   - Advanced RLHF techniques
   - Distributed training optimizations

4. **Evaluation and benchmarking**
   - New benchmark integrations
   - Evaluation metrics
   - Analysis tools

5. **Documentation and examples**
   - Tutorial notebooks
   - Use case examples
   - API documentation

## ❓ Questions?

- **General questions**: Open a [Discussion](https://github.com/yourusername/ultrathink/discussions)
- **Bug reports**: Open an [Issue](https://github.com/yourusername/ultrathink/issues)
- **Feature requests**: Open an [Issue](https://github.com/yourusername/ultrathink/issues) with the feature template
- **Security issues**: Email security@ultrathink.ai

## 🙏 Recognition

Contributors will be:
- Listed in the README
- Mentioned in release notes
- Invited to the contributors team (for significant contributions)

Thank you for helping make ULTRATHINK better! 🚀
