# Development Guide

## Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Code style & checks
- Format: Black
- Lint: flake8
- Types: mypy
- Tests: pytest

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
pytest -q
```

## Project layout
- `train_ultrathink.py` — entry point for training
- `src/models/` — model architecture and components
- `src/data/` — datasets and loaders
- `src/training/` — distributed, RLHF utilities
- `src/evaluation/` — benchmarks

## Making changes
1. Create a branch
2. Add tests for new behavior
3. Update docs if user-facing changes
4. Run checks before opening a PR
