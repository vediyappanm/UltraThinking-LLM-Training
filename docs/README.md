# ULTRATHINK Documentation

This folder contains user and developer documentation for the ULTRATHINK project.

Complete guides for training state-of-the-art language models.

## Getting Started (5 minutes)

Start here if you're new to ULTRATHINK:

1. **[Installation](../INSTALLATION_GUIDE.md)** - Set up your environment
2. **[Getting Started](getting_started.md)** - Your first training run
3. **[Training Small Models](training_small.md)** - Best practices for small datasets

## Training Guides

### Basic Training
- **[Training Small Models](training_small.md)** - Start with small datasets (recommended)
- **[Google Colab](colab.md)** - Train with free GPU in your browser
- **[Datasets](datasets.md)** - Using built-in, custom, and mixed datasets

### Advanced Training
- **[DeepSpeed](training_deepspeed.md)** - ZeRO optimization for memory efficiency
- **[Distributed Training](accelerate.md)** - Multi-GPU with Accelerate/DDP
- **[Advanced Features](training_full.md)** - 4D parallelism, RLHF, MoE

## Reference

- **[Model Card](../MODEL_CARD.md)** - Architecture specifications and limitations
- **[Testing Guide](../TESTING_GUIDE.md)** - Running and writing tests
- **[Development](development.md)** - Code structure and contributing
- **[Evaluation](evaluation.md)** - Benchmarking your models
- **[FAQ](faq.md)** - Common questions and solutions

## Monitoring & Tools

ULTRATHINK includes production-grade monitoring:

```python
from src.monitoring import MetricsLogger

# Track metrics
metrics = MetricsLogger(window_size=100)
metrics.log(loss, lr, model, batch_size, seq_length)
```

See [Testing Guide](../TESTING_GUIDE.md#monitoring--profiling) for details.

## Quick Reference

| Task | Command |
|------|---------|
| Train tiny model | `python train_ultrathink.py --hidden_size 256 --num_layers 2` |
| Profile model | `python scripts/profile_model.py --size tiny` |
| Run tests | `pytest` |
| Clean cache | `python scripts/cleanup.py` |

## Need Help?

- **Issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)
- **FAQ**: [faq.md](faq.md) — Frequently asked questions
