# ULTRATHINK Documentation

This folder contains user and developer documentation for the ULTRATHINK project.

Complete guides for training state-of-the-art language models.

## Getting Started (5 minutes)

Start here if you're new to ULTRATHINK:

1. **[Training Quickstart](TRAINING_QUICKSTART.md)** - Get started in 5 minutes
2. **[Training Small Models](training_small.md)** - Best practices for limited hardware
3. **[Google Colab](colab.md)** - Train with free GPU in your browser

## Training Guides

### Basic Training
- **[Training Small Models](training_small.md)** - Start with small datasets (recommended)
- **[Google Colab](colab.md)** - Train with free GPU in your browser
- **[Datasets](datasets.md)** - Using built-in, custom, and mixed datasets

### Advanced Training
- **[Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md)** - MoE, DRE, Constitutional AI
- **[DeepSpeed](training_deepspeed.md)** - ZeRO optimization for memory efficiency
- **[Distributed Training](training_distributed.md)** - Multi-GPU setup

## Reference

- **[Model Card](MODEL_CARD.md)** - Architecture specifications and limitations
- **[Benchmarks](BENCHMARKS.md)** - Performance metrics and results
- **[Framework Comparison](COMPARISON.md)** - vs GPT-NeoX, Megatron-LM, Axolotl
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Project Structure](PROJECT_STRUCTURE.md)** - Understanding the codebase
- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - Visual system diagrams
- **[FAQ](faq.md)** - Common questions and solutions

## Planning & Community

- **[Roadmap](ROADMAP.md)** - Future plans and features
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards
- **[Changelog](CHANGELOG.md)** - Version history

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
