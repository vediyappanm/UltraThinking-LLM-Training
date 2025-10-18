# ULTRATHINK

<p align="center">
  <a href="https://ultrathinking-llm-training.netlify.app/" target="_blank">
    <strong>📖 Read the Complete ULTRATHINK Book (Interactive)</strong>
  </a>
  <br/>
  <em>Full project documentation, diagrams, and guides</em>
  <br/><br/>
</p>

<p align="center">
  <img src="docs/images/pp.jpg" alt="ULTRATHINK Logo" width="250" />
</p>

<p align="center">
  <strong>🚀 Production-ready training framework for advanced Large Language Models</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/actions">
    <img src="https://github.com/vediyappanm/UltraThinking-LLM-Training/workflows/CI/badge.svg" alt="CI Status"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/>
  </a>
  <a href="https://ultrathinking-llm-training.netlify.app/">
    <img src="https://img.shields.io/badge/Book-Read%20Online-00C7B7?logo=readthedocs&logoColor=white" alt="Read the Book"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/>
  </a>
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/stargazers">
    <img src="https://img.shields.io/github/stars/vediyappanm/UltraThinking-LLM-Training?style=social" alt="GitHub stars"/>
  </a>
</p>

<p align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  </a>
  <a href="https://huggingface.co/">
    <img src="https://huggingface.co/Vedisasi/UltraThinking-LLM-Training" alt="Hugging Face"/>
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  </a>
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/issues">
    <img src="https://img.shields.io/github/issues/vediyappanm/UltraThinking-LLM-Training" alt="Issues"/>
  </a>
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/pulls">
    <img src="https://img.shields.io/github/issues-pr/vediyappanm/UltraThinking-LLM-Training" alt="Pull Requests"/>
  </a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="https://ultrathinking-llm-training.netlify.app/">Book</a> •
  <a href="#-documentation">Documentation</a> •
  <a href="docs/BENCHMARKS.md">Benchmarks</a> •
  <a href="docs/COMPARISON.md">Comparisons</a> •
  <a href="docs/ROADMAP.md">Roadmap</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

ULTRATHINK provides a complete, modular stack for training custom LLMs with state-of-the-art architectures, distributed training, and comprehensive monitoring.

## 🎯 Why ULTRATHINK?

**Train state-of-the-art LLMs in 10 lines of code** - From prototype to production in minutes, not days.

```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --hidden_size 768 --num_layers 12 \
  --enable_moe --enable_dre \
  --use_amp --gradient_checkpointing
```

### 🏆 What Makes Us Different

| Feature | ULTRATHINK | Others |
|---------|-----------|--------|
| **Setup Time** | ⚡ 5 minutes | 30-120 minutes |
| **Lines to Train** | 📝 ~10 | 50-100+ |
| **MoE Support** | ✅ Native | ❌ or Limited |
| **Dynamic Reasoning** | ✅ Unique | ❌ None |
| **Constitutional AI** | ✅ Built-in | ❌ None |
| **Documentation** | 📚 Comprehensive | Varies |

**[See detailed comparison →](docs/COMPARISON.md)**

## ✨ Key Features

- 🏗️ **Modern Architecture** - GQA, RoPE, SwiGLU, Flash Attention, RMSNorm
- 🧠 **Advanced Components** - Mixture-of-Experts, Dynamic Reasoning Engine, Constitutional AI
- 📊 **Production Monitoring** - MLflow, W&B, TensorBoard integration
- ⚡ **Optimized Training** - DeepSpeed ZeRO, FSDP, gradient checkpointing, AMP
- 🧪 **Fully Tested** - Unit & integration tests with pytest
- 🐳 **Docker Support** - Ready-to-use containers for training and inference
- 📚 **Complete Docs** - Step-by-step guides for all experience levels

**[View benchmarks and performance metrics →](docs/BENCHMARKS.md)**

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training

# Install dependencies
pip install -r requirements.txt
```

### Training Examples

**Tiny Model (CPU-friendly, for testing):**
```bash
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 256 --num_layers 2 --num_heads 4 \
  --batch_size 2 --max_samples 1000 \
  --num_epochs 1
```

**Small Model (GPU recommended):**
```bash
python train_advanced.py --config configs/train_small.yaml
```

**With Advanced Features:**
```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --hidden_size 768 --num_layers 12 --num_heads 12 \
  --enable_moe --enable_dre --enable_constitutional \
  --use_amp --gradient_checkpointing \
  --use_mlflow
```

### Docker

```bash
# Run Gradio web interface
docker compose up

# Or build and run manually
docker build -t ultrathink:latest .
docker run -p 7860:7860 ultrathink:latest
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Quick smoke test
python tests/smoke_test.py
```

## 📚 Documentation

### 🚀 Getting Started
- **[Training Quickstart](docs/TRAINING_QUICKSTART.md)** - Get started in 5 minutes
- **[Advanced Training Guide](docs/ADVANCED_TRAINING_GUIDE.md)** - Deep dive into all features
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Google Colab](docs/colab.md)** - Train in the cloud for free

### 📊 Performance & Comparisons
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance metrics and results
- **[Framework Comparison](docs/COMPARISON.md)** - vs GPT-NeoX, Megatron-LM, Axolotl
- **[Model Card](docs/MODEL_CARD.md)** - Model specifications

### 🏗️ Architecture & Development
- **[Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)** - Visual system diagrams
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Understanding the codebase
- **[Roadmap](docs/ROADMAP.md)** - Future plans and features

### 📖 Training Guides
- [Small Models](docs/training_small.md) - Train on limited hardware
- [DeepSpeed Integration](docs/training_deepspeed.md) - Distributed training setup
- [Dataset Configuration](docs/datasets.md) - Using custom datasets

### 🤝 Community
- **[Contributing](docs/CONTRIBUTING.md)** - Contribution guidelines
- **[Code of Conduct](docs/CODE_OF_CONDUCT.md)** - Community standards
- **[Changelog](docs/CHANGELOG.md)** - Version history

**[📖 Full Documentation Index](docs/README.md)**

## 📁 Project Structure

```
UltraThinking-LLM-Training/
├── train_ultrathink.py        # Main training script
├── train_advanced.py          # YAML config-based training
├── app_gradio.py              # Web UI for inference
├── src/
│   ├── models/               # UltraThink, MoE, DRE, architecture
│   ├── data/                 # Datasets, tokenization, validation
│   ├── training/             # Optimizers, distributed, RLHF
│   ├── monitoring/           # Metrics and system monitoring
│   ├── security/             # Input validation and safety
│   └── evaluation/           # Benchmarks and metrics
├── tests/                    # Unit and integration tests
├── configs/                  # YAML configuration files
├── scripts/                  # Utilities (profiling, inference)
└── docs/                     # Documentation and guides
```

See **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** for detailed explanations.

## 🔥 Training Examples

### Small Dataset Training
```bash
# WikiText-2 (fast iteration)
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 512 --num_layers 6 --num_heads 8 \
  --batch_size 4 --num_epochs 3 \
  --use_mlflow
```

### Production Training (C4 Dataset)
```bash
# Streaming C4 with all optimizations
python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --hidden_size 768 --num_layers 12 --num_heads 12 \
  --batch_size 2 --gradient_accumulation_steps 64 \
  --learning_rate 3e-4 --warmup_steps 5000 \
  --use_amp --gradient_checkpointing \
  --max_seq_length 1024 \
  --output_dir ./outputs/c4_production
```

### Using Configuration Files
```bash
# Small model (4-8GB GPU)
python train_advanced.py --config configs/train_small.yaml

# Medium model (16-32GB GPU)
python train_advanced.py --config configs/train_medium.yaml

# Large model (40GB+ GPU)
python train_advanced.py --config configs/train_large.yaml
```

## 🐳 Docker Usage

**Web Interface (Gradio):**
```bash
docker compose up
# Visit http://localhost:7860
```

**Custom Training:**
```bash
docker run -v $(pwd)/outputs:/app/outputs ultrathink:latest \
  python train_ultrathink.py \
    --dataset wikitext \
    --hidden_size 256 --num_layers 2 \
    --output_dir /app/outputs/my_model
```

**GPU Training:**
```bash
docker run --gpus all \
  -v $(pwd)/outputs:/app/outputs \
  ultrathink:latest \
  python train_ultrathink.py --use_amp
```

## 🤝 Contributing

We welcome contributions! Please see:
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Guidelines and setup
- **[Code of Conduct](docs/CODE_OF_CONDUCT.md)** - Community standards
- **[Roadmap](docs/ROADMAP.md)** - See what we're building next

### 🌟 Star History

If you find ULTRATHINK useful, please consider giving us a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=vediyappanm/UltraThinking-LLM-Training&type=Date)](https://star-history.com/#vediyappanm/UltraThinking-LLM-Training&Date)

## 📊 Model Specifications

| Size | Parameters | Layers | Hidden | Context | Min GPU |
|------|-----------|--------|--------|---------|---------|
| Tiny | 125M | 12 | 768 | 2048 | 6GB |
| Small | 350M | 24 | 1024 | 4096 | 16GB |
| Medium | 760M | 24 | 1536 | 4096 | 24GB |
| Large | 1.3B | 32 | 2048 | 8192 | 40GB |

See **[MODEL_CARD.md](docs/MODEL_CARD.md)** for complete specifications.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Citation

If you use ULTRATHINK in your research or project, please cite:

```bibtex
@software{ultrathink2025,
  title={ULTRATHINK: Advanced LLM Training Framework with Mixture-of-Experts and Dynamic Reasoning},
  author={ULTRATHINK Team},
  year={2025},
  url={https://github.com/vediyappanm/UltraThinking-LLM-Training},
  version={1.0.0}
}
```

## 🌐 Community & Support

<p align="center">
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions">
    <img src="https://img.shields.io/badge/Discussions-Join%20Us-blue?logo=github" alt="Discussions"/>
  </a>
  <a href="https://github.com/vediyappanm/UltraThinking-LLM-Training/issues">
    <img src="https://img.shields.io/badge/Issues-Report%20Bug-red?logo=github" alt="Issues"/>
  </a>
  <a href="https://twitter.com/intent/tweet?text=Check%20out%20ULTRATHINK%20-%20Advanced%20LLM%20Training%20Framework&url=https://github.com/vediyappanm/UltraThinking-LLM-Training">
    <img src="https://img.shields.io/badge/Twitter-Share-1DA1F2?logo=twitter&logoColor=white" alt="Twitter"/>
  </a>
</p>

### 💬 Get Help
- **[GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)** - Ask questions, share ideas
- **[Issue Tracker](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)** - Report bugs, request features
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[FAQ](docs/faq.md)** - Frequently asked questions

### 🚀 Share Your Work
Built something cool with ULTRATHINK? We'd love to hear about it!
- Open a discussion to share your project
- Submit a PR to add your model to our showcase
- Tweet about it and tag us

### 📢 Stay Updated
- ⭐ **Star this repo** to get notifications
- 👀 **Watch releases** for new features
- 🐦 **Follow on Twitter** for updates

---

<p align="center">
  <strong>Made with ❤️ by the ULTRATHINK Team</strong>
</p>

<p align="center">
  <a href="#ultrathink">Back to Top ↑</a>
</p>