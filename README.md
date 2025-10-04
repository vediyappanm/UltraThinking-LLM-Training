# ULTRATHINK

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/colab.ipynb)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![ULTRATHINK – Open-Source LLM Training Pipeline](docs/images/ultrathink_cover.jpg)

**Production-ready training framework for advanced Large Language Models**

ULTRATHINK provides a complete, modular stack for training custom LLMs with state-of-the-art architectures, distributed training, and comprehensive monitoring.

## 🎬 Demo Video

[Watch ULTRATHINK in action](link-to-youtube)

## 🌐 Community

- **Discord**: [Join the ULTRATHINK community](https://discord.gg/ek2x9Rmk)
- **X (Twitter)**: [UltraThinkingLLM (@UltraThinkLLM)](https://x.com/UltraThinkLLM)

## 📊 Training Results

![Loss Curves](docs/images/training_curves.png)
![Expert Routing](docs/images/routing_distribution.png)
![Performance Comparison](docs/images/benchmark_comparison.png)

## 🏆 Benchmarks

| Model             | MMLU   | HellaSwag | TruthfulQA |
|------------------|--------|-----------|------------|
| ULTRATHINK-350M  | 45.2%  | 67.8%     | 52.3%      |
| GPT-2-345M       | 39.5%  | 61.2%     | 48.7%      |


## ✨ Key Features

- 🏗️ **Modern Architecture** - GQA, RoPE, SwiGLU, Flash Attention, RMSNorm
- 🧠 **Advanced Components** - Mixture-of-Experts, Dynamic Reasoning Engine, Constitutional AI
- 📊 **Production Monitoring** - MLflow, W&B, TensorBoard integration
- ⚡ **Optimized Training** - DeepSpeed ZeRO, FSDP, gradient checkpointing, AMP
- 🧪 **Fully Tested** - Unit & integration tests with pytest
- 🐳 **Docker Support** - Ready-to-use containers for training and inference
- 📚 **Complete Docs** - Step-by-step guides for all experience levels

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training/deep

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

### Essential Guides
- **[Training Quickstart](TRAINING_QUICKSTART.md)** - Get started in 5 minutes
- **[Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md)** - Deep dive into all features
- **[Project Structure](PROJECT_STRUCTURE.md)** - Understanding the codebase
- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)** - Visual system diagrams
- **[Model Card](MODEL_CARD.md)** - Model specifications and benchmarks

### Training Guides
- [Small Models](docs/training_small.md) - Train on limited hardware
- [DeepSpeed Integration](docs/training_deepspeed.md) - Distributed training setup
- [Dataset Configuration](docs/datasets.md) - Using custom datasets
- [Google Colab](docs/colab.md) - Train in the cloud for free

### Community
- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards
- **[Changelog](CHANGELOG.md)** - Version history

**[📖 Full Documentation Index](docs/README.md)**

## 📁 Project Structure

```
deep/
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

See **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** for detailed explanations.

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
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines and setup
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community standards

## 📊 Model Specifications

| Size | Parameters | Layers | Hidden | Context | Min GPU |
|------|-----------|--------|--------|---------|---------|
| Tiny | 125M | 12 | 768 | 2048 | 6GB |
| Small | 350M | 24 | 1024 | 4096 | 16GB |
| Medium | 760M | 24 | 1536 | 4096 | 24GB |
| Large | 1.3B | 32 | 2048 | 8192 | 40GB |

See **[MODEL_CARD.md](MODEL_CARD.md)** for complete specifications.

## 📈 Results: Expert-Forced Sanity Check (2025-10-04)

This run validates that the Mixture of Experts (MoE) is activating when the Dynamic Reasoning Engine (DRE) routes to the EXPERT path. It also confirms detailed MoE and DRE metrics logging during training.

  - Model: `hidden_size=512`, `num_layers=6`, `num_heads=8`, `intermediate=2048`, `max_seq_length=256`
  - Dataset: `C4` (subset `en`), streaming
  - MoE: Enabled, 3 MoE layers (`moe_layers=[1,3,5]`), `top_k=1`
  - DRE: Enabled, forced path=`expert`
  - Precision/Perf: AMP + gradient checkpointing

- **Highlights (first logged optimizer step)**
  - Train loss: `11.0715` (ppl `~64,310`)
  - Validation snapshots: `val_loss ≈ 11.03 – 11.08`
  - MoE: `used_moe=True`, `aux_total=13.0268`
    - Load balance: `0.0024`
    - z-loss: `6.1708`
    - Importance: `0.0028`
    - Entropy reg: `2.0303`
  - DRE: `path=expert`, complexity ranged `~0.15 – 0.47`
  - Latency per routing: `~0.62 – 1.04s`

### Curves and Visuals

- **Interactive Curves (MLflow)**
  - Start UI:
    ```bash
    mlflow ui --backend-store-uri file:./mlruns
    ```
  - Experiment: `UltraThinking-LLM-Training`
  - Run name: `expert_forced_colab_check`
  - Useful metrics: `train/step_loss`, `train/step_perplexity`, `moe/*`, `dre/*`

- **Routing Path Distribution** (forced Expert)

![Routing Path Distribution Dashboard](docs/images/routing_dashboard.png)


### Reproduce This Run

```bash
python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --hidden_size 512 --num_layers 6 --num_heads 8 --num_kv_heads 4 \
  --intermediate_size 2048 --max_seq_length 256 --activation swiglu \
  --enable_moe --num_knowledge_experts 4 --num_skill_experts 2 --num_meta_experts 1 --num_safety_experts 1 \
  --moe_top_k 1 --expert_capacity 1.25 --load_balance_weight 0.01 --z_loss_weight 0.001 --importance_weight 0.01 \
  --enable_dre --dre_warmup_steps 0 --dre_force_path expert \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 3e-4 --weight_decay 0.01 --warmup_steps 500 \
  --num_epochs 1 --max_steps 1000000 \
  --gradient_clipping 1.0 --dropout 0.0 --attention_dropout 0.0 \
  --use_amp --amp_warmup_steps 0 --gradient_checkpointing \
  --use_mlflow --mlflow_tracking_uri file:./mlruns --mlflow_experiment UltraThinking-LLM-Training \
  --run_name expert_forced_colab_check \
  --output_dir ./outputs/expert_forced_colab_check
```

Notes:
- MoE layers are automatically aligned to depth via `UltraThinkConfig.moe_layers` (`[1,3,5]` for 6 layers).
- Detailed per-step MoE metrics and `used_moe` flag are printed in the training log and logged to MLflow.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Citation

```bibtex
@software{ultrathink2025,
  title={ULTRATHINK: Advanced LLM Training Framework},
  author={ULTRATHINK Team},
  year={2025},
  url={https://github.com/vediyappanm/UltraThinking-LLM-Training}
}
```
