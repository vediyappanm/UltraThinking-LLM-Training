# 🚀 Training Quickstart Guide

Get started training your first ULTRATHINK model in **5 minutes**!

## Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB+ recommended)
- GPU with 6GB+ VRAM (optional but recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training

# Install dependencies
pip install -r requirements.txt
```

## Your First Training Run

### Option 1: Tiny Model (CPU-Friendly)

Perfect for testing the pipeline on any machine:

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 256 \
  --num_layers 2 \
  --num_heads 4 \
  --batch_size 2 \
  --max_samples 1000 \
  --num_epochs 1 \
  --output_dir ./outputs/my_first_model
```

**Expected time**: 5-10 minutes on CPU, 1-2 minutes on GPU

### Option 2: Small Model (GPU Recommended)

For actual model training with better quality:

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 512 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 4 \
  --num_epochs 3 \
  --use_amp \
  --output_dir ./outputs/small_model
```

**Expected time**: 30-60 minutes on RTX 3090

### Option 3: Using Config Files

The easiest way for reproducible training:

```bash
python train_advanced.py --config configs/train_small.yaml
```

## Understanding the Output

During training, you'll see:

```
Epoch 1/3 | Step 100/500 | Loss: 4.23 | LR: 0.0003 | Tokens/sec: 12500
Epoch 1/3 | Step 200/500 | Loss: 3.87 | LR: 0.0003 | Tokens/sec: 12800
...
```

**Key metrics**:
- **Loss**: Should decrease over time (good training)
- **LR**: Learning rate (changes with scheduler)
- **Tokens/sec**: Training speed

## Monitoring Training

### MLflow (Recommended)

```bash
# Start training with MLflow
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 256 --num_layers 2 \
  --use_mlflow

# View results
mlflow ui
# Open http://localhost:5000
```

### Weights & Biases

```bash
# Login first
wandb login

# Train with W&B
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 256 --num_layers 2 \
  --use_wandb
```

## Testing Your Model

After training completes, test text generation:

```bash
python scripts/inference.py \
  --model_path ./outputs/my_first_model \
  --prompt "Once upon a time" \
  --max_length 100
```

## Common Issues

### Out of Memory (OOM)

**Solution**: Reduce batch size or enable gradient checkpointing

```bash
python train_ultrathink.py \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing
```

### Slow Training

**Solution**: Enable mixed precision and optimize workers

```bash
python train_ultrathink.py \
  --use_amp \
  --num_workers 4
```

### Import Errors

**Solution**: Reinstall dependencies

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

### 📚 Learn More
- **[Advanced Training Guide](../ADVANCED_TRAINING_GUIDE.md)** - MoE, DRE, Constitutional AI
- **[Datasets Guide](datasets.md)** - Using custom datasets
- **[Troubleshooting](TROUBLESHOOTING.md)** - Detailed problem solving

### 🎯 Try Advanced Features

**Enable Mixture-of-Experts**:
```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --hidden_size 768 --num_layers 12 \
  --enable_moe \
  --num_experts 8 --top_k_experts 2
```

**Enable Dynamic Reasoning Engine**:
```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --enable_dre \
  --dre_paths 5
```

**Enable Constitutional AI**:
```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --enable_constitutional \
  --safety_threshold 0.7
```

### 🚀 Production Training

Ready for serious training? See:
- **[Production Training Guide](training_production.md)**
- **[DeepSpeed Integration](training_deepspeed.md)**
- **[Multi-GPU Setup](training_distributed.md)**

## Quick Reference

| Task | Command |
|------|---------|
| **Tiny model (test)** | `python train_ultrathink.py --hidden_size 256 --num_layers 2 --max_samples 1000` |
| **Small model** | `python train_advanced.py --config configs/train_small.yaml` |
| **With MLflow** | Add `--use_mlflow` to any command |
| **With W&B** | Add `--use_wandb` to any command |
| **Resume training** | Add `--resume_from_checkpoint ./outputs/my_model` |
| **Evaluate model** | `python scripts/evaluate.py --model_path ./outputs/my_model` |

## Getting Help

- **[FAQ](faq.md)** - Frequently asked questions
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
- **[GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)** - Report bugs
- **[Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)** - Ask questions

---

**Ready to train?** Pick an option above and start your first model! 🚀
