# 🚀 Complete Training Guide for UltraThinking LLM

This guide provides everything you need to train the UltraThinking model with all features and proper checkpointing.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Information](#dataset-information)
3. [Training Scripts](#training-scripts)
4. [All Available Flags](#all-available-flags)
5. [Checkpointing Guide](#checkpointing-guide)
6. [Monitoring Training](#monitoring-training)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

### Prerequisites

1. **Python 3.9+** installed
2. **Dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **GPU** (recommended) or CPU for training

### Fastest Way to Start

**Option 1: Easy Training (Recommended for beginners)**
```bash
# Windows
train_easy.bat

# Linux/Mac
python train_complete.py --dataset custom --data_path ./easy_dataset.json --num_epochs 3
```

**Option 2: Advanced Training (All features)**
```bash
# Windows
train_advanced.bat

# Linux/Mac - see the Advanced Training section below
```

**Option 3: Training with Checkpointing**
```bash
# Windows
train_with_checkpoints.bat
```

---

## 📚 Dataset Information

### Easy Dataset (`easy_dataset.json`)

- **Size**: 100 text examples
- **Content**: Simple, diverse sentences covering various topics
- **Purpose**: Perfect for testing and learning the training pipeline
- **Training Time**: 10-30 minutes on modest hardware

**Dataset Structure**:
```json
[
  {"text": "Your training text here..."},
  {"text": "Another training example..."}
]
```

### Using Your Own Dataset

1. **Create a JSON file** with the same structure as `easy_dataset.json`
2. **Update the data path**:
   ```bash
   python train_complete.py --data_path ./your_dataset.json
   ```

### Using HuggingFace Datasets

```bash
python train_complete.py \
    --dataset wikitext \
    --dataset_subset wikitext-103-v1 \
    --streaming
```

**Available datasets**: `wikitext`, `openwebtext`, `pile`, `c4`, `bookcorpus`

---

## 🎮 Training Scripts

### 1. Easy Training (`train_easy.bat`)

**Best for**: Testing, learning, modest hardware

**Configuration**:
- Hidden Size: 768
- Layers: 12
- Batch Size: 4
- Epochs: 3
- Features: Basic (no MoE, no DRE)

**Hardware Requirements**: 4-8GB GPU or CPU

**Run**:
```bash
train_easy.bat
```

---

### 2. Advanced Training (`train_advanced.bat`)

**Best for**: Full feature demonstration, powerful hardware

**Configuration**:
- Hidden Size: 1024
- Layers: 12
- Batch Size: 4
- Epochs: 5
- **Features Enabled**:
  - ✅ Mixture of Experts (MoE)
  - ✅ Dynamic Reasoning Engine (DRE)
  - ✅ Constitutional AI
  - ✅ Flash Attention
  - ✅ Gradient Checkpointing
  - ✅ Mixed Precision (AMP)

**Hardware Requirements**: 8GB+ GPU

**Run**:
```bash
train_advanced.bat
```

---

### 3. Checkpoint Training (`train_with_checkpoints.bat`)

**Best for**: Long training runs, resumable training

**Configuration**:
- Saves checkpoint every epoch
- Keeps last 5 checkpoints
- Automatic resumption support
- Full state preservation

**Run**:
```bash
train_with_checkpoints.bat
```

**Resume from checkpoint**:
```bash
python train_complete.py --resume_checkpoint ./outputs/checkpoint_training/checkpoint_epoch_5.pt
```

---

## 🎛️ All Available Flags

### Model Architecture

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab_size` | 50257 | Vocabulary size |
| `--hidden_size` | 768 | Hidden dimension |
| `--num_layers` | 12 | Number of transformer layers |
| `--num_heads` | 12 | Number of attention heads |
| `--num_kv_heads` | 12 | Key-value heads (GQA) |
| `--intermediate_size` | 3072 | FFN intermediate size |
| `--max_seq_length` | 1024 | Maximum sequence length |
| `--activation` | swiglu | Activation function |

### Mixture of Experts (MoE)

| Flag | Default | Description |
|------|---------|-------------|
| `--enable_moe` | False | Enable MoE architecture |
| `--num_knowledge_experts` | 8 | Number of knowledge experts |
| `--num_skill_experts` | 4 | Number of skill experts |
| `--num_meta_experts` | 2 | Number of meta experts |
| `--num_safety_experts` | 2 | Number of safety experts |
| `--moe_top_k` | 2 | Top-k experts to route to |
| `--expert_capacity` | 1.25 | Expert capacity factor |
| `--load_balance_weight` | 0.01 | Load balancing loss weight |
| `--z_loss_weight` | 0.001 | Router regularization weight |
| `--importance_weight` | 0.01 | Routing diversity weight |

### Advanced Features

| Flag | Default | Description |
|------|---------|-------------|
| `--enable_dre` | False | Enable Dynamic Reasoning Engine |
| `--enable_constitutional` | False | Enable Constitutional AI |
| `--enable_rlhf` | False | Enable RLHF training |
| `--enable_multimodal` | False | Enable multimodal (image/audio) |
| `--dre_warmup_steps` | 1000 | DRE warmup steps |

### Training Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 8 | Batch size per device |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--learning_rate` | 3e-4 | Peak learning rate |
| `--weight_decay` | 0.01 | Weight decay |
| `--warmup_steps` | 2000 | LR warmup steps |
| `--num_epochs` | 3 | Number of epochs |
| `--gradient_clipping` | 1.0 | Gradient clipping threshold |

### Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--use_flash_attention` | False | Use Flash Attention 2 |
| `--gradient_checkpointing` | False | Enable gradient checkpointing |
| `--use_amp` | False | Use mixed precision (AMP) |
| `--amp_warmup_steps` | 0 | AMP warmup steps |

### Dataset

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | custom | Dataset name |
| `--data_path` | ./easy_dataset.json | Path to dataset |
| `--text_column` | text | Text column name |
| `--tokenizer_name` | gpt2 | Tokenizer to use |
| `--max_samples` | None | Max samples to use |
| `--streaming` | False | Enable streaming mode |

### Checkpointing

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | ./outputs/complete_training | Output directory |
| `--resume_checkpoint` | None | Resume from checkpoint |
| `--save_checkpoint_every` | 1 | Save every N epochs |
| `--keep_last_n_checkpoints` | 3 | Keep last N checkpoints |

### Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--use_mlflow` | False | Enable MLflow tracking |
| `--use_wandb` | False | Enable W&B tracking |
| `--mlflow_experiment` | UltraThinking-Complete | Experiment name |
| `--run_name` | complete_training | Run name |

---

## 💾 Checkpointing Guide

### Automatic Checkpointing

Checkpoints are automatically saved every epoch (configurable with `--save_checkpoint_every`).

**Checkpoint Contents**:
- Model weights
- Optimizer state
- Scheduler state
- Training epoch
- Validation loss
- Full configuration

### Resume Training

**From latest checkpoint**:
```bash
python train_complete.py --resume_checkpoint ./outputs/checkpoint_training/checkpoint_epoch_5.pt
```

**Continue with different settings**:
```bash
python train_complete.py \
    --resume_checkpoint ./outputs/checkpoint_training/checkpoint_epoch_5.pt \
    --learning_rate 1e-4 \
    --num_epochs 10
```

### Checkpoint Management

**Keep last N checkpoints** (saves disk space):
```bash
python train_complete.py --keep_last_n_checkpoints 3
```

This automatically deletes older checkpoints, keeping only the 3 most recent.

### Checkpoint Location

Default: `./outputs/<experiment_name>/checkpoint_epoch_<N>.pt`

Example:
```
outputs/
└── checkpoint_training/
    ├── checkpoint_epoch_0.pt
    ├── checkpoint_epoch_1.pt
    ├── checkpoint_epoch_2.pt
    ├── final_model/          # HuggingFace format
    └── training.log
```

---

## 📊 Monitoring Training

### MLflow (Recommended)

**Start MLflow UI**:
```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open: http://localhost:5000

**What you can see**:
- Training/validation loss curves
- Learning rate schedule
- All hyperparameters
- Checkpoints and artifacts
- Compare multiple runs

### Logs

**View training log**:
```bash
# Windows
type outputs\checkpoint_training\training.log

# Linux/Mac
tail -f outputs/checkpoint_training/training.log
```

### Real-time Progress

Training shows real-time progress with:
- Current loss
- Average loss
- Epoch progress
- Estimated time remaining

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. **Reduce batch size**: `--batch_size 2`
2. **Increase gradient accumulation**: `--gradient_accumulation_steps 8`
3. **Enable gradient checkpointing**: `--gradient_checkpointing`
4. **Reduce model size**: `--hidden_size 512 --num_layers 6`
5. **Reduce sequence length**: `--max_seq_length 512`

### Slow Training

**Solutions**:
1. **Enable AMP**: `--use_amp`
2. **Enable Flash Attention**: `--use_flash_attention`
3. **Increase batch size**: `--batch_size 16`
4. **Reduce gradient accumulation**: `--gradient_accumulation_steps 2`
5. **Use more workers**: `--num_workers 4`

### NaN Loss

**Solutions**:
1. **Reduce learning rate**: `--learning_rate 1e-4`
2. **Increase warmup**: `--warmup_steps 5000`
3. **Enable gradient clipping**: `--gradient_clipping 1.0`
4. **Disable AMP initially**: Remove `--use_amp` or use `--amp_warmup_steps 1000`
5. **Disable DRE initially**: Remove `--enable_dre` or use `--dre_warmup_steps 2000`

### Dataset Issues

**Solutions**:
1. **Check file exists**: Verify `easy_dataset.json` is present
2. **Check JSON format**: Ensure valid JSON with `[{"text": "..."}]` structure
3. **Use dummy dataset**: `--dataset dummy --train_samples 1000`
4. **Check text column**: `--text_column your_column_name`

### Import Errors

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install specific packages
pip install torch transformers datasets accelerate mlflow
```

---

## 🎓 Example Commands

### Minimal Training (CPU-friendly)
```bash
python train_complete.py \
    --dataset custom \
    --data_path ./easy_dataset.json \
    --hidden_size 256 \
    --num_layers 4 \
    --num_heads 4 \
    --batch_size 2 \
    --num_epochs 2 \
    --output_dir ./outputs/minimal
```

### Production Training
```bash
python train_complete.py \
    --dataset custom \
    --data_path ./easy_dataset.json \
    --hidden_size 2048 \
    --num_layers 24 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --num_epochs 10 \
    --use_amp \
    --use_flash_attention \
    --gradient_checkpointing \
    --enable_moe \
    --enable_dre \
    --use_mlflow \
    --output_dir ./outputs/production
```

### Resume and Continue
```bash
python train_complete.py \
    --resume_checkpoint ./outputs/production/checkpoint_epoch_5.pt \
    --num_epochs 20 \
    --learning_rate 1e-4
```

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)
- **Documentation**: [Full Docs](https://ultrathinking-llm-training.netlify.app/)

---

## ✅ Training Checklist

Before starting training, ensure:

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset file exists (`easy_dataset.json`)
- [ ] Output directory is writable
- [ ] GPU drivers installed (if using GPU)
- [ ] Sufficient disk space for checkpoints
- [ ] MLflow installed (if using `--use_mlflow`)

---

**Happy Training! 🚀**
