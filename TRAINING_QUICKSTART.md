# ULTRATHINK Training Quick Start Guide

Get up and running with ULTRATHINK model training in minutes.

## 🚀 5-Minute Quick Start

### Prerequisites
```bash
# Python 3.8+, CUDA 11.8+ (for GPU)
python --version
nvidia-smi  # Check GPU
```

### Installation
```bash
# Clone repository
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training

# Install dependencies
pip install -r requirements.txt

# Install MLflow (experiment tracking)
pip install mlflow
```

### Your First Training Run

#### Option A: Using Helper Script (Recommended)
```bash
# Windows
scripts\run_training.bat --profile small

# Linux/Mac
bash scripts/run_training.sh --profile small
```

#### Option B: Using Config File
```bash
python train_advanced.py --config configs/train_small.yaml
```

#### Option C: Direct Command
```bash
python train_ultrathink.py \
  --use_mlflow \
  --dataset dummy \
  --hidden_size 256 \
  --num_layers 2 \
  --num_heads 4 \
  --batch_size 2 \
  --train_samples 100 \
  --val_samples 20 \
  --num_epochs 1
```

### Monitor Training
```bash
# Start MLflow UI (in separate terminal)
mlflow ui

# Open browser: http://localhost:5000
```

## 📊 Training Profiles

Choose based on your hardware:

### Small (Testing/Development)
- **GPU**: 4-8GB VRAM (GTX 1080 Ti, RTX 2060)
- **Training Time**: ~2-4 hours
- **Use Case**: Testing, debugging, prototyping

```bash
python train_advanced.py --config configs/train_small.yaml
```

### Medium (Production)
- **GPU**: 16-32GB VRAM (RTX 3090, V100)
- **Training Time**: ~1-2 days
- **Use Case**: Production models, research

```bash
python train_advanced.py --config configs/train_medium.yaml
```

### Large (State-of-the-Art)
- **GPU**: 40-80GB VRAM (A100, H100)
- **Training Time**: ~1-2 weeks
- **Use Case**: Frontier models, large-scale training

```bash
python train_advanced.py --config configs/train_large.yaml
```

## 🌐 Google Colab Training

### Setup
1. Open [colab_training.ipynb](colab_training.ipynb) in Google Colab
2. Runtime → Change runtime type → GPU (T4/V100/A100)
3. Run setup cells
4. Choose a training configuration and run

### Quick Test (Colab)
```python
# In Colab notebook
!python train_advanced.py \
  --config configs/train_small.yaml \
  --override \
    training.batch_size=2 \
    data.train_samples=1000 \
    output.output_dir=/content/drive/MyDrive/ultrathink
```

## 🎯 Common Training Commands

### Basic Training
```bash
# Small model, quick test
python train_advanced.py --config configs/train_small.yaml

# With custom run name
python train_advanced.py \
  --config configs/train_small.yaml \
  --run-name "my_experiment_v1"
```

### Advanced Features Enabled
```bash
# Medium model with MoE and DRE
python train_advanced.py \
  --config configs/train_medium.yaml \
  --override \
    advanced.enable_moe=true \
    advanced.enable_dre=true \
    advanced.enable_constitutional=true
```

### Resume Training
```bash
# Resume from checkpoint
python train_advanced.py \
  --config configs/train_medium.yaml \
  --resume ./outputs/medium_model/checkpoint_epoch_5.pt
```

### Fine-tuning with RLHF
```bash
# Load pretrained model and fine-tune
python train_advanced.py \
  --config configs/train_medium.yaml \
  --init-from ./outputs/medium_model/final_model \
  --override \
    advanced.enable_rlhf=true \
    training.learning_rate=1e-5 \
    training.num_epochs=1
```

### Custom Overrides
```bash
# Override any config value
python train_advanced.py \
  --config configs/train_small.yaml \
  --override \
    training.batch_size=8 \
    model.hidden_size=1024 \
    data.dataset=wikitext \
    training.num_epochs=5
```

## 📁 File Structure

```
UltraThinking-LLM-Training/
├── configs/                    # Training configurations
│   ├── train_small.yaml       # Small model config
│   ├── train_medium.yaml      # Medium model config
│   └── train_large.yaml       # Large model config
├── scripts/                    # Helper scripts
│   ├── run_training.sh        # Linux/Mac launcher
│   └── run_training.bat       # Windows launcher
├── train_advanced.py          # Advanced training script (YAML configs)
├── train_ultrathink.py        # Original training script (CLI args)
├── colab_training.ipynb       # Google Colab notebook
└── outputs/                    # Training outputs
    └── [model_name]/
        ├── checkpoints/       # Model checkpoints
        ├── training.log       # Training logs
        └── final_model/       # Final trained model
```

## 🔧 Configuration Overview

### Key Configuration Sections

**Model Architecture**:
```yaml
model:
  hidden_size: 512      # Model dimension
  num_layers: 6         # Transformer layers
  num_heads: 8          # Attention heads
  max_seq_length: 1024  # Context length
```

**Advanced Features**:
```yaml
advanced:
  enable_moe: true           # Mixture of Experts
  enable_dre: true           # Dynamic Reasoning Engine
  enable_constitutional: true # Constitutional AI
  enable_rlhf: false         # RLHF (for fine-tuning)
  enable_multimodal: false   # Multimodal capabilities
```

**Training Settings**:
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 3e-4
  num_epochs: 3
  use_amp: true  # Mixed precision
```

**Data Configuration**:
```yaml
data:
  dataset: wikitext          # Dataset name
  train_samples: 50000       # Training samples
  val_samples: 5000          # Validation samples
  num_workers: 2             # Data loader workers
```

## 📈 Monitoring & Logging

### MLflow UI
```bash
# Start MLflow server
mlflow ui

# View at: http://localhost:5000
```

**Key Metrics to Monitor**:
- `train/loss` - Training loss (should decrease)
- `val/loss` - Validation loss (check for overfitting)
- `train/learning_rate` - Current learning rate
- `train/grad_norm` - Gradient norms (should be 0.1-10)

### Log Files
```bash
# View training logs
tail -f outputs/[model_name]/training.log

# On Windows
type outputs\[model_name]\training.log
```

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size, increase accumulation
python train_advanced.py \
  --config configs/train_small.yaml \
  --override \
    training.batch_size=1 \
    training.gradient_accumulation_steps=32
```

### Slow Training
```bash
# Enable optimizations
python train_advanced.py \
  --config configs/train_small.yaml \
  --override \
    training.use_amp=true \
    model.use_flash_attention=true \
    model.gradient_checkpointing=false
```

### NaN/Inf Loss
```bash
# Lower learning rate, add gradient clipping
python train_advanced.py \
  --config configs/train_small.yaml \
  --override \
    training.learning_rate=1e-5 \
    training.gradient_clipping=1.0
```

## 💡 Best Practices

1. **Start Small**: Test with small config first
2. **Monitor Closely**: Watch loss curves and GPU utilization
3. **Save Often**: Set `eval_frequency` appropriately
4. **Use MLflow**: Track all experiments
5. **Document Changes**: Use meaningful run names
6. **Enable AMP**: 2x speedup with minimal accuracy loss
7. **Optimize Data**: Use appropriate `num_workers`
8. **Scale Gradually**: Add features incrementally

## 🆘 Getting Help

- **Full Guide**: [ADVANCED_TRAINING_GUIDE.md](ADVANCED_TRAINING_GUIDE.md)
- **Documentation**: [README.md](README.md)
- **Model Card**: [MODEL_CARD.md](MODEL_CARD.md)
- **Issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)

## 🎓 Example Workflows

### Workflow 1: Quick Prototype
```bash
# Test with dummy data (1 minute)
python train_advanced.py \
  --config configs/train_small.yaml \
  --override data.dataset=dummy data.train_samples=100

# Full small model (2-4 hours)
python train_advanced.py --config configs/train_small.yaml
```

### Workflow 2: Production Training
```bash
# 1. Baseline training
python train_advanced.py \
  --config configs/train_medium.yaml \
  --run-name "baseline_v1"

# 2. Add advanced features
python train_advanced.py \
  --config configs/train_medium.yaml \
  --init-from ./outputs/medium_model/final_model \
  --override advanced.enable_moe=true advanced.enable_dre=true \
  --run-name "advanced_v1"

# 3. Fine-tune with RLHF
python train_advanced.py \
  --config configs/train_medium.yaml \
  --init-from ./outputs/medium_model/final_model \
  --override advanced.enable_rlhf=true training.learning_rate=1e-5 \
  --run-name "rlhf_v1"
```

### Workflow 3: Colab Training
```python
# In Colab notebook
!git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
%cd UltraThinking-LLM-Training
!pip install -r requirements.txt

!python train_advanced.py \
  --config configs/train_small.yaml \
  --override output.output_dir=/content/drive/MyDrive/ultrathink
```

## 📊 Expected Results

### Small Model (after 3 epochs on WikiText)
- Training Loss: ~2.5-3.0
- Validation Loss: ~3.0-3.5
- Training Time: ~4 hours on RTX 3090
- Model Size: ~400MB

### Medium Model (after 3 epochs)
- Training Loss: ~1.8-2.2
- Validation Loss: ~2.0-2.5
- Training Time: ~2 days on RTX 3090
- Model Size: ~8GB

### Large Model (after pretraining)
- Training Loss: ~1.2-1.5
- Validation Loss: ~1.5-2.0
- Training Time: ~2 weeks on 4x A100
- Model Size: ~40GB

---

**Ready to train?** Start with the small config and scale up! 🚀

```bash
python train_advanced.py --config configs/train_small.yaml
```
