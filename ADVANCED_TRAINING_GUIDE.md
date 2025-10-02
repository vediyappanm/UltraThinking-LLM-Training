# ULTRATHINK Advanced Training Guide

Complete guide for training the ULTRATHINK model with all advanced features.

## Table of Contents
- [Quick Start](#quick-start)
- [Training Profiles](#training-profiles)
- [Advanced Features](#advanced-features)
- [Configuration System](#configuration-system)
- [Training Environments](#training-environments)
- [Monitoring & Debugging](#monitoring--debugging)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Local Training

#### Option 1: Using Training Profiles
```bash
# Small model (for testing)
python scripts/run_training.bat --profile small

# Medium model (production-ready)
python scripts/run_training.bat --profile medium

# Large model (full-scale)
python scripts/run_training.bat --profile large
```

#### Option 2: Using Configuration Files
```bash
python train_advanced.py --config configs/train_small.yaml
```

#### Option 3: Direct Command Line
```bash
python train_ultrathink.py \
  --use_mlflow \
  --dataset wikitext \
  --hidden_size 512 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 4 \
  --enable_moe \
  --enable_dre
```

### Google Colab Training

1. Open `colab_training.ipynb` in Google Colab
2. Choose your GPU runtime (T4/V100/A100)
3. Select a configuration cell and run it
4. Monitor progress in MLflow UI or logs

---

## Training Profiles

### Small Profile
**Hardware**: Local machines, 4-8GB VRAM  
**Use Case**: Testing, development, prototyping

```yaml
Model Size: 512 hidden, 6 layers
Features: Basic transformer
Training Time: ~2-4 hours (10K samples)
Memory: ~6GB VRAM
```

**When to use**:
- Testing new features
- Debugging training pipeline
- Quick experiments
- Limited hardware resources

### Medium Profile
**Hardware**: Single GPU (16-32GB), cloud instances  
**Use Case**: Production models, research

```yaml
Model Size: 2048 hidden, 24 layers
Features: MoE, DRE, Constitutional AI
Training Time: ~1-2 days (100K samples)
Memory: ~20GB VRAM
```

**When to use**:
- Production deployments
- Research experiments
- Competitive benchmarks
- Full feature validation

### Large Profile
**Hardware**: Multi-GPU, A100/H100 clusters  
**Use Case**: State-of-the-art models

```yaml
Model Size: 4096 hidden, 32 layers
Features: All advanced features + Multimodal
Training Time: ~1-2 weeks (1M+ samples)
Memory: ~40-80GB VRAM
```

**When to use**:
- Frontier model development
- Large-scale pretraining
- Multimodal applications
- Maximum performance

---

## Advanced Features

### 1. Mixture of Experts (MoE)

**What it does**: Routes inputs to specialized expert networks for efficient scaling.

**Enable**:
```yaml
advanced:
  enable_moe: true

moe:
  num_knowledge_experts: 32
  num_skill_experts: 16
  num_meta_experts: 8
  num_safety_experts: 4
  moe_top_k: 2
```

**Benefits**:
- ✅ 5-10x parameter scaling with minimal compute increase
- ✅ Specialized knowledge domains
- ✅ Better performance on diverse tasks

**Considerations**:
- Requires more memory for expert parameters
- Best with batch_size >= 4
- Works with expert parallelism for large scale

### 2. Dynamic Reasoning Engine (DRE)

**What it does**: Adaptive computational paths based on input complexity.

**Enable**:
```yaml
advanced:
  enable_dre: true
  dre_warmup_steps: 5000
```

**Benefits**:
- ✅ Adaptive reasoning depth
- ✅ Better on complex problems
- ✅ Improved efficiency

**Considerations**:
- Use warmup to stabilize training
- May increase training time initially
- Best for reasoning-heavy tasks

### 3. Constitutional AI

**What it does**: Built-in safety and alignment mechanisms.

**Enable**:
```yaml
advanced:
  enable_constitutional: true
```

**Benefits**:
- ✅ Safer outputs
- ✅ Better alignment
- ✅ Reduced harmful content

**Considerations**:
- Adds overhead (~5-10%)
- Best combined with RLHF
- Requires safety datasets

### 4. RLHF (Reinforcement Learning from Human Feedback)

**What it does**: Fine-tunes model based on human preferences.

**Enable**:
```yaml
advanced:
  enable_rlhf: true

rlhf:
  rlhf_frequency: 5
  rlhf_iterations: 100
  ppo_epochs: 4
```

**When to use**:
- After pretraining
- For instruction following
- For alignment fine-tuning

**Process**:
1. Pretrain model without RLHF
2. Save checkpoint
3. Resume with RLHF enabled
4. Fine-tune with preference data

### 5. Multimodal Capabilities

**What it does**: Process images, audio, and text together.

**Enable**:
```yaml
advanced:
  enable_multimodal: true

multimodal:
  image_size: 224
  patch_size: 14
  audio_sample_rate: 16000
```

**Requirements**:
- Multimodal datasets
- Larger memory (images/audio)
- Vision/audio encoders

---

## Configuration System

### YAML Configuration Structure

```yaml
# Model architecture
model:
  vocab_size: 100352
  hidden_size: 2048
  num_layers: 24
  ...

# Advanced features
advanced:
  enable_moe: true
  enable_dre: true
  ...

# Training hyperparameters
training:
  batch_size: 8
  learning_rate: 1e-4
  ...

# Data configuration
data:
  dataset: wikitext
  ...

# Logging and monitoring
logging:
  use_mlflow: true
  ...
```

### Override Configuration Values

```bash
python train_advanced.py \
  --config configs/train_medium.yaml \
  --override \
    training.batch_size=4 \
    model.hidden_size=1024 \
    advanced.enable_moe=false
```

### Create Custom Configurations

1. Copy existing config:
```bash
cp configs/train_medium.yaml configs/my_config.yaml
```

2. Edit values in `my_config.yaml`

3. Run training:
```bash
python train_advanced.py --config configs/my_config.yaml
```

---

## Training Environments

### Local Training

#### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow UI
mlflow ui

# Run training
python train_advanced.py --config configs/train_small.yaml
```

#### Monitor
- MLflow UI: http://localhost:5000
- Logs: `./outputs/[model_name]/training.log`
- Checkpoints: `./outputs/[model_name]/checkpoint_*.pt`

### Google Colab Training

#### Setup
1. Open `colab_training.ipynb`
2. Select GPU runtime
3. Mount Google Drive
4. Install dependencies

#### Benefits
- Free GPU access (T4)
- Paid options (V100/A100)
- Persistent storage via Drive
- Easy sharing

#### Limitations
- Session timeouts (~12 hours)
- GPU availability varies
- Slower than dedicated hardware

### Cloud Training (AWS/GCP/Azure)

#### AWS SageMaker
```python
# Use train_advanced.py with SageMaker estimator
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train_advanced.py',
    instance_type='ml.p3.8xlarge',
    instance_count=1,
    hyperparameters={
        'config': 'configs/train_large.yaml'
    }
)
estimator.fit()
```

#### GCP AI Platform
```bash
gcloud ai-platform jobs submit training ultrathink_job \
  --region=us-central1 \
  --master-machine-type=n1-highmem-16 \
  --master-accelerator=type=nvidia-tesla-v100,count=4 \
  --package-path=./src \
  --module-name=train_advanced \
  -- \
  --config=configs/train_large.yaml
```

### Distributed Training

#### Multi-GPU (Single Node)
```bash
# Using torchrun
torchrun --nproc_per_node=4 train_advanced.py \
  --config configs/train_large.yaml \
  --override distributed.enabled=true
```

#### Multi-Node with DeepSpeed
```bash
# Create hostfile
echo "node1 slots=8" > hostfile
echo "node2 slots=8" >> hostfile

# Run with DeepSpeed
deepspeed --hostfile=hostfile train_advanced.py \
  --config configs/train_large.yaml \
  --override \
    distributed.enabled=true \
    distributed.launcher=deepspeed \
    distributed.deepspeed_config=config/deepspeed_z3.json
```

---

## Monitoring & Debugging

### MLflow Tracking

**View Experiments**:
```bash
mlflow ui
# Open http://localhost:5000
```

**Key Metrics**:
- `train/loss`: Training loss
- `val/loss`: Validation loss
- `train/learning_rate`: Current LR
- `train/grad_norm`: Gradient norms
- `eval/*`: Evaluation metrics

**Artifacts**:
- Config files
- Checkpoints
- Evaluation results
- Model exports

### Logging

**Log Levels**:
```python
# In config or code
logging.basicConfig(level=logging.INFO)  # INFO, DEBUG, WARNING, ERROR
```

**Log Files**:
- Training log: `./outputs/[model_name]/training.log`
- MLflow logs: `./mlruns/`

### Debugging Tips

**Memory Issues**:
```bash
# Check GPU memory
nvidia-smi

# Enable memory profiling
python train_advanced.py \
  --config configs/train_small.yaml \
  --override training.use_amp=true model.gradient_checkpointing=true
```

**Slow Training**:
```bash
# Enable profiling
python -m torch.utils.bottleneck train_advanced.py --config configs/train_small.yaml
```

**NaN/Inf Loss**:
```yaml
# Add gradient clipping
training:
  gradient_clipping: 1.0
  
# Reduce learning rate
training:
  learning_rate: 1e-5
  
# Enable AMP carefully
training:
  use_amp: true
```

---

## Best Practices

### 1. Start Small, Scale Up
- ✅ Test on small config first
- ✅ Verify all features work
- ✅ Then scale to production

### 2. Use Mixed Precision (AMP)
```yaml
training:
  use_amp: true
```
- 2x faster training
- 50% less memory
- Minimal accuracy loss

### 3. Gradient Checkpointing for Large Models
```yaml
model:
  gradient_checkpointing: true
```
- Trades compute for memory
- Enables larger models
- ~20% slower, 40% less memory

### 4. Optimize Data Loading
```yaml
data:
  num_workers: 8  # Set to CPU cores
  streaming: true  # For large datasets
```

### 5. Save Checkpoints Regularly
```yaml
evaluation:
  eval_frequency: 1  # Save every epoch
```

### 6. Monitor Gradient Norms
- Healthy range: 0.1 - 10
- Too high (>100): Reduce LR or increase clipping
- Too low (<0.01): Increase LR or check optimizer

### 7. Use Learning Rate Warmup
```yaml
training:
  warmup_steps: 2000  # Gradual LR increase
```

### 8. Enable Advanced Features Gradually
1. Train baseline model
2. Add MoE
3. Add DRE
4. Add Constitutional AI
5. Fine-tune with RLHF

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
```yaml
# 1. Reduce batch size
training:
  batch_size: 2
  gradient_accumulation_steps: 16  # Maintain effective batch size

# 2. Enable gradient checkpointing
model:
  gradient_checkpointing: true

# 3. Reduce sequence length
model:
  max_seq_length: 512

# 4. Use smaller model
model:
  hidden_size: 512
  num_layers: 6
```

### Slow Training

**Symptoms**: Low tokens/second

**Solutions**:
```yaml
# 1. Enable flash attention
model:
  use_flash_attention: true

# 2. Use AMP
training:
  use_amp: true

# 3. Increase batch size
training:
  batch_size: 16

# 4. Optimize data loading
data:
  num_workers: 8
  streaming: true
```

### Training Instability

**Symptoms**: NaN loss, exploding gradients

**Solutions**:
```yaml
# 1. Enable gradient clipping
training:
  gradient_clipping: 1.0

# 2. Reduce learning rate
training:
  learning_rate: 1e-5

# 3. Increase warmup
training:
  warmup_steps: 10000

# 4. Use DRE warmup
advanced:
  dre_warmup_steps: 5000
```

### Poor Performance

**Symptoms**: High validation loss, poor benchmarks

**Solutions**:
1. Train longer (more epochs/steps)
2. Increase model size
3. Use better datasets
4. Enable advanced features (MoE, DRE)
5. Fine-tune with RLHF
6. Check data quality
7. Verify tokenization
8. Monitor overfitting

### Dataset Issues

**Symptoms**: Dataset loading errors

**Solutions**:
```bash
# 1. Use dummy dataset for testing
python train_advanced.py \
  --config configs/train_small.yaml \
  --override data.dataset=dummy

# 2. Check dataset availability
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')"

# 3. Use streaming for large datasets
--override data.streaming=true
```

---

## Example Training Workflows

### Workflow 1: Quick Prototype
```bash
# 1. Test with dummy data
python train_advanced.py --config configs/train_small.yaml \
  --override data.dataset=dummy data.train_samples=1000

# 2. Run on real data
python train_advanced.py --config configs/train_small.yaml \
  --override data.dataset=wikitext training.num_epochs=1
```

### Workflow 2: Production Training
```bash
# 1. Pretrain baseline
python train_advanced.py --config configs/train_medium.yaml \
  --run-name "baseline_v1"

# 2. Resume with advanced features
python train_advanced.py --config configs/train_medium.yaml \
  --init-from ./outputs/medium_model/final_model \
  --override advanced.enable_moe=true advanced.enable_dre=true \
  --run-name "advanced_v1"

# 3. Fine-tune with RLHF
python train_advanced.py --config configs/train_medium.yaml \
  --init-from ./outputs/medium_model/final_model \
  --override advanced.enable_rlhf=true training.learning_rate=1e-5 \
  --run-name "rlhf_v1"
```

### Workflow 3: Multi-GPU Training
```bash
# Use torchrun for data parallelism
torchrun --nproc_per_node=4 train_advanced.py \
  --config configs/train_large.yaml \
  --override distributed.enabled=true
```

---

## Performance Benchmarks

### Expected Throughput

| Profile | Hardware | Tokens/sec | Time (100K samples) |
|---------|----------|------------|---------------------|
| Small   | GTX 1080 Ti | ~5000 | 4 hours |
| Medium  | RTX 3090 | ~2000 | 1 day |
| Large   | A100 40GB | ~800 | 1 week |
| Large   | 4x A100 | ~3000 | 2 days |

### Memory Requirements

| Profile | Parameters | Memory (FP32) | Memory (FP16) |
|---------|------------|---------------|---------------|
| Small   | 100M | 6 GB | 4 GB |
| Medium  | 2B | 24 GB | 16 GB |
| Large   | 10B | 80 GB | 48 GB |

---

## Additional Resources

- **Documentation**: [README.md](README.md)
- **Model Card**: [MODEL_CARD.md](MODEL_CARD.md)
- **API Reference**: [docs/README.md](docs/README.md)
- **GitHub Issues**: Report bugs and request features
- **Discord/Slack**: Community support (if available)

---

## Getting Help

1. **Check logs**: `./outputs/[model_name]/training.log`
2. **Search issues**: GitHub issues page
3. **Community**: Discord/Slack channels
4. **Documentation**: This guide and other docs

Happy training! 🚀
