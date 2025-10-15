# 📚 Training Small Models Guide

Complete guide for training ULTRATHINK models on limited hardware and small datasets.

## Why Train Small Models?

- **Fast Iteration** - Test ideas quickly (minutes, not hours)
- **Limited Hardware** - Works on consumer GPUs or even CPU
- **Learning** - Understand the framework before scaling up
- **Prototyping** - Validate architecture changes
- **Cost-Effective** - No expensive cloud compute needed

## Hardware Requirements

### Minimum Specs

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None (CPU works) | GTX 1080 Ti (11GB) |
| **Storage** | 10 GB | 50+ GB SSD |

### Model Size vs Hardware

| Model Size | Parameters | Min GPU | Training Speed |
|-----------|-----------|---------|----------------|
| **Tiny** | 125M | CPU / 6GB GPU | Fast |
| **Small** | 350M | 12GB GPU | Medium |
| **Medium** | 760M | 24GB GPU | Slow |

## Quick Start Examples

### 1. CPU Training (No GPU Required)

Perfect for testing the pipeline:

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 256 \
  --num_layers 2 \
  --num_heads 4 \
  --batch_size 2 \
  --max_samples 1000 \
  --num_epochs 1 \
  --output_dir ./outputs/cpu_test
```

**Expected**: 5-10 minutes, loss ~6.5 → ~4.2

### 2. Small GPU Training (6-12GB)

For actual model training:

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 512 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 4 \
  --num_epochs 3 \
  --use_amp \
  --gradient_checkpointing \
  --output_dir ./outputs/small_gpu
```

**Expected**: 30-60 minutes, loss ~5.8 → ~3.1

### 3. Using Configuration Files

Easiest and most reproducible:

```bash
python train_advanced.py --config configs/train_small.yaml
```

## Best Datasets for Small Models

### 1. WikiText-2 (Recommended for Testing)

**Size**: ~4MB, ~2M tokens  
**Training Time**: 5-15 minutes  
**Use Case**: Quick iteration, testing

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --dataset_subset wikitext-2-raw-v1 \
  --hidden_size 512 --num_layers 6 \
  --batch_size 4 --num_epochs 5
```

### 2. WikiText-103 (Better Quality)

**Size**: ~500MB, ~100M tokens  
**Training Time**: 2-4 hours  
**Use Case**: Actual model training

```bash
python train_ultrathink.py \
  --dataset wikitext \
  --dataset_subset wikitext-103-raw-v1 \
  --hidden_size 768 --num_layers 8 \
  --batch_size 2 --gradient_accumulation_steps 8 \
  --use_amp --gradient_checkpointing
```

### 3. C4 Streaming (Production-Like)

**Size**: Unlimited (streaming)  
**Training Time**: Depends on max_steps  
**Use Case**: Production training

```bash
python train_ultrathink.py \
  --dataset c4 --streaming \
  --max_steps 10000 \
  --hidden_size 768 --num_layers 12 \
  --batch_size 2 --gradient_accumulation_steps 16 \
  --use_amp --gradient_checkpointing
```

## Memory Optimization Techniques

### 1. Gradient Checkpointing

**Saves**: 40-50% memory  
**Cost**: 15-20% slower training

```bash
python train_ultrathink.py \
  --gradient_checkpointing \
  --hidden_size 768 --num_layers 12
```

### 2. Gradient Accumulation

**Saves**: Allows smaller batch sizes  
**Benefit**: Simulate larger batches

```bash
python train_ultrathink.py \
  --batch_size 1 \
  --gradient_accumulation_steps 32 \
  # Effective batch size = 1 × 32 = 32
```

### 3. Mixed Precision (FP16)

**Saves**: 50% memory  
**Benefit**: 1.5-2x faster training

```bash
python train_ultrathink.py \
  --use_amp \
  --hidden_size 768 --num_layers 12
```

### 4. Reduce Sequence Length

**Saves**: Quadratic memory reduction  
**Trade-off**: Shorter context

```bash
python train_ultrathink.py \
  --max_seq_length 512 \  # Instead of 2048
  --hidden_size 768 --num_layers 12
```

### 5. Smaller Vocabulary

**Saves**: Embedding memory  
**Trade-off**: Less token coverage

```bash
python train_ultrathink.py \
  --vocab_size 32000 \  # Instead of 50257
  --hidden_size 768 --num_layers 12
```

## Complete Small Model Configuration

### Tiny Model (CPU-Friendly)

```yaml
# configs/train_tiny_custom.yaml
model:
  vocab_size: 50257
  hidden_size: 256
  num_layers: 2
  num_heads: 4
  max_seq_length: 512
  
training:
  batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 3
  learning_rate: 3e-4
  warmup_steps: 100
  
  # Optimizations
  use_amp: false  # CPU doesn't support AMP
  gradient_checkpointing: false
  
data:
  dataset: wikitext
  dataset_subset: wikitext-2-raw-v1
  max_samples: 5000
  num_workers: 2
  
output:
  output_dir: ./outputs/tiny_model
  save_steps: 500
  logging_steps: 50
```

### Small Model (6-16GB GPU)

```yaml
# configs/train_small_custom.yaml
model:
  vocab_size: 50257
  hidden_size: 512
  num_layers: 6
  num_heads: 8
  max_seq_length: 1024
  
training:
  batch_size: 4
  gradient_accumulation_steps: 16
  num_epochs: 5
  learning_rate: 1e-4
  warmup_steps: 500
  
  # Optimizations
  use_amp: true
  gradient_checkpointing: true
  
data:
  dataset: wikitext
  dataset_subset: wikitext-103-raw-v1
  streaming: false
  num_workers: 4
  
monitoring:
  use_mlflow: true
  use_wandb: false
  
output:
  output_dir: ./outputs/small_model
  save_steps: 1000
  logging_steps: 100
```

## Training Tips

### 1. Start Small, Scale Up

```bash
# Step 1: Test on tiny model (5 min)
python train_ultrathink.py --hidden_size 128 --num_layers 2 --max_samples 500

# Step 2: Small model, small data (30 min)
python train_ultrathink.py --hidden_size 256 --num_layers 4 --max_samples 5000

# Step 3: Full small model (2-4 hours)
python train_advanced.py --config configs/train_small.yaml
```

### 2. Monitor Memory Usage

```bash
# Watch GPU memory
watch -n 1 nvidia-smi

# Or use built-in monitoring
python train_ultrathink.py \
  --monitor_system \
  --hidden_size 512 --num_layers 6
```

### 3. Use Checkpointing

```bash
python train_ultrathink.py \
  --save_steps 500 \
  --save_total_limit 3 \  # Keep only last 3 checkpoints
  --output_dir ./outputs/my_model
```

### 4. Resume from Checkpoint

```bash
python train_ultrathink.py \
  --resume_from_checkpoint ./outputs/my_model/checkpoint-1000 \
  --hidden_size 512 --num_layers 6
```

## Common Issues & Solutions

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions** (try in order):
1. Reduce batch size: `--batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Reduce sequence length: `--max_seq_length 512`
4. Use gradient accumulation: `--gradient_accumulation_steps 16`
5. Reduce model size: `--hidden_size 256 --num_layers 4`

### Slow Training

**Symptoms**: <1000 tokens/sec

**Solutions**:
1. Enable mixed precision: `--use_amp`
2. Increase workers: `--num_workers 4`
3. Use smaller dataset: `--max_samples 10000`
4. Disable unnecessary logging: `--logging_steps 500`

### Poor Model Quality

**Symptoms**: High loss, poor generation

**Solutions**:
1. Train longer: `--num_epochs 10`
2. Use more data: Remove `--max_samples`
3. Increase model size: `--hidden_size 768 --num_layers 8`
4. Lower learning rate: `--learning_rate 1e-4`
5. Add warmup: `--warmup_steps 1000`

### NaN Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
1. Lower learning rate: `--learning_rate 1e-5`
2. Enable gradient clipping: `--max_grad_norm 1.0`
3. Use mixed precision carefully: Try without `--use_amp`
4. Check data quality: Ensure no corrupted samples

## Monitoring Training

### MLflow (Recommended)

```bash
# Start training with MLflow
python train_ultrathink.py \
  --use_mlflow \
  --hidden_size 512 --num_layers 6

# View in browser
mlflow ui
# Open http://localhost:5000
```

### Weights & Biases

```bash
# Login once
wandb login

# Train with W&B
python train_ultrathink.py \
  --use_wandb \
  --wandb_project my_project \
  --hidden_size 512 --num_layers 6
```

### TensorBoard

```bash
# Training automatically logs to tensorboard
python train_ultrathink.py \
  --output_dir ./outputs/my_model \
  --hidden_size 512 --num_layers 6

# View logs
tensorboard --logdir ./outputs/my_model/tensorboard
```

## Evaluation

### Generate Text

```bash
python scripts/inference.py \
  --model_path ./outputs/small_model \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.8
```

### Calculate Perplexity

```bash
python scripts/evaluate.py \
  --model_path ./outputs/small_model \
  --dataset wikitext \
  --split test
```

### Benchmark Performance

```bash
python scripts/profile_model.py \
  --model_path ./outputs/small_model \
  --batch_size 1 \
  --seq_length 512
```

## Example Training Schedule

### Week 1: Learning (CPU/Small GPU)

```bash
# Day 1-2: Tiny model, understand pipeline
python train_ultrathink.py --hidden_size 128 --num_layers 2 --max_samples 1000

# Day 3-4: Small model, WikiText-2
python train_ultrathink.py --hidden_size 256 --num_layers 4 --dataset wikitext

# Day 5-7: Medium model, WikiText-103
python train_ultrathink.py --hidden_size 512 --num_layers 6 --use_amp
```

### Week 2: Optimization (GPU)

```bash
# Enable advanced features one by one
python train_ultrathink.py \
  --hidden_size 768 --num_layers 8 \
  --enable_moe --num_experts 4 \
  --use_amp --gradient_checkpointing
```

### Week 3: Production (GPU)

```bash
# Full training on C4
python train_advanced.py --config configs/train_small.yaml
```

## Cost Estimation

### Local Training (One-time GPU cost)

| GPU | Price | Tiny Model | Small Model | Medium Model |
|-----|-------|-----------|-------------|--------------|
| RTX 3060 (12GB) | $300 | ✅ Fast | ✅ Medium | ❌ |
| RTX 3090 (24GB) | $1,500 | ✅ Very Fast | ✅ Fast | ✅ Slow |
| RTX 4090 (24GB) | $1,600 | ✅ Very Fast | ✅ Very Fast | ✅ Medium |

### Cloud Training (Pay per use)

| Provider | Instance | GPU | Cost/hour | 10hr Training |
|----------|----------|-----|-----------|---------------|
| Google Colab Pro+ | - | T4/V100 | $0 | **Free** |
| AWS | p3.2xlarge | V100 | $3.06 | $30.60 |
| GCP | n1-highmem-8 + T4 | T4 | $0.95 | $9.50 |
| Lambda Labs | gpu_1x_a6000 | A6000 | $0.80 | $8.00 |

**Recommendation**: Start with Google Colab Pro+ ($50/month) for unlimited training.

## Next Steps

### Ready for More?

- **[Advanced Training Guide](../ADVANCED_TRAINING_GUIDE.md)** - MoE, DRE, Constitutional AI
- **[DeepSpeed Integration](training_deepspeed.md)** - Memory-efficient training
- **[Custom Datasets](datasets.md)** - Use your own data
- **[Benchmarks](BENCHMARKS.md)** - Compare your results

### Need Help?

- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
- **[FAQ](faq.md)** - Frequently asked questions
- **[GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)** - Report bugs

---

**Happy Training!** 🚀 Start small, iterate fast, scale when ready.
