# ULTRATHINK Quick Reference

Fast lookup guide for common tasks and file locations.

---

## 🚀 Common Commands

### Training
```bash
# Tiny model (testing)
python train_ultrathink.py --hidden_size 256 --num_layers 2 --batch_size 2

# Small model (laptop)
python train_ultrathink.py --hidden_size 384 --num_layers 4 --num_heads 6

# Medium model (single GPU)
python train_ultrathink.py --hidden_size 768 --num_layers 12 --use_amp

# Large model (multi-GPU)
deepspeed train_ultrathink.py --deepspeed config/deepspeed_z2.json
```

### Testing
```bash
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest tests/unit/                  # Unit tests only
pytest --cov=src                    # With coverage
pytest -k "test_attention"          # Specific test
```

### Utilities
```bash
python scripts/profile_model.py     # Profile performance
python scripts/cleanup.py           # Clean cache
python scripts/inference.py         # Generate text
python scripts/recover_checkpoint.py # Fix broken checkpoint
```

---

## 📁 File Locations Quick Reference

| Need to... | Go to... |
|------------|----------|
| **Modify model architecture** | `src/models/architecture.py` |
| **Change transformer blocks** | `src/models/architecture.py` (TransformerBlock class) |
| **Add new dataset** | `src/data/datasets.py` (DATASET_CONFIGS) |
| **Modify training loop** | `src/training/loop.py` |
| **Add new optimizer** | `src/training/optimizers.py` |
| **Change logging** | `src/monitoring/metrics.py` |
| **Add MoE expert** | `src/models/moe_advanced.py` |
| **Modify DRE logic** | `src/models/dynamic_reasoning.py` |
| **Add multimodal support** | `src/models/multimodal.py` |
| **Configure safety** | `src/models/constitutional_ai.py` |
| **Write unit test** | `tests/unit/` |
| **Profile code** | `scripts/profile_model.py` |
| **DeepSpeed config** | `config/deepspeed_z1.json` or `deepspeed_z3.json` |

---

## 🎯 Key Classes & Functions

### Model Classes
```python
# Core architecture
from src.models.architecture import (
    RMSNorm,          # Normalization
    RoPE,             # Position encoding
    Attention,        # Multi-head attention
    FeedForward,      # MLP layer
    TransformerBlock, # Complete layer
    GPTModel          # Base model
)

# Main model
from src.models.ultrathink import UltraThinkModel

# Advanced features
from src.models.moe_advanced import HierarchicalMoE
from src.models.dynamic_reasoning import DynamicReasoningEngine
from src.models.multimodal import MultimodalEncoder
from src.models.constitutional_ai import ConstitutionalAI
```

### Data Classes
```python
from src.data.datasets import (
    TextDataset,      # Generic text dataset
    MixedDataset,     # Mix multiple datasets
    DummyDataset,     # Quick testing
    load_dataset      # Main function
)

from src.data.validation import (
    validate_sample,  # Check data quality
    detect_duplicates # Find duplicates
)
```

### Training Classes
```python
from src.training.optimizers import (
    create_optimizer,  # Factory function
    AdamWOptimizer,
    SophiaOptimizer,
    LAMBOptimizer
)

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
```

### Monitoring Classes
```python
from src.monitoring.metrics import MetricsLogger
from src.monitoring.system_monitor import SystemMonitor
```

---

## 🔧 Configuration Arguments

### Model Size Presets
```bash
# Tiny (42M params)
--hidden_size 384 --num_layers 6 --num_heads 6

# Small (125M params)  
--hidden_size 768 --num_layers 12 --num_heads 12

# Medium (350M params)
--hidden_size 1024 --num_layers 24 --num_heads 16

# Large (760M params)
--hidden_size 1536 --num_layers 24 --num_heads 16

# XL (1.3B params)
--hidden_size 2048 --num_layers 24 --num_heads 16
```

### Common Training Args
```bash
# Learning
--learning_rate 3e-4        # LR (default: 3e-5)
--warmup_steps 2000         # Warmup (default: 10000)
--weight_decay 0.1          # Weight decay (default: 0.01)
--gradient_clipping 1.0     # Grad clip (default: 1.0)

# Batch size
--batch_size 8              # Per-device batch
--gradient_accumulation_steps 4  # Grad accumulation

# Optimization
--use_amp                   # Mixed precision
--gradient_checkpointing    # Save memory
--use_flash_attention       # Fast attention

# Dataset
--dataset wikitext          # Dataset name
--streaming                 # Stream large datasets
--max_samples 10000         # Limit samples

# Monitoring
--use_wandb                 # Enable W&B
--run_name my_experiment    # Run name
--eval_frequency 100        # Eval every N steps
```

### Advanced Features
```bash
# Dynamic Reasoning
--enable_dre
--dre_warmup_steps 500

# Mixture of Experts
--enable_moe
--num_knowledge_experts 64
--num_skill_experts 32
--moe_top_k 2

# Multimodal
--enable_multimodal
--image_size 224

# RLHF
--enable_rlhf
--rlhf_frequency 5
```

---

## 📊 Monitoring Metrics

### Training Metrics (Logged)
- `loss` - Training loss
- `val_loss` - Validation loss
- `perplexity` - Model perplexity (exp(loss))
- `learning_rate` - Current LR
- `grad_norm` - Gradient norm
- `throughput` - Tokens/second

### System Metrics (Logged)
- `gpu_memory_allocated` - GPU memory used
- `gpu_memory_reserved` - GPU memory reserved
- `gpu_utilization` - GPU usage %
- `cpu_percent` - CPU usage %
- `ram_percent` - RAM usage %

### Model Metrics (Logged)
- `param_norm` - Model parameter norm
- `moe_load_balance` - Expert load balance (if MoE)
- `dre_complexity` - Average complexity (if DRE)

---

## 🐛 Debugging Guide

### Problem: Out of Memory
**Solutions**:
1. Reduce `--batch_size`
2. Add `--gradient_checkpointing`
3. Use `--use_amp` (mixed precision)
4. Enable `--streaming` for data
5. Use DeepSpeed ZeRO: `--deepspeed config/deepspeed_z3.json`

### Problem: Loss is NaN
**Solutions**:
1. Lower `--learning_rate` (try 1e-5)
2. Increase `--warmup_steps` (try 1000)
3. Check data: `from src.data.validation import validate_sample`
4. Enable `--use_amp` with `--amp_warmup_steps 200`
5. Add `--gradient_clipping 0.5`

### Problem: Training is slow
**Solutions**:
1. Profile: `python scripts/profile_model.py`
2. Enable `--use_flash_attention`
3. Increase `--batch_size` or `--gradient_accumulation_steps`
4. Use `--streaming` for large datasets
5. Enable `--use_amp`
6. Use multiple GPUs with DeepSpeed

### Problem: Model not learning
**Solutions**:
1. Check loss curve in W&B/TensorBoard
2. Verify data: print first batch
3. Try higher `--learning_rate`
4. Check if gradients flow: `--log_level DEBUG`
5. Reduce `--gradient_clipping` or remove it

---

## 🔍 Code Navigation

### Finding Things
```bash
# Find where a class is defined
grep -r "class UltraThinkModel" src/

# Find where a function is used
grep -r "load_dataset" src/

# Find all TODO comments
grep -r "TODO" src/

# Find all imports of a module
grep -r "from src.models import" .
```

### Understanding Control Flow
1. **Start**: `train_ultrathink.py` → `main()` function
2. **Model creation**: Line ~300 → `UltraThinkTrainer.__init__()`
3. **Training loop**: Line ~800 → `UltraThinkTrainer.train()`
4. **Forward pass**: `src/models/ultrathink.py` → `forward()` method
5. **Backward pass**: `src/training/loop.py` → loss computation

---

## 📈 Performance Benchmarks

### Typical Training Speed (A100 GPU)

| Model Size | Tokens/sec | Memory (GB) | Batch Size |
|------------|-----------|-------------|------------|
| 125M       | ~50K      | ~8          | 32         |
| 350M       | ~20K      | ~16         | 16         |
| 760M       | ~10K      | ~32         | 8          |
| 1.3B       | ~5K       | ~48         | 4          |

### With Optimizations

| Optimization | Speed Boost | Memory Savings |
|-------------|-------------|----------------|
| Flash Attention | 2-4x | 0% |
| Mixed Precision | 1.5-2x | 50% |
| Gradient Checkpointing | 0.8x | 30-50% |
| DeepSpeed ZeRO-2 | 1x | 50% |
| DeepSpeed ZeRO-3 | 0.9x | 70% |

---

## 🎓 Learning Resources

### Code Reading Path
1. **Beginner**: `src/models/architecture.py` → Understand transformers
2. **Intermediate**: `train_ultrathink.py` → Understand training
3. **Advanced**: `src/training/distributed_4d.py` → Understand parallelism

### Key Papers
- **Transformer**: "Attention is All You Need"
- **GPT**: "Language Models are Unsupervised Multitask Learners"
- **MoE**: "Switch Transformers"
- **RLHF**: "Training language models to follow instructions"
- **Flash Attention**: "FlashAttention: Fast and Memory-Efficient"

### External Docs
- [PyTorch Docs](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [DeepSpeed Docs](https://www.deepspeed.ai/docs/)
- [W&B Docs](https://docs.wandb.ai/)

---

## 💡 Pro Tips

### Tip 1: Always test small first
```bash
# Quick sanity check before long training
python train_ultrathink.py \
  --hidden_size 128 --num_layers 2 \
  --max_samples 100 --num_epochs 1
```

### Tip 2: Use dummy dataset for debugging
```bash
python train_ultrathink.py --dataset dummy --train_samples 100
```

### Tip 3: Profile before optimizing
```bash
python scripts/profile_model.py --size tiny
# Look at Chrome trace to find bottlenecks
```

### Tip 4: Monitor system resources
```python
from src.monitoring import SystemMonitor
monitor = SystemMonitor()
monitor.check_and_log()  # In training loop
```

### Tip 5: Save checkpoints frequently
```bash
--save_frequency 100  # Save every 100 steps
```

---

## 🔗 Quick Links

- [Full Project Structure](PROJECT_STRUCTURE.md)
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Installation Guide](INSTALLATION_GUIDE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [Model Card](MODEL_CARD.md)

---

**Bookmark this page for quick reference! 📌**
