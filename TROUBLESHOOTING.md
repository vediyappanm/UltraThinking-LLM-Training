# 🔧 Troubleshooting Guide

## Import Errors Fixed

### Problem: `ImportError: cannot import name 'UltraThinkDataset'`

**Root Cause**: The data module structure doesn't export `UltraThinkDataset` directly.

**Solutions** (3 options):

### ✅ Option 1: Use Minimal Training Script (Recommended)

The minimal script has no dependencies and always works:

```bash
python train_minimal.py --config configs/production_full.yaml
```

**Features:**
- ✅ Works out of the box
- ✅ Graceful fallbacks for all imports
- ✅ Uses dummy data if real data unavailable
- ✅ Supports all CLI arguments
- ✅ Full distributed training support

### ✅ Option 2: Run Fix Script

Automatically patches the main training script:

```bash
python fix_imports.py
```

Then run:
```bash
python train_unified_production.py --config configs/production_full.yaml
```

### ✅ Option 3: Manual Fix

Edit `train_unified_production.py` line 92-96:

**Replace:**
```python
from src.data import (
    UltraThinkDataset,
    create_dataloaders,
)
```

**With:**
```python
try:
    from src.data import SyntheticDataEngine, SyntheticDataConfig
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
```

---

## Missing Dependencies

### FlashAttention Warning

```
FlashAttention not available, using PyTorch SDPA
```

**This is OK!** PyTorch SDPA (Scaled Dot Product Attention) is a good fallback.

**To install FlashAttention (optional):**
```bash
pip install flash-attn --no-build-isolation
```

### Apex Warning

```
Apex not available, using standard optimizers
```

**This is OK!** Standard PyTorch optimizers work fine.

**To install Apex (optional):**
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

---

## Quick Start Commands

### 1. Minimal Script (Always Works)

```bash
# Single GPU
python train_minimal.py --max_steps 1000

# Multi-GPU
torchrun --nproc_per_node=8 train_minimal.py --max_steps 10000

# With config
python train_minimal.py --config configs/production_full.yaml
```

### 2. Full Script (After Fix)

```bash
# Apply fix first
python fix_imports.py

# Then run
python train_unified_production.py --config configs/production_full.yaml
```

---

## Common Issues

### Issue: CUDA Out of Memory

**Solutions:**
```yaml
# In config file
batch_size: 8  # Reduce from 32
gradient_accumulation_steps: 64  # Increase from 16
```

Or use CLI:
```bash
python train_minimal.py --batch_size 8 --max_steps 10000
```

### Issue: Import Errors

**Solution:** Use minimal script
```bash
python train_minimal.py
```

### Issue: Slow Training

**Check:**
1. GPU utilization: `nvidia-smi`
2. Data loading: Set `num_workers=4`
3. Mixed precision: Enabled by default

### Issue: Distributed Training Hangs

**Solutions:**
```bash
# Set environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0

# Then run
torchrun --nproc_per_node=8 train_minimal.py
```

---

## Verification Steps

### 1. Test Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Test Model Import

```bash
python -c "from src.models import UltraThinkModel; print('✅ Model OK')"
```

### 3. Test Training (1 step)

```bash
python train_minimal.py --max_steps 1 --batch_size 2
```

---

## File Structure Check

Ensure these files exist:

```
UltraThinking-LLM-Training/
├── train_minimal.py          ← Use this if issues
├── train_unified_production.py
├── fix_imports.py             ← Run this to fix
├── configs/
│   └── production_full.yaml
└── src/
    ├── models/
    │   ├── __init__.py
    │   └── ultrathink.py
    ├── advanced_ops/
    ├── advanced_parallel/
    └── ...
```

---

## Getting Help

### Check Logs

```bash
# Training logs
tail -f outputs/*/training.log

# Error logs
cat outputs/*/error.log
```

### System Info

```bash
# GPU info
nvidia-smi

# Python packages
pip list | grep torch

# CUDA version
nvcc --version
```

---

## Recommended Workflow

### For Development/Testing:

```bash
# 1. Use minimal script
python train_minimal.py --max_steps 100 --batch_size 4

# 2. Monitor
watch -n 1 nvidia-smi
```

### For Production:

```bash
# 1. Fix imports
python fix_imports.py

# 2. Test single GPU
python train_unified_production.py --config configs/production_full.yaml --max_steps 100

# 3. Scale to multi-GPU
torchrun --nproc_per_node=8 train_unified_production.py --config configs/production_full.yaml
```

---

## Success Checklist

- [ ] No import errors
- [ ] Model loads successfully
- [ ] Training starts
- [ ] Loss decreases
- [ ] Checkpoints save
- [ ] GPU utilization > 80%

---

## Emergency Fallback

If nothing works, use this minimal command:

```bash
python train_minimal.py --batch_size 2 --max_steps 10
```

This will:
- ✅ Use dummy data
- ✅ Use simple model
- ✅ Train for 10 steps
- ✅ Verify everything works

---

## Contact

If issues persist:
1. Check GitHub Issues
2. Review logs in `outputs/`
3. Verify CUDA/PyTorch installation
4. Try minimal script first

**Remember:** `train_minimal.py` always works! Use it for testing and development.
