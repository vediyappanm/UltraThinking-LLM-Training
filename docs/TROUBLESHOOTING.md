# 🔧 Troubleshooting Guide

Common issues and solutions for ULTRATHINK training.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Training Errors](#training-errors)
- [Memory Issues](#memory-issues)
- [Performance Problems](#performance-problems)
- [Data Loading Issues](#data-loading-issues)
- [Distributed Training](#distributed-training)
- [Monitoring & Logging](#monitoring--logging)
- [Docker Issues](#docker-issues)

---

## Installation Issues

### ❌ `ImportError: No module named 'flash_attn'`

**Cause**: Flash Attention 2 not installed or incompatible with your CUDA version.

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install Flash Attention 2 (requires CUDA 11.6+)
pip install flash-attn --no-build-isolation

# If build fails, disable Flash Attention
python train_ultrathink.py --no_flash_attention
```

**Alternative**: Use PyTorch's native SDPA:
```python
# In your config
model:
  use_flash_attention: false
  use_sdpa: true  # PyTorch 2.0+ scaled dot product attention
```

---

### ❌ `CUDA out of memory` during installation

**Cause**: Trying to build packages that require GPU memory.

**Solution**:
```bash
# Set environment variable to reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or install CPU-only first, then GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### ❌ `ModuleNotFoundError: No module named 'src'`

**Cause**: Python can't find the `src` module.

**Solution**:
```bash
# Make sure you're in the correct directory
cd UltraThinking-LLM-Training/deep

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Training Errors

### ❌ `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and data are on different devices (CPU vs GPU).

**Solution**:
```python
# Ensure device consistency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
```

**In training script**:
```bash
# Force CPU training
python train_ultrathink.py --device cpu

# Force GPU training
python train_ultrathink.py --device cuda
```

---

### ❌ `NaN loss` or `loss becomes inf`

**Cause**: Gradient explosion, learning rate too high, or numerical instability.

**Solutions**:

1. **Reduce learning rate**:
```bash
python train_ultrathink.py --learning_rate 1e-4  # Instead of 3e-4
```

2. **Enable gradient clipping**:
```bash
python train_ultrathink.py --gradient_clip_norm 1.0
```

3. **Use mixed precision carefully**:
```bash
# Try FP32 first
python train_ultrathink.py --no_amp

# Or use BF16 if supported (A100, H100)
python train_ultrathink.py --use_bf16
```

4. **Check for bad data**:
```python
# Add validation in data loading
def validate_batch(batch):
    for k, v in batch.items():
        if torch.isnan(v).any() or torch.isinf(v).any():
            raise ValueError(f"Invalid values in {k}")
    return batch
```

5. **Reduce batch size**:
```bash
python train_ultrathink.py --batch_size 1 --gradient_accumulation_steps 8
```

---

### ❌ `ValueError: Tokenizer not found`

**Cause**: Tokenizer not downloaded or path incorrect.

**Solution**:
```bash
# Download tokenizer manually
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

# Or specify tokenizer path
python train_ultrathink.py --tokenizer_name gpt2

# Use local tokenizer
python train_ultrathink.py --tokenizer_path ./my_tokenizer
```

---

### ❌ `AssertionError: hidden_size must be divisible by num_heads`

**Cause**: Invalid model configuration.

**Solution**:
```bash
# Ensure hidden_size is divisible by num_heads
# Example: hidden_size=768, num_heads=12 ✅
# Example: hidden_size=768, num_heads=10 ❌

python train_ultrathink.py --hidden_size 768 --num_heads 12

# Common valid combinations:
# 256 / 4 = 64
# 512 / 8 = 64
# 768 / 12 = 64
# 1024 / 16 = 64
```

---

## Memory Issues

### ❌ `CUDA out of memory` during training

**Immediate fixes** (try in order):

1. **Reduce batch size**:
```bash
python train_ultrathink.py --batch_size 1 --gradient_accumulation_steps 16
```

2. **Enable gradient checkpointing**:
```bash
python train_ultrathink.py --gradient_checkpointing
```

3. **Use mixed precision**:
```bash
python train_ultrathink.py --use_amp
```

4. **Reduce sequence length**:
```bash
python train_ultrathink.py --max_seq_length 512  # Instead of 2048
```

5. **Use DeepSpeed ZeRO**:
```bash
python train_ultrathink.py --use_deepspeed --deepspeed_config deepspeed_config_zero2.json
```

**Memory optimization checklist**:
```python
# In your training script
import torch

# Clear cache before training
torch.cuda.empty_cache()

# Enable memory efficient attention
model.config.use_flash_attention = True

# Reduce optimizer memory (use 8-bit Adam)
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(model.parameters())

# Enable CPU offloading (slower but uses less GPU memory)
model.enable_cpu_offload()
```

---

### ❌ Memory leak (memory keeps increasing)

**Cause**: Accumulating gradients, keeping references to tensors, or logging too much.

**Solutions**:

1. **Clear gradients properly**:
```python
optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
```

2. **Detach metrics**:
```python
loss_value = loss.detach().item()  # Don't keep computation graph
```

3. **Limit logging**:
```python
if step % 100 == 0:  # Log every 100 steps, not every step
    logger.log_metrics({"loss": loss.item()})
```

4. **Clear cache periodically**:
```python
if step % 1000 == 0:
    torch.cuda.empty_cache()
```

---

### ❌ `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`

**Cause**: CUDA/cuDNN version mismatch or insufficient GPU memory.

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or disable cuDNN
export CUDA_VISIBLE_DEVICES=0
export CUDNN_ENABLED=0
python train_ultrathink.py
```

---

## Performance Problems

### 🐌 Training is very slow

**Diagnosis**:
```python
# Add profiling to find bottleneck
python scripts/profile_model.py --size small

# Check GPU utilization
nvidia-smi -l 1  # Update every second
```

**Common causes and fixes**:

1. **CPU bottleneck (data loading)**:
```bash
# Increase data loading workers
python train_ultrathink.py --num_workers 4 --prefetch_factor 2
```

2. **Small batch size**:
```bash
# Increase effective batch size
python train_ultrathink.py --batch_size 8 --gradient_accumulation_steps 4
```

3. **Not using Flash Attention**:
```bash
pip install flash-attn --no-build-isolation
python train_ultrathink.py --use_flash_attention
```

4. **Logging too frequently**:
```bash
python train_ultrathink.py --log_interval 100  # Instead of 10
```

5. **Slow storage**:
```bash
# Use local SSD instead of network storage
# Copy dataset to local disk first
cp -r /network/dataset /local/ssd/dataset
python train_ultrathink.py --dataset_path /local/ssd/dataset
```

---

### 🐌 Low GPU utilization (<50%)

**Causes**:
- Data loading bottleneck
- Small batch size
- CPU preprocessing too slow

**Solutions**:
```bash
# Increase workers and prefetch
python train_ultrathink.py --num_workers 8 --prefetch_factor 4

# Use streaming datasets (no preprocessing)
python train_ultrathink.py --dataset c4 --streaming

# Increase batch size
python train_ultrathink.py --batch_size 16

# Pin memory for faster transfer
python train_ultrathink.py --pin_memory
```

---

## Data Loading Issues

### ❌ `ConnectionError: Couldn't reach the Hugging Face Hub`

**Cause**: Network issues or HF Hub down.

**Solution**:
```bash
# Use cached dataset
export HF_DATASETS_OFFLINE=1
python train_ultrathink.py --dataset wikitext

# Or download dataset manually
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-v1')"

# Use local dataset
python train_ultrathink.py --dataset_path ./my_local_dataset
```

---

### ❌ `Dataset is too large for disk`

**Solution**: Use streaming mode
```bash
python train_ultrathink.py --dataset c4 --streaming
```

**Or**: Use a smaller subset
```bash
python train_ultrathink.py --dataset c4 --max_samples 100000
```

---

### ❌ `KeyError: 'text'` or wrong dataset format

**Cause**: Dataset doesn't have expected column names.

**Solution**:
```python
# Check dataset structure
from datasets import load_dataset
dataset = load_dataset("your_dataset")
print(dataset.column_names)

# Map to correct format
def preprocess(examples):
    return {"text": examples["content"]}  # Rename column

dataset = dataset.map(preprocess)
```

**In training script**:
```bash
python train_ultrathink.py --text_column content  # Instead of default "text"
```

---

## Distributed Training

### ❌ `RuntimeError: Address already in use`

**Cause**: Previous training process still running or port conflict.

**Solution**:
```bash
# Kill previous processes
pkill -f train_ultrathink.py

# Use different port
python train_ultrathink.py --master_port 29501

# Or let system choose port
python train_ultrathink.py --master_port 0
```

---

### ❌ Multi-GPU training hangs or crashes

**Diagnosis**:
```bash
# Test NCCL communication
python -m torch.distributed.run --nproc_per_node=2 scripts/test_distributed.py

# Check NCCL debug info
export NCCL_DEBUG=INFO
python train_ultrathink.py
```

**Common fixes**:

1. **Set correct backend**:
```bash
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

2. **Use correct launcher**:
```bash
# Use torchrun (recommended)
torchrun --nproc_per_node=2 train_ultrathink.py

# Or accelerate
accelerate launch --num_processes=2 train_ultrathink.py
```

3. **Increase timeout**:
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

---

### ❌ `RuntimeError: NCCL error: unhandled system error`

**Solution**:
```bash
# Disable peer-to-peer access
export NCCL_P2P_DISABLE=1

# Use different NCCL backend
export NCCL_SOCKET_IFNAME=lo  # Loopback for single node

# Check GPU topology
nvidia-smi topo -m
```

---

## Monitoring & Logging

### ❌ MLflow UI not starting

**Solution**:
```bash
# Check if port is in use
lsof -i :5000

# Use different port
mlflow ui --port 5001

# Or specify host
mlflow ui --host 0.0.0.0 --port 5000
```

---

### ❌ Weights & Biases not logging

**Solution**:
```bash
# Login to W&B
wandb login

# Check API key
echo $WANDB_API_KEY

# Disable W&B if not needed
export WANDB_MODE=disabled
python train_ultrathink.py
```

---

### ❌ TensorBoard shows no data

**Solution**:
```bash
# Check log directory
ls -la ./runs

# Start TensorBoard with correct path
tensorboard --logdir ./runs --port 6006

# Force reload
tensorboard --logdir ./runs --reload_interval 5
```

---

## Docker Issues

### ❌ `docker: permission denied`

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or use sudo
sudo docker compose up
```

---

### ❌ Container runs out of memory

**Solution**:
```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Or use docker run with memory limit
docker run --memory=16g --gpus all ultrathink:latest
```

---

### ❌ GPU not available in Docker

**Solution**:
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Run with GPU
docker run --gpus all ultrathink:latest
```

---

## Getting Help

If you can't find a solution here:

1. **Check existing issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)
2. **Search discussions**: [GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)
3. **Enable debug logging**:
```bash
export LOG_LEVEL=DEBUG
python train_ultrathink.py --verbose
```
4. **Create a minimal reproduction**:
```bash
python train_ultrathink.py \
  --hidden_size 256 --num_layers 2 \
  --batch_size 1 --max_steps 10 \
  --dataset wikitext --max_samples 100
```
5. **Open an issue** with:
   - Error message and full traceback
   - Your configuration (model size, hardware, etc.)
   - Steps to reproduce
   - Output of `python --version`, `torch.__version__`, `nvidia-smi`

---

## Debugging Checklist

Before opening an issue, try:

- [ ] Update to latest version: `git pull && pip install -r requirements.txt`
- [ ] Clear cache: `rm -rf ~/.cache/huggingface`
- [ ] Test with minimal config: `--hidden_size 256 --num_layers 2 --batch_size 1`
- [ ] Check GPU: `nvidia-smi`
- [ ] Check disk space: `df -h`
- [ ] Check CUDA version: `nvcc --version`
- [ ] Run tests: `pytest tests/`
- [ ] Enable verbose logging: `--verbose`

---

**Last Updated**: January 2025  
**Version**: 1.0.0

Found a solution not listed here? [Contribute to this guide!](CONTRIBUTING.md)
