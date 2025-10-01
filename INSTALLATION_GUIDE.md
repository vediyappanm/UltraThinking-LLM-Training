# ULTRATHINK Installation Guide

Complete installation guide for the ULTRATHINK LLM training framework.

## Quick Install

```bash
# Clone repository
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training/deep

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Detailed Installation

### 1. Prerequisites

#### System Requirements
- **Python**: 3.9 - 3.12
- **OS**: Linux, macOS, Windows 10/11
- **RAM**: 16GB minimum, 32GB+ recommended
- **Storage**: 50GB+ free space

#### For GPU Training
- **NVIDIA GPU**: Compute Capability 7.0+ (V100, A100, RTX 30/40 series)
- **CUDA**: 11.8 or 12.1
- **cuDNN**: Latest version for your CUDA
- **VRAM**: 8GB minimum, 40GB+ for large models

### 2. Environment Setup

#### Option A: Conda (Recommended for GPU)

```bash
# Create conda environment
conda create -n ultrathink python=3.10
conda activate ultrathink

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Option B: venv (CPU or existing CUDA)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch
# For GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run smoke test
python -m tests.smoke_test

# Run unit tests
pytest tests/unit -v
```

### 4. Optional Components

#### DeepSpeed (for distributed training)
```bash
pip install deepspeed>=0.12.0
```

#### Flash Attention (for faster attention)
```bash
# Requires CUDA and correct GPU arch
pip install flash-attn --no-build-isolation
```

#### Apex (NVIDIA optimizations)
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

### 5. Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### 6. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or your preferred editor

# Required settings:
# - WANDB_API_KEY (if using W&B)
# - CUDA_VISIBLE_DEVICES (which GPUs to use)
# - Output directories
```

### 7. Verify GPU Setup (if applicable)

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Run GPU smoke test
python -c "import torch; x=torch.randn(1000,1000).cuda(); print('GPU test passed')"
```

## Common Installation Issues

### Issue: PyTorch not finding CUDA

**Solution:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Flash Attention compilation fails

**Solution:**
```bash
# Flash Attention requires specific GPU architecture
# Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus
# If older GPU, disable flash attention:
# Set flash_attention=False in model config
```

### Issue: DeepSpeed installation fails on Windows

**Solution:**
```bash
# Windows users: DeepSpeed has limited Windows support
# Use WSL2 for full DeepSpeed functionality
# Or use Accelerate instead:
pip install accelerate
```

### Issue: Out of memory during installation

**Solution:**
```bash
# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Or install packages individually
pip install torch torchvision torchaudio
pip install transformers datasets
# ... etc
```

### Issue: Pre-commit hooks failing

**Solution:**
```bash
# Update pre-commit
pip install --upgrade pre-commit

# Clear cache
pre-commit clean

# Reinstall hooks
pre-commit install
```

## Platform-Specific Notes

### Linux

Most straightforward installation. All features supported.

### macOS

**M1/M2/M3 (Apple Silicon):**
```bash
# Use PyTorch with Metal Performance Shaders
pip install torch torchvision torchaudio
```

**Intel Mac:**
```bash
# CPU only or external GPU
pip install torch torchvision torchaudio
```

**Note**: Limited GPU acceleration support.

### Windows

**Recommended**: Use WSL2 for full functionality.

**Native Windows**:
- DeepSpeed support is limited
- Some features may not work (Flash Attention)
- Set `num_workers=0` for DataLoader

**WSL2 Setup**:
```bash
# Install WSL2 with Ubuntu
wsl --install

# Follow Linux installation steps inside WSL2
```

## Docker Installation

### CPU-only
```bash
docker build -t ultrathink:latest .
docker run -it ultrathink:latest
```

### GPU (NVIDIA Docker)
```bash
# Requires nvidia-docker2
docker build -f Dockerfile.gpu -t ultrathink:gpu .
docker run --gpus all -it ultrathink:gpu
```

### Docker Compose
```bash
docker compose up --build
```

## Cloud Platform Setup

### Google Colab

See [colab.md](colab.md) for Colab-specific setup.

### AWS

```bash
# Launch EC2 instance with Deep Learning AMI
# Connect to instance
ssh -i key.pem ubuntu@instance-ip

# Clone and install
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training/deep
pip install -r requirements.txt
```

### Azure

```bash
# Use Azure ML or NC-series VMs
# Similar to AWS setup
```

### Vast.ai / Lambda Labs

```bash
# Instances usually come with PyTorch pre-installed
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training/deep
pip install -r requirements.txt
```

## Minimal Installation (CPU-only, for testing)

```bash
# Minimal dependencies for testing
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets
pip install pytest

# Run tests
pytest tests/unit
```

## Next Steps

After installation:

1. **Verify setup**: Run `python -m tests.smoke_test`
2. **Configure environment**: Edit `.env` file
3. **Run profiler**: `python scripts/profile_model.py --size tiny`
4. **Try training**: See [training_small.md](docs/training_small.md)
5. **Read docs**: See [docs/README.md](docs/README.md)

## Getting Help

- **Installation issues**: [GitHub Issues](https://github.com/vediyappanm/UltraThinking-LLM-Training/issues)
- **Questions**: [GitHub Discussions](https://github.com/vediyappanm/UltraThinking-LLM-Training/discussions)
- **Documentation**: [docs/](docs/)

## Updating

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Run tests
pytest
```
