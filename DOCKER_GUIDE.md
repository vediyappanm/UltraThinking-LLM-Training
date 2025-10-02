# Docker Usage Guide

Complete guide for using ULTRATHINK with Docker.

## 🚀 Quick Start

### Run Web Interface (Default)
```bash
docker compose up app
# Visit http://localhost:7860
```

### Run Training (CPU)
```bash
docker compose --profile train up
```

### Run Training (GPU)
```bash
docker compose --profile train-gpu up
```

## 📦 Available Services

### 1. Web Interface (`app`)
**Purpose**: Interactive Gradio UI for model inference

```bash
# Start the web interface
docker compose up app

# Run in background
docker compose up -d app

# View logs
docker compose logs -f app
```

**Ports**:
- 7860: Gradio web interface
- 8000: FastAPI (if needed)

**Volumes**:
- `./outputs` - Model outputs
- `./checkpoints` - Model checkpoints

---

### 2. CPU Training (`train`)
**Purpose**: Train models on CPU (for testing/small models)

```bash
# Start training with default config
docker compose --profile train up

# Custom training command
docker compose run --rm train python train_ultrathink.py \
  --dataset wikitext \
  --hidden_size 512 \
  --num_layers 6 \
  --batch_size 2 \
  --num_epochs 3
```

**Example - WikiText Training**:
```bash
docker compose run --rm train python train_advanced.py \
  --config /app/configs/train_small.yaml
```

---

### 3. GPU Training (`train-gpu`)
**Purpose**: Train models with GPU acceleration

**Prerequisites**:
- NVIDIA GPU
- NVIDIA Docker runtime
- nvidia-container-toolkit

```bash
# Start GPU training
docker compose --profile train-gpu up

# Custom GPU training
docker compose run --rm train-gpu python train_ultrathink.py \
  --dataset c4 --streaming \
  --hidden_size 768 --num_layers 12 \
  --use_amp --gradient_checkpointing \
  --output_dir /app/outputs/c4_model
```

---

### 4. MLflow Tracking (`mlflow`)
**Purpose**: Experiment tracking and model registry

```bash
# Start MLflow server
docker compose --profile mlflow up -d

# Access UI
# http://localhost:5000
```

**Train with MLflow tracking**:
```bash
docker compose run --rm \
  --env MLFLOW_TRACKING_URI=http://mlflow:5000 \
  train python train_ultrathink.py \
    --use_mlflow \
    --dataset wikitext \
    --num_epochs 3
```

---

### 5. Development Environment (`dev`)
**Purpose**: Interactive development with all tools

```bash
# Start dev container
docker compose --profile dev run --rm dev

# Inside container:
pytest                  # Run tests
python train_ultrathink.py --help
jupyter notebook --ip 0.0.0.0 --port 8888
```

**Access Jupyter**:
- http://localhost:8888

---

## 🎯 Common Use Cases

### Use Case 1: Quick Demo
```bash
# Start web interface only
docker compose up app
# Visit http://localhost:7860
```

### Use Case 2: Training + Monitoring
```bash
# Start MLflow and training
docker compose --profile mlflow up -d
docker compose --profile train up
```

### Use Case 3: Full Development Stack
```bash
# Start all services
docker compose --profile dev --profile mlflow up
```

### Use Case 4: Production Training
```bash
# GPU training with checkpointing
docker compose run --rm train-gpu \
  python train_advanced.py \
    --config /app/configs/train_medium.yaml \
    --checkpoint_frequency 1000 \
    --output_dir /app/outputs/production_model
```

---

## 🔧 Advanced Usage

### Building Specific Stages

**Production image (minimal)**:
```bash
docker build --target production -t ultrathink:prod .
```

**Development image (with tools)**:
```bash
docker build --target development -t ultrathink:dev .
```

**Training image**:
```bash
docker build --target training -t ultrathink:train .
```

### Custom Environment Variables

Create `.env` file:
```env
WANDB_API_KEY=your_api_key_here
HF_TOKEN=your_hf_token_here
MLFLOW_TRACKING_URI=http://mlflow:5000
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Run with env file:
```bash
docker compose --env-file .env up
```

### Volume Mounting for Development

```bash
# Mount entire project (live editing)
docker run -it --rm \
  -v $(pwd):/app \
  ultrathink:dev bash
```

### Multi-GPU Training

```bash
# Use specific GPUs
docker compose run --rm \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  train-gpu python train_ultrathink.py \
    --distributed \
    --num_gpus 4
```

---

## 📊 Monitoring

### View Logs
```bash
# View all logs
docker compose logs

# Follow specific service
docker compose logs -f app

# Last 100 lines
docker compose logs --tail 100 train
```

### Check Resource Usage
```bash
# Container stats
docker stats

# Specific container
docker stats ultrathink_train_gpu
```

### Access Running Container
```bash
# Execute command in running container
docker compose exec app bash

# Run pytest in running container
docker compose exec app pytest
```

---

## 🧹 Cleanup

### Stop Services
```bash
# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Remove Images
```bash
# Remove project images
docker rmi ultrathink:latest ultrathink:training ultrathink:dev

# Prune unused images
docker image prune -a
```

### Clean Build Cache
```bash
docker builder prune -a
```

---

## 🐛 Troubleshooting

### Issue: GPU not detected
**Solution**:
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-container-toolkit if needed
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Issue: Out of memory
**Solution**:
```bash
# Reduce batch size
docker compose run --rm train python train_ultrathink.py \
  --batch_size 1 \
  --gradient_accumulation_steps 32

# Use gradient checkpointing
docker compose run --rm train python train_ultrathink.py \
  --gradient_checkpointing
```

### Issue: Slow training
**Solution**:
```bash
# Enable AMP and optimizations
docker compose run --rm train-gpu python train_ultrathink.py \
  --use_amp \
  --gradient_checkpointing \
  --use_flash_attention
```

### Issue: Container fails to start
**Solution**:
```bash
# Check logs
docker compose logs app

# Rebuild image
docker compose build --no-cache app

# Check disk space
docker system df
```

---

## 🔐 Security Best Practices

1. **Don't hardcode secrets** - Use environment variables
2. **Use .env file** - Keep secrets out of docker-compose.yml
3. **Limit port exposure** - Only expose necessary ports
4. **Use specific tags** - Avoid `latest` in production
5. **Scan images** - Use `docker scan ultrathink:latest`
6. **Non-root user** - Run containers as non-root (future enhancement)

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Main README](README.md)
- [Training Guide](TRAINING_QUICKSTART.md)

---

## 🎓 Examples

### Example 1: Complete Training Pipeline
```bash
# 1. Start MLflow
docker compose --profile mlflow up -d

# 2. Train model
docker compose run --rm train python train_ultrathink.py \
  --dataset wikitext \
  --use_mlflow \
  --num_epochs 5 \
  --output_dir /app/outputs/wikitext_model

# 3. Check MLflow UI
# Visit http://localhost:5000

# 4. Start inference UI
docker compose up app

# 5. Cleanup
docker compose down
```

### Example 2: Distributed Training
```bash
# Multi-GPU training with DeepSpeed
docker compose run --rm \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  train-gpu python train_ultrathink.py \
    --distributed \
    --deepspeed \
    --deepspeed_config /app/configs/deepspeed_z3.json \
    --dataset c4 --streaming
```

### Example 3: Development Workflow
```bash
# Start dev container
docker compose --profile dev run --rm dev bash

# Inside container:
# 1. Run tests
pytest

# 2. Train small model
python train_ultrathink.py --dataset wikitext --num_epochs 1

# 3. Profile performance
python scripts/profile_model.py --size tiny

# 4. Exit
exit
```

---

For more information, see the [main README](README.md) or [Advanced Training Guide](ADVANCED_TRAINING_GUIDE.md).
