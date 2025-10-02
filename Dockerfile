# ULTRATHINK - Multi-stage Dockerfile
# Supports both CPU and GPU training/inference

# ============================================
# Stage 1: Base image with dependencies
# ============================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================
# Stage 2: Development image
# ============================================
FROM base AS development

# Install development tools
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

COPY . .

# ============================================
# Stage 3: Production image (minimal)
# ============================================
FROM base AS production

# Copy only necessary files
COPY src/ ./src/
COPY train_ultrathink.py train_advanced.py app_gradio.py ./
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create directories for outputs
RUN mkdir -p /app/outputs /app/checkpoints

# Expose ports
EXPOSE 7860 8000 5000

# Default command: run Gradio app
CMD ["python", "app_gradio.py"]

# ============================================
# Stage 4: Training image
# ============================================
FROM production AS training

# Copy test data and utilities
COPY tests/ ./tests/

# Default command: show help
CMD ["python", "train_ultrathink.py", "--help"]
