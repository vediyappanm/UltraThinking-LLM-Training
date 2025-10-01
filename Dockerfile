# ULTRATHINK - CPU Dockerfile
# Lightweight image for running the demo app or CPU-only training

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (audio/image libs, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project
COPY . .

# Default ports
EXPOSE 7860 8000

# Default command: run Gradio app (change as needed)
CMD ["python", "app_gradio.py"]
