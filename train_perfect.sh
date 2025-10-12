#!/bin/bash

# ============================================================================
# ULTRATHINK Perfect Training Script
# ============================================================================
# Fixes: Routing collapse, high perplexity, auxiliary loss issues
# Optimized for: Small-scale model (512 hidden, 6 layers) on single GPU
# Expected Results: Entropy 0.8-1.2, Max Expert <70%, Aux Loss 2-4
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ULTRATHINK Perfect Training${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if train_ultrathink.py exists
if [ ! -f "train_ultrathink.py" ]; then
    echo -e "${RED}Error: train_ultrathink.py not found!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: No GPU detected. Training will be slow on CPU.${NC}"
fi

# Create output directory
mkdir -p ./outputs/ultrathink_fixed

echo -e "${GREEN}Starting training with optimized configuration...${NC}"
echo ""

# ============================================================================
# MAIN TRAINING COMMAND - PERFECT CONFIGURATION
# ============================================================================

python train_ultrathink.py \
  `# === Model Architecture ===` \
  --vocab_size 50257 \
  --hidden_size 512 \
  --num_layers 6 \
  --num_heads 8 \
  --num_kv_heads 4 \
  --intermediate_size 2048 \
  --max_seq_length 256 \
  --activation swiglu \
  \
  `# === MoE Configuration (FIXED) ===` \
  --enable_moe \
  --num_knowledge_experts 4 \
  --num_skill_experts 2 \
  --num_meta_experts 1 \
  --num_safety_experts 1 \
  --moe_top_k 2 \
  --expert_capacity 1.5 \
  \
  `# === Load Balancing Weights (CRITICAL FIXES) ===` \
  --load_balance_weight 0.1 \
  --z_loss_weight 0.0001 \
  --importance_weight 0.05 \
  \
  `# === Training Hyperparameters (OPTIMIZED) ===` \
  --batch_size 2 \
  --gradient_accumulation_steps 32 \
  --learning_rate 0.0001 \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --warmup_steps 1000 \
  --max_steps 100000 \
  --num_epochs 1 \
  --gradient_clipping 0.5 \
  \
  `# === Regularization (INCREASED) ===` \
  --dropout 0.15 \
  --attention_dropout 0.15 \
  \
  `# === Optimization Features ===` \
  --gradient_checkpointing \
  --use_amp \
  --amp_warmup_steps 500 \
  \
  `# === Dynamic Reasoning Engine ===` \
  --enable_dre \
  --dre_warmup_steps 1000 \
  \
  `# === Dataset Configuration ===` \
  --dataset c4 \
  --dataset_subset en \
  --tokenizer_name gpt2 \
  --streaming \
  --train_samples 10000 \
  --val_samples 1000 \
  --num_workers 2 \
  \
  `# === Logging & Monitoring (ENHANCED) ===` \
  --use_mlflow \
  --mlflow_tracking_uri "file:./mlruns" \
  --mlflow_experiment "UltraThinking-LLM-Training" \
  --run_name "ultrathink_fixed_routing_v2" \
  --perf_log_interval 5 \
  --eval_frequency 50 \
  \
  `# === Output ===` \
  --output_dir "./outputs/ultrathink_fixed"

# ============================================================================
# Post-training
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Output directory: ./outputs/ultrathink_fixed"
    echo "MLflow logs: ./mlruns"
    echo ""
    echo "To view training metrics:"
    echo "  mlflow ui --backend-store-uri ./mlruns --port 5000"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the logs above for error details."
    exit 1
fi
