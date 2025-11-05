#!/bin/bash
# UltraThinking Production Training Launcher
# Integrates all advanced features for maximum performance

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Number of nodes and GPUs
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

# Configuration file
CONFIG_FILE=${CONFIG_FILE:-configs/production_full.yaml}

# Output directory
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/production_run_$(date +%Y%m%d_%H%M%S)}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "========================================="
echo "UltraThinking Production Training"
echo "========================================="
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $((NUM_NODES * NUM_GPUS_PER_NODE))"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $CONFIG_FILE $OUTPUT_DIR/config.yaml

# ============================================================================
# LAUNCH TRAINING
# ============================================================================

# Option 1: Single Node Training
if [ $NUM_NODES -eq 1 ]; then
    echo "Launching single-node training..."
    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        train_unified_production.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR

# Option 2: Multi-Node Training
else
    echo "Launching multi-node training..."
    
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --max_restarts=10 \
        train_unified_production.py \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR
fi

echo "========================================="
echo "Training Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
