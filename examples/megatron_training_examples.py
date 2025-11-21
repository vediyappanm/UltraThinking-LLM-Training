"""
Megatron-LM Training Examples
Demonstrates how to use all Megatron features in UltraThinking-LLM-Training
"""

# Example 1: Tensor Parallel Training (2 GPUs)
# ============================================
# Windows (single process per GPU):
# python train_ultrathink.py --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2 --distributed

# Linux (torchrun):
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 2 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2


# Example 2: Pipeline Parallel Training (4 GPUs, 2 stages)
# =========================================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 4 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_pipeline_parallel --pipeline_parallel_size 2 --pipeline_schedule 1f1b --num_pipeline_microbatches 4


# Example 3: Tensor + Pipeline Parallel (8 GPUs: 2 TP x 4 PP)
# ============================================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 8 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2 --use_pipeline_parallel --pipeline_parallel_size 4


# Example 4: Expert Parallel MoE (4 GPUs)
# ========================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 4 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --enable_moe --use_expert_parallel --expert_parallel_size 4


# Example 5: Full 4D Parallelism (16 GPUs: 2 DP x 2 TP x 2 PP x 2 EP)
# ====================================================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 16 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --data_parallel_size 2 --use_megatron_tp --tensor_parallel_size 2 --use_pipeline_parallel --pipeline_parallel_size 2 --use_expert_parallel --expert_parallel_size 2


# Example 6: Fused Operations + FP8 (Transformer Engine)
# =======================================================
# Requires: pip install transformer-engine
# python train_ultrathink.py --config configs/moe_dre_minimal.yaml --use_fused_ops --use_fp8 --use_fused_optimizer


# Example 7: Sequence Parallelism (reduces activation memory)
# ============================================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 2 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2 --use_sequence_parallel


# Example 8: Distributed Checkpointing
# =====================================
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 4 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2 --use_distributed_checkpoint


# Example 9: DeepSpeed Launch (Linux)
# ====================================
# python scripts/launch_distributed.py --launcher deepspeed --num_gpus 4 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --deepspeed configs/deepspeed_config.json


# Example 10: Multi-Node Training (2 nodes, 4 GPUs each)
# =======================================================
# Node 0:
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 4 --num_nodes 2 --node_rank 0 --master_addr <node0_ip> --master_port 29500 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2

# Node 1:
# python scripts/launch_distributed.py --launcher torchrun --num_gpus 4 --num_nodes 2 --node_rank 1 --master_addr <node0_ip> --master_port 29500 --script train_ultrathink.py -- --config configs/moe_dre_minimal.yaml --use_megatron_tp --tensor_parallel_size 2


# YAML Configuration Examples
# ============================

# Example YAML with Megatron features (save as configs/megatron_full.yaml):
MEGATRON_FULL_CONFIG = """
seed: 42

model_config_dict:
  vocab_size: 50257
  n_positions: 2048
  n_embd: 1024
  n_layer: 24
  n_head: 16
  n_kv_head: 4
  rotary_dim: 64
  attention_type: standard
  use_megatron_tp: true
  tensor_parallel_size: 2

training:
  batch_size: 64
  micro_batch_size: 2
  gradient_accumulation_steps: 32
  epochs: 3
  max_steps: 10000
  learning_rate: 2.0e-4
  warmup_steps: 500
  mixed_precision: bf16
  max_grad_norm: 1.0
  use_fused_ops: true
  use_fused_optimizer: true

data:
  tokenizer: gpt2
  dataset: wikitext
  subset: wikitext-2-raw-v1
  text_column: text
  streaming: false

ultrathink:
  enable_moe: true
  enable_dre: true
  moe_layers: [4, 8, 12, 16, 20]
  moe_config:
    num_knowledge_experts: 16
    num_skill_experts: 8
    num_meta_experts: 4
    num_safety_experts: 2
    top_k: 2
    expert_dropout: 0.1
    capacity_factor: 1.25
    load_balance_weight: 0.01

logging:
  log_interval: 10
  eval_interval: 100
  tensorboard: true
  mlflow:
    enable: true

checkpointing:
  output_dir: ./checkpoints_megatron
  save_interval: 500
  use_distributed_checkpoint: true
"""

# Windows-Specific Notes
# ======================
# 1. Tensor parallel requires multi-process: use torchrun or launch_distributed.py
# 2. Set USE_LIBUV=0 if you encounter libuv errors
# 3. NCCL not available on Windows; use gloo backend (automatic fallback)
# 4. DeepSpeed has limited Windows support; prefer torchrun
# 5. For single GPU testing, disable all parallelism flags

# Linux-Specific Optimizations
# =============================
# 1. Install Apex for fused ops: pip install git+https://github.com/NVIDIA/apex.git
# 2. Install Transformer Engine for FP8: pip install transformer-engine
# 3. Install flash-attn: pip install flash-attn --no-build-isolation
# 4. Use NCCL backend for best multi-GPU performance
# 5. Set NCCL_DEBUG=INFO for debugging communication issues

# Performance Tips
# ================
# 1. Tensor Parallel: Best for large models that don't fit on single GPU
# 2. Pipeline Parallel: Reduces memory per GPU but adds communication overhead
# 3. Sequence Parallel: Reduces activation memory when using TP
# 4. Expert Parallel: Essential for large MoE models
# 5. Fused Ops: 10-20% speedup with Apex/TE
# 6. FP8: 2x speedup on H100 GPUs with Transformer Engine
# 7. Distributed Checkpointing: Faster save/load for large models

# Troubleshooting
# ===============
# 1. OOM errors: Increase pipeline stages, reduce micro batch size, enable gradient checkpointing
# 2. Slow training: Check if pipeline bubbles are large, reduce num_microbatches
# 3. Divergence: Reduce learning rate, increase warmup steps, disable FP8
# 4. Communication hangs: Check NCCL_DEBUG output, verify network connectivity
# 5. Windows libuv error: Set USE_LIBUV=0 environment variable

# Recommended Configurations by Model Size
# =========================================

# Small (1B params, 4 GPUs):
# - TP=2, PP=1, DP=2
# - Micro batch=4, Grad accum=8
# - No FP8, standard precision

# Medium (7B params, 8 GPUs):
# - TP=2, PP=2, DP=2
# - Micro batch=2, Grad accum=16
# - Fused ops, bf16

# Large (70B params, 32 GPUs):
# - TP=4, PP=4, DP=2
# - Micro batch=1, Grad accum=32
# - Fused ops, FP8 (if H100)

# MoE (1T params, 64 GPUs):
# - TP=2, PP=4, EP=4, DP=2
# - Micro batch=1, Grad accum=64
# - Expert parallel essential
# - Distributed checkpointing
