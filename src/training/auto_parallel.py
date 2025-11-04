"""
Auto Parallel Planner for 4D Parallelism

Plans data/tensor/pipeline/sequence parallelism based on:
- Model parameter count
- Available GPUs and memory
- Target global batch size and context length

Produces a DistributedConfig (from training.distributed_4d) and optional DeepSpeed config dict.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Dict, Any

import torch

from .distributed_4d import DistributedConfig


def estimate_params(model: torch.nn.Module) -> int:
    """Rough parameter count for planning."""
    return sum(p.numel() for p in model.parameters())


def detect_num_gpus() -> int:
    if torch.cuda.is_available():
        try:
            return torch.cuda.device_count()
        except Exception:
            return 1
    return 0


def plan_parallelism(
    model: torch.nn.Module,
    global_batch_size: int = 128,
    sequence_length: int = 4096,
    prefer_tensor_parallel: bool = True,
    prefer_pipeline_parallel: bool = True,
) -> DistributedConfig:
    """
    Compute a simple, robust parallelism plan.

    Heuristics:
    - 0-2B params: DP only
    - 2-20B params: TP if multi-GPU, optional PP
    - 20B-70B: DP + TP + (optional PP)
    - >70B: combine DP + TP + PP; enable sequence parallel
    """
    n_params = estimate_params(model)
    n_gpus = max(1, detect_num_gpus())

    # Defaults
    dp = min(n_gpus, max(1, global_batch_size // 32))
    tp = 1
    pp = 1
    sp = 1

    if n_params <= 2_000_000_000:  # ~2B
        dp = n_gpus
    elif n_params <= 20_000_000_000:  # ~20B
        if prefer_tensor_parallel:
            tp = min(n_gpus, 2 if n_gpus >= 2 else 1)
        dp = max(1, n_gpus // tp)
    elif n_params <= 70_000_000_000:  # ~70B
        if prefer_tensor_parallel:
            tp = min(4, n_gpus)
        if prefer_pipeline_parallel and n_gpus // tp >= 2:
            pp = min(2, n_gpus // tp)
        dp = max(1, n_gpus // (tp * pp))
    else:  # >70B
        if prefer_tensor_parallel:
            tp = min(8, n_gpus)
        if prefer_pipeline_parallel and n_gpus // tp >= 2:
            pp = min(4, max(2, n_gpus // tp))
        dp = max(1, n_gpus // (tp * pp))
        sp = 2 if sequence_length > 8192 else 1

    cfg = DistributedConfig(
        data_parallel_size=max(1, dp),
        tensor_parallel_size=max(1, tp),
        pipeline_parallel_size=max(1, pp),
        expert_parallel_size=1,
        enable_sequence_parallel=(sp > 1),
        sequence_parallel_size=sp,
        mixed_precision="bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16",
        zero_stage=3 if n_params >= 20_000_000_000 else 2,
        offload_optimizer=(n_params >= 70_000_000_000),
        gradient_accumulation_steps=max(1, (global_batch_size // max(1, dp)) // 2),
    )
    return cfg


def make_deepspeed_config(cfg: DistributedConfig) -> Dict[str, Any]:
    """Generate a basic DeepSpeed config dict from DistributedConfig."""
    return {
        "train_batch_size": 1,  # filled by trainer
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "fp16": {"enabled": cfg.mixed_precision == "fp16"},
        "bf16": {"enabled": cfg.mixed_precision == "bf16"},
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "offload_param": {"device": "cpu", "pin_memory": True} if cfg.offload_param else {"device": "none"},
            "offload_optimizer": {"device": "cpu", "pin_memory": True} if cfg.offload_optimizer else {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": cfg.cpu_offload,
        },
    }
