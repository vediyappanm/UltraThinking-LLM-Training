#!/usr/bin/env python3
"""
Distributed Training Script for Multi-GPU/Multi-Node Training
Supports FSDP, DeepSpeed, and DDP
"""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.architecture import AdvancedGPTModel, ModelConfig, TransformerBlock

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not running in distributed mode")
        return 0, 1, 0
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_fsdp_model(model, config):
    """Setup FSDP wrapped model"""
    # Auto wrap policy for transformer blocks
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={TransformerBlock},
    )
    
    # Mixed precision policy
    from torch.distributed.fsdp import MixedPrecision
    if config['training']['mixed_precision'] == 'bf16':
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif config['training']['mixed_precision'] == 'fp16':
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=None,
    )
    
    return model


def setup_deepspeed_model(model, config, optimizer=None):
    """Setup DeepSpeed model"""
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed not available")
    
    deepspeed_config = config['training'].get('deepspeed_config')
    if deepspeed_config and os.path.exists(deepspeed_config):
        with open(deepspeed_config, 'r') as f:
            ds_config = yaml.safe_load(f)
    else:
        # Default DeepSpeed config
        ds_config = {
            "train_batch_size": config['training']['batch_size'],
            "train_micro_batch_size_per_gpu": config['training'].get('micro_batch_size', 1),
            "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "fp16": {"enabled": config['training']['mixed_precision'] == 'fp16'},
            "bf16": {"enabled": config['training']['mixed_precision'] == 'bf16'},
        }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config
    )
    
    return model_engine, optimizer


def main():
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--backend", type=str, choices=['fsdp', 'deepspeed', 'ddp'], default='fsdp')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    print(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model_config = ModelConfig(**config['model_config_dict'])
    model = AdvancedGPTModel(model_config)
    model = model.cuda(local_rank)
    
    # Setup distributed model
    if args.backend == 'fsdp':
        model = setup_fsdp_model(model, config)
        print("Using FSDP")
    elif args.backend == 'deepspeed':
        # Note: DeepSpeed initialization happens in the training script
        print("Using DeepSpeed")
    elif args.backend == 'ddp':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        print("Using DDP")
    
    print(f"Model setup complete on rank {rank}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
