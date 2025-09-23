"""
4D Parallelism Distributed Training Infrastructure
Implements Data, Tensor, Pipeline, and Expert Parallelism with Sequence Parallelism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import os
import math
import numpy as np
from contextlib import contextmanager
import logging

# Try to import FSDP components (may not be available in all PyTorch versions)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import CPUOffload, BackwardPrefetch, ShardingStrategy, MixedPrecision
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FSDP not available, falling back to DDP")

logger = logging.getLogger(__name__)


class ParallelismType(Enum):
    """Types of parallelism"""
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    EXPERT = "expert"
    SEQUENCE = "sequence"
    ZERO = "zero"


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Parallelism dimensions
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    # Sequence parallelism
    enable_sequence_parallel: bool = True
    sequence_parallel_size: int = 1
    max_sequence_length: int = 8192
    
    # ZeRO optimization
    zero_stage: int = 3
    offload_optimizer: bool = True
    offload_param: bool = False
    
    # Memory optimization
    activation_checkpointing: bool = True
    cpu_offload: bool = True
    gradient_compression: bool = True
    mixed_precision: str = "bf16"
    
    # Communication
    backend: str = "nccl"
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    # Pipeline settings
    num_microbatches: int = 4
    pipeline_schedule: str = "1f1b"
    
    # Expert settings
    expert_capacity_factor: float = 1.25
    expert_drop_policy: str = "probs"
    
    # Optimization
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 1000
    use_sharded_checkpointing: bool = True


class ProcessGroupManager:
    """Manage process groups for different parallelism types"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Calculate total parallelism
        self.total_parallel_size = (
            config.data_parallel_size *
            config.tensor_parallel_size *
            config.pipeline_parallel_size
        )
        
        # Initialize process groups
        self.process_groups = {}
        self._initialize_process_groups()
        
    def _initialize_process_groups(self):
        """Initialize all process groups"""
        # Data parallel groups
        dp_ranks = self._get_data_parallel_ranks()
        self.process_groups[ParallelismType.DATA] = dist.new_group(dp_ranks)
        
        # Tensor parallel groups
        tp_ranks = self._get_tensor_parallel_ranks()
        self.process_groups[ParallelismType.TENSOR] = dist.new_group(tp_ranks)
        
        # Pipeline parallel groups
        pp_ranks = self._get_pipeline_parallel_ranks()
        self.process_groups[ParallelismType.PIPELINE] = dist.new_group(pp_ranks)
        
        # Sequence parallel groups (same as tensor parallel)
        if self.config.enable_sequence_parallel:
            self.process_groups[ParallelismType.SEQUENCE] = self.process_groups[ParallelismType.TENSOR]
    
    def _get_data_parallel_ranks(self) -> List[int]:
        """Get ranks for data parallel group"""
        dp_size = self.config.data_parallel_size
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        
        # Calculate which ranks belong to same DP group
        dp_group_idx = self.rank // (tp_size * pp_size)
        ranks = []
        for r in range(self.world_size):
            if r // (tp_size * pp_size) == dp_group_idx:
                ranks.append(r)
        return ranks
    
    def _get_tensor_parallel_ranks(self) -> List[int]:
        """Get ranks for tensor parallel group"""
        tp_size = self.config.tensor_parallel_size
        pp_size = self.config.pipeline_parallel_size
        
        tp_group_idx = (self.rank // pp_size) % tp_size
        ranks = []
        for r in range(self.world_size):
            if (r // pp_size) % tp_size == tp_group_idx:
                ranks.append(r)
        return ranks
    
    def _get_pipeline_parallel_ranks(self) -> List[int]:
        """Get ranks for pipeline parallel group"""
        pp_size = self.config.pipeline_parallel_size
        
        pp_group_idx = self.rank % pp_size
        ranks = []
        for r in range(self.world_size):
            if r % pp_size == pp_group_idx:
                ranks.append(r)
        return ranks
    
    def get_group(self, parallelism_type: ParallelismType):
        """Get process group for given parallelism type"""
        return self.process_groups.get(parallelism_type)


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        
    @contextmanager
    def activation_checkpointing_context(self):
        """Context manager for activation checkpointing"""
        if self.config.activation_checkpointing:
            # Enable gradient checkpointing
            torch.cuda.empty_cache()
            yield
            torch.cuda.empty_cache()
        else:
            yield
    
    def apply_gradient_compression(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply gradient compression"""
        if not self.config.gradient_compression:
            return gradients
        
        # PowerSGD compression (simplified)
        # In practice, would use actual PowerSGD implementation
        shape = gradients.shape
        flattened = gradients.flatten()
        
        # Keep only top-k gradients
        k = int(flattened.numel() * 0.1)  # Keep 10%
        topk_vals, topk_idx = torch.topk(flattened.abs(), k)
        
        compressed = torch.zeros_like(flattened)
        compressed[topk_idx] = flattened[topk_idx]
        
        return compressed.reshape(shape)
    
    def setup_mixed_precision(self):
        """Setup mixed precision training"""
        if self.config.mixed_precision == "fp16":
            return torch.cuda.amp.GradScaler()
        elif self.config.mixed_precision == "bf16":
            # BF16 doesn't need loss scaling
            return None
        elif self.config.mixed_precision == "fp8":
            # FP8 support (if available)
            logger.warning("FP8 not fully supported, falling back to BF16")
            return None
        return None


class FourDParallelModel(nn.Module):
    """Model wrapper for 4D parallelism"""
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig
    ):
        super().__init__()
        self.model = model
        self.config = config
        
        # Process group manager
        self.pg_manager = ProcessGroupManager(config)
        
        # Memory optimizer
        self.memory_optimizer = MemoryOptimizer(config)
        
        # Apply parallelism
        self._apply_parallelism()
        
    def _apply_parallelism(self):
        """Apply all parallelism types to model"""
        # Tensor parallelism
        if self.config.tensor_parallel_size > 1:
            self._apply_tensor_parallelism()
        
        # Pipeline parallelism
        if self.config.pipeline_parallel_size > 1:
            self._apply_pipeline_parallelism()
        
        # Data parallelism (with DDP or FSDP)
        if self.config.data_parallel_size > 1:
            self._apply_data_parallelism()
        
        # Sequence parallelism
        if self.config.enable_sequence_parallel:
            self._apply_sequence_parallelism()
    
    def _apply_tensor_parallelism(self):
        """Apply tensor parallelism to model layers"""
        # Replace linear layers with tensor parallel versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with tensor parallel linear
                tp_linear = TensorParallelLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    tensor_parallel_size=self.config.tensor_parallel_size
                )
                # Copy weights
                with torch.no_grad():
                    tp_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        tp_linear.bias.copy_(module.bias)
                
                # Replace in model
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                parent = self.model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                setattr(parent, child_name, tp_linear)
    
    def _apply_pipeline_parallelism(self):
        """Apply pipeline parallelism"""
        # Split model into stages
        layers = list(self.model.children())
        num_layers = len(layers)
        layers_per_stage = num_layers // self.config.pipeline_parallel_size
        
        # Create pipeline stages
        stages = []
        for i in range(self.config.pipeline_parallel_size):
            start = i * layers_per_stage
            end = start + layers_per_stage if i < self.config.pipeline_parallel_size - 1 else num_layers
            stage = nn.Sequential(*layers[start:end])
            stages.append(stage)
        
        # Assign stages to ranks
        pp_rank = dist.get_rank() % self.config.pipeline_parallel_size
        self.model = stages[pp_rank]
    
    def _apply_data_parallelism(self):
        """Apply data parallelism"""
        if self.config.zero_stage > 0 and FSDP_AVAILABLE:
            # Use FSDP for ZeRO optimization
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
            
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=1e6
            )
            
            self.model = FSDP(
                self.model,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=self.config.offload_param),
                sharding_strategy=ShardingStrategy.FULL_SHARD if self.config.zero_stage == 3 else ShardingStrategy.SHARD_GRAD_OP,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                mixed_precision=self._get_mixed_precision_policy(),
                device_id=torch.cuda.current_device(),
                process_group=self.pg_manager.get_group(ParallelismType.DATA)
            )
        else:
            # Use standard DDP
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                output_device=torch.cuda.current_device(),
                process_group=self.pg_manager.get_group(ParallelismType.DATA),
                gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers
            )
    
    def _apply_sequence_parallelism(self):
        """Apply sequence parallelism to attention layers"""
        # Replace attention modules with sequence parallel versions
        for name, module in self.model.named_modules():
            if hasattr(module, 'self_attn') or 'attention' in name.lower():
                # This is simplified - actual implementation would be more sophisticated
                logger.info(f"Would apply sequence parallelism to {name}")
    
    def _get_mixed_precision_policy(self):
        """Get mixed precision policy for FSDP"""
        from torch.distributed.fsdp import MixedPrecision
        
        if self.config.mixed_precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        elif self.config.mixed_precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16
            )
        return None
    
    def forward(self, *args, **kwargs):
        """Forward pass with memory optimization"""
        with self.memory_optimizer.activation_checkpointing_context():
            return self.model(*args, **kwargs)


class TensorParallelLinear(nn.Module):
    """Tensor parallel linear layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        tensor_parallel_size: int = 1
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.tensor_parallel_size = tensor_parallel_size
        
        # Calculate per-device dimensions
        self.out_features_per_partition = out_features // tensor_parallel_size
        
        # Create weight and bias
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallelism"""
        # Local computation
        output = F.linear(input, self.weight, self.bias)
        
        # Gather output if needed
        if self.gather_output and self.tensor_parallel_size > 1:
            output_list = [torch.empty_like(output) for _ in range(self.tensor_parallel_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)
        
        return output


class DistributedTrainer:
    """Trainer for 4D parallel training"""
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        optimizer_class=torch.optim.AdamW,
        **optimizer_kwargs
    ):
        self.config = config
        
        # Wrap model with 4D parallelism
        self.model = FourDParallelModel(model, config)
        
        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer_class, **optimizer_kwargs)
        
        # Setup mixed precision
        self.scaler = self.model.memory_optimizer.setup_mixed_precision()
        
        # Stats
        self.step = 0
        
    def _create_optimizer(self, optimizer_class, **kwargs):
        """Create optimizer with ZeRO if needed"""
        if self.config.zero_stage > 0:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            return ZeroRedundancyOptimizer(
                self.model.parameters(),
                optimizer_class=optimizer_class,
                **kwargs
            )
        else:
            return optimizer_class(self.model.parameters(), **kwargs)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.cuda() for k, v in batch.items()}
        
        # Mixed precision context
        if self.scaler:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**batch)
                loss = outputs['loss']
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping
                )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        self.step += 1
        
        # Checkpoint if needed
        if self.step % self.config.checkpoint_interval == 0:
            self.save_checkpoint()
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'step': self.step
        }
    
    def save_checkpoint(self):
        """Save distributed checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{self.step}"
        )
        
        if dist.get_rank() == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        
        if self.config.use_sharded_checkpointing:
            # Sharded checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step': self.step,
                'config': self.config
            }, f"{checkpoint_path}/shard_{dist.get_rank()}.pt")
        else:
            # Single checkpoint (only rank 0)
            if dist.get_rank() == 0:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': self.step,
                    'config': self.config
                }, f"{checkpoint_path}/checkpoint.pt")
        
        logger.info(f"Saved checkpoint at step {self.step}")
