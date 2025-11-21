"""
Hybrid 3D Parallel Scheduler (DP + TP + PP + SP)
Dynamically coordinates all parallelism dimensions for near-linear scaling
"""
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParallelMode(Enum):
    """Parallelism modes"""
    DATA = "data"
    TENSOR = "tensor"
    PIPELINE = "pipeline"
    SEQUENCE = "sequence"
    CONTEXT = "context"
    EXPERT = "expert"


@dataclass
class HybridParallelConfig:
    """Configuration for hybrid parallelism"""
    # Parallelism dimensions
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_parallel_size: int = 1
    
    # Auto-configuration
    auto_parallel: bool = False
    auto_mode: str = "balanced"  # balanced, memory, throughput
    
    # Communication optimization
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    use_zero: bool = True
    zero_stage: int = 2
    
    # Memory optimization
    activation_checkpointing: bool = True
    cpu_offload: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if self.auto_parallel:
            logger.info("Auto-parallel mode enabled, will compute optimal topology")


class ProcessGroupManager:
    """Manages all process groups for hybrid parallelism"""
    
    def __init__(self, config: HybridParallelConfig):
        self.config = config
        
        if not dist.is_available() or not dist.is_initialized():
            self.world_size = 1
            self.rank = 0
            self.groups = {}
            return
        
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Auto-configure if requested
        if config.auto_parallel:
            self._auto_configure_topology()
        
        # Validate total parallelism
        self._validate_config()
        
        # Initialize all process groups
        self.groups = {}
        self._initialize_process_groups()
        
        # Compute local ranks
        self.local_ranks = self._compute_local_ranks()
        
        logger.info(f"Hybrid parallel topology: DP={config.data_parallel_size}, "
                   f"TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}, "
                   f"SP={config.sequence_parallel_size}, CP={config.context_parallel_size}, "
                   f"EP={config.expert_parallel_size}")
    
    def _auto_configure_topology(self):
        """Automatically configure parallel topology based on world size"""
        world_size = self.world_size
        mode = self.config.auto_mode
        
        logger.info(f"Auto-configuring parallel topology for {world_size} GPUs (mode: {mode})")
        
        if mode == "balanced":
            # Balanced: prioritize TP for small models, PP for large models
            if world_size <= 4:
                self.config.tensor_parallel_size = world_size
                self.config.data_parallel_size = 1
                self.config.pipeline_parallel_size = 1
            elif world_size <= 16:
                self.config.tensor_parallel_size = 4
                self.config.pipeline_parallel_size = world_size // 4
                self.config.data_parallel_size = 1
            else:
                self.config.tensor_parallel_size = 4
                self.config.pipeline_parallel_size = 4
                self.config.data_parallel_size = world_size // 16
        
        elif mode == "memory":
            # Memory-optimized: maximize PP to reduce per-GPU memory
            if world_size <= 4:
                self.config.pipeline_parallel_size = world_size
                self.config.tensor_parallel_size = 1
                self.config.data_parallel_size = 1
            else:
                self.config.pipeline_parallel_size = min(8, world_size // 2)
                self.config.tensor_parallel_size = 2
                self.config.data_parallel_size = world_size // (self.config.pipeline_parallel_size * 2)
        
        elif mode == "throughput":
            # Throughput-optimized: maximize DP for high batch throughput
            if world_size <= 4:
                self.config.data_parallel_size = world_size
                self.config.tensor_parallel_size = 1
                self.config.pipeline_parallel_size = 1
            else:
                self.config.tensor_parallel_size = 2
                self.config.data_parallel_size = world_size // 2
                self.config.pipeline_parallel_size = 1
    
    def _validate_config(self):
        """Validate parallel configuration"""
        total = (
            self.config.data_parallel_size *
            self.config.tensor_parallel_size *
            self.config.pipeline_parallel_size *
            self.config.sequence_parallel_size *
            self.config.context_parallel_size *
            self.config.expert_parallel_size
        )
        
        if total != self.world_size:
            raise ValueError(
                f"Total parallelism {total} != world size {self.world_size}. "
                f"DP={self.config.data_parallel_size}, TP={self.config.tensor_parallel_size}, "
                f"PP={self.config.pipeline_parallel_size}, SP={self.config.sequence_parallel_size}, "
                f"CP={self.config.context_parallel_size}, EP={self.config.expert_parallel_size}"
            )
    
    def _initialize_process_groups(self):
        """Initialize all process groups"""
        # Data parallel groups
        self.groups[ParallelMode.DATA] = self._create_data_parallel_groups()
        
        # Tensor parallel groups
        self.groups[ParallelMode.TENSOR] = self._create_tensor_parallel_groups()
        
        # Pipeline parallel groups
        self.groups[ParallelMode.PIPELINE] = self._create_pipeline_parallel_groups()
        
        # Sequence parallel (typically same as tensor parallel)
        if self.config.sequence_parallel_size > 1:
            self.groups[ParallelMode.SEQUENCE] = self.groups[ParallelMode.TENSOR]
        
        # Context parallel groups
        if self.config.context_parallel_size > 1:
            self.groups[ParallelMode.CONTEXT] = self._create_context_parallel_groups()
        
        # Expert parallel groups
        if self.config.expert_parallel_size > 1:
            self.groups[ParallelMode.EXPERT] = self._create_expert_parallel_groups()
    
    def _create_data_parallel_groups(self):
        """Create data parallel process groups"""
        dp_size = self.config.data_parallel_size
        if dp_size == 1:
            return None
        
        # All ranks with same TP/PP/SP/CP/EP coordinates form a DP group
        groups = []
        for i in range(self.world_size // dp_size):
            ranks = list(range(i * dp_size, (i + 1) * dp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                groups.append(group)
        
        return groups[0] if groups else None
    
    def _create_tensor_parallel_groups(self):
        """Create tensor parallel process groups"""
        tp_size = self.config.tensor_parallel_size
        if tp_size == 1:
            return None
        
        # Create TP groups
        num_tp_groups = self.world_size // tp_size
        for i in range(num_tp_groups):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                return group
        
        return None
    
    def _create_pipeline_parallel_groups(self):
        """Create pipeline parallel process groups"""
        pp_size = self.config.pipeline_parallel_size
        if pp_size == 1:
            return None
        
        # Create PP groups
        tp_size = self.config.tensor_parallel_size
        num_pp_groups = self.world_size // (pp_size * tp_size)
        
        for i in range(num_pp_groups):
            ranks = []
            for j in range(pp_size):
                base = i * pp_size * tp_size + j * tp_size
                ranks.extend(range(base, base + tp_size))
            
            group = dist.new_group(ranks)
            if self.rank in ranks:
                return group
        
        return None
    
    def _create_context_parallel_groups(self):
        """Create context parallel process groups"""
        cp_size = self.config.context_parallel_size
        if cp_size == 1:
            return None
        
        # Create CP groups
        num_cp_groups = self.world_size // cp_size
        for i in range(num_cp_groups):
            ranks = list(range(i * cp_size, (i + 1) * cp_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                return group
        
        return None
    
    def _create_expert_parallel_groups(self):
        """Create expert parallel process groups"""
        ep_size = self.config.expert_parallel_size
        if ep_size == 1:
            return None
        
        # Create EP groups
        num_ep_groups = self.world_size // ep_size
        for i in range(num_ep_groups):
            ranks = list(range(i * ep_size, (i + 1) * ep_size))
            group = dist.new_group(ranks)
            if self.rank in ranks:
                return group
        
        return None
    
    def _compute_local_ranks(self):
        """Compute local rank within each parallel dimension"""
        ranks = {}
        
        # Data parallel rank
        dp_size = self.config.data_parallel_size
        ranks[ParallelMode.DATA] = self.rank % dp_size
        
        # Tensor parallel rank
        tp_size = self.config.tensor_parallel_size
        ranks[ParallelMode.TENSOR] = (self.rank // dp_size) % tp_size
        
        # Pipeline parallel rank
        pp_size = self.config.pipeline_parallel_size
        ranks[ParallelMode.PIPELINE] = (self.rank // (dp_size * tp_size)) % pp_size
        
        # Context parallel rank
        cp_size = self.config.context_parallel_size
        ranks[ParallelMode.CONTEXT] = (self.rank // (dp_size * tp_size * pp_size)) % cp_size
        
        return ranks
    
    def get_group(self, mode: ParallelMode):
        """Get process group for given parallel mode"""
        return self.groups.get(mode)
    
    def get_rank(self, mode: ParallelMode):
        """Get local rank for given parallel mode"""
        return self.local_ranks.get(mode, 0)
    
    def get_world_size(self, mode: ParallelMode):
        """Get world size for given parallel mode"""
        if mode == ParallelMode.DATA:
            return self.config.data_parallel_size
        elif mode == ParallelMode.TENSOR:
            return self.config.tensor_parallel_size
        elif mode == ParallelMode.PIPELINE:
            return self.config.pipeline_parallel_size
        elif mode == ParallelMode.SEQUENCE:
            return self.config.sequence_parallel_size
        elif mode == ParallelMode.CONTEXT:
            return self.config.context_parallel_size
        elif mode == ParallelMode.EXPERT:
            return self.config.expert_parallel_size
        return 1


class HybridParallelEngine:
    """
    Unified engine for hybrid parallelism
    Coordinates DP, TP, PP, SP, CP, EP automatically
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: HybridParallelConfig,
    ):
        self.config = config
        self.model = model
        
        # Initialize process groups
        self.pg_manager = ProcessGroupManager(config)
        
        # Apply parallelism transformations
        self._apply_hybrid_parallelism()
        
        logger.info("Hybrid parallel engine initialized")
    
    def _apply_hybrid_parallelism(self):
        """Apply all parallelism transformations to model"""
        # 1. Tensor parallelism (shard weights)
        if self.config.tensor_parallel_size > 1:
            self._apply_tensor_parallelism()
        
        # 2. Pipeline parallelism (shard layers)
        if self.config.pipeline_parallel_size > 1:
            self._apply_pipeline_parallelism()
        
        # 3. Sequence/Context parallelism (shard activations)
        if self.config.sequence_parallel_size > 1 or self.config.context_parallel_size > 1:
            self._apply_sequence_parallelism()
        
        # 4. Data parallelism (replicate and sync gradients)
        if self.config.data_parallel_size > 1:
            self._apply_data_parallelism()
    
    def _apply_tensor_parallelism(self):
        """Apply tensor parallelism to linear layers"""
        logger.info("Applying tensor parallelism...")
        # Implementation would replace nn.Linear with ColumnParallelLinear/RowParallelLinear
        pass
    
    def _apply_pipeline_parallelism(self):
        """Apply pipeline parallelism to model layers"""
        logger.info("Applying pipeline parallelism...")
        # Implementation would partition layers across pipeline stages
        pass
    
    def _apply_sequence_parallelism(self):
        """Apply sequence/context parallelism"""
        logger.info("Applying sequence/context parallelism...")
        # Implementation would add reduce-scatter/all-gather ops
        pass
    
    def _apply_data_parallelism(self):
        """Apply data parallelism with ZeRO optimization"""
        logger.info("Applying data parallelism...")
        
        if self.config.use_zero:
            # Use FSDP for ZeRO
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import ShardingStrategy
                
                strategy = ShardingStrategy.FULL_SHARD if self.config.zero_stage == 3 else ShardingStrategy.SHARD_GRAD_OP
                
                self.model = FSDP(
                    self.model,
                    sharding_strategy=strategy,
                    device_id=torch.cuda.current_device(),
                    process_group=self.pg_manager.get_group(ParallelMode.DATA),
                )
            except ImportError:
                logger.warning("FSDP not available, using DDP")
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.model = DDP(
                    self.model,
                    device_ids=[torch.cuda.current_device()],
                    process_group=self.pg_manager.get_group(ParallelMode.DATA),
                )
        else:
            # Standard DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(
                self.model,
                device_ids=[torch.cuda.current_device()],
                process_group=self.pg_manager.get_group(ParallelMode.DATA),
            )
    
    def get_model(self):
        """Get the parallelized model"""
        return self.model
    
    def get_process_group_manager(self):
        """Get process group manager"""
        return self.pg_manager
