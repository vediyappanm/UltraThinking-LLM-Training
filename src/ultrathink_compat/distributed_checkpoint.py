"""
Distributed/Sharded Checkpointing for Megatron-LM
Compatible with DP/TP/PP/EP parallel groups
"""
import torch
import torch.distributed as dist
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ParallelState:
    """Track parallel group information"""
    
    def __init__(self):
        self.data_parallel_group = None
        self.tensor_parallel_group = None
        self.pipeline_parallel_group = None
        self.expert_parallel_group = None
        
        self.data_parallel_rank = 0
        self.tensor_parallel_rank = 0
        self.pipeline_parallel_rank = 0
        self.expert_parallel_rank = 0
        
        self.data_parallel_size = 1
        self.tensor_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.expert_parallel_size = 1
    
    def initialize(
        self,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ):
        """Initialize parallel state"""
        if not dist.is_available() or not dist.is_initialized():
            return
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Validate sizes
        total_parallel = (
            data_parallel_size *
            tensor_parallel_size *
            pipeline_parallel_size *
            expert_parallel_size
        )
        
        if total_parallel != world_size:
            raise ValueError(
                f"Total parallelism {total_parallel} != world size {world_size}"
            )
        
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.expert_parallel_size = expert_parallel_size
        
        # Compute ranks (simplified)
        self.data_parallel_rank = rank // (tensor_parallel_size * pipeline_parallel_size * expert_parallel_size)
        self.tensor_parallel_rank = (rank // (pipeline_parallel_size * expert_parallel_size)) % tensor_parallel_size
        self.pipeline_parallel_rank = (rank // expert_parallel_size) % pipeline_parallel_size
        self.expert_parallel_rank = rank % expert_parallel_size


# Global parallel state
_PARALLEL_STATE = ParallelState()


def get_parallel_state() -> ParallelState:
    """Get global parallel state"""
    return _PARALLEL_STATE


class DistributedCheckpoint:
    """Distributed checkpoint manager"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save distributed checkpoint"""
        parallel_state = get_parallel_state()
        
        # Create checkpoint directory
        ckpt_dir = self.checkpoint_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state (sharded by TP/PP/EP)
        model_state = self._get_model_state(model)
        
        # Construct shard filename
        shard_name = self._get_shard_name(parallel_state)
        model_path = ckpt_dir / f"model_{shard_name}.pt"
        
        checkpoint = {
            'model': model_state,
            'step': step,
            'parallel_state': {
                'data_parallel_rank': parallel_state.data_parallel_rank,
                'tensor_parallel_rank': parallel_state.tensor_parallel_rank,
                'pipeline_parallel_rank': parallel_state.pipeline_parallel_rank,
                'expert_parallel_rank': parallel_state.expert_parallel_rank,
                'data_parallel_size': parallel_state.data_parallel_size,
                'tensor_parallel_size': parallel_state.tensor_parallel_size,
                'pipeline_parallel_size': parallel_state.pipeline_parallel_size,
                'expert_parallel_size': parallel_state.expert_parallel_size,
            }
        }
        
        # Save optimizer state (only on DP rank 0)
        if optimizer is not None and parallel_state.data_parallel_rank == 0:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        # Save scheduler state (only on rank 0)
        if scheduler is not None and dist.get_rank() == 0:
            checkpoint['scheduler'] = scheduler.state_dict()
        
        # Save metadata (only on rank 0)
        if metadata is not None and dist.get_rank() == 0:
            checkpoint['metadata'] = metadata
        
        # Save checkpoint shard
        torch.save(checkpoint, model_path)
        
        # Save metadata file (rank 0 only)
        if dist.get_rank() == 0:
            self._save_metadata(ckpt_dir, step, parallel_state, metadata)
        
        logger.info(f"Saved checkpoint shard: {model_path}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load distributed checkpoint"""
        parallel_state = get_parallel_state()
        
        # Find checkpoint directory
        if step is None:
            # Load latest checkpoint
            ckpt_dirs = sorted(self.checkpoint_dir.glob("step_*"))
            if not ckpt_dirs:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
            ckpt_dir = ckpt_dirs[-1]
        else:
            ckpt_dir = self.checkpoint_dir / f"step_{step}"
        
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
        
        # Load shard
        shard_name = self._get_shard_name(parallel_state)
        model_path = ckpt_dir / f"model_{shard_name}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint shard not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model'], strict=False)
        
        # Load optimizer state
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        logger.info(f"Loaded checkpoint from: {model_path}")
        
        return {
            'step': checkpoint.get('step', 0),
            'metadata': checkpoint.get('metadata', {}),
        }
    
    def _get_model_state(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Get model state dict"""
        # Handle DDP/FSDP wrapped models
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()
    
    def _get_shard_name(self, parallel_state: ParallelState) -> str:
        """Generate shard name based on parallel ranks"""
        return (
            f"dp{parallel_state.data_parallel_rank}_"
            f"tp{parallel_state.tensor_parallel_rank}_"
            f"pp{parallel_state.pipeline_parallel_rank}_"
            f"ep{parallel_state.expert_parallel_rank}"
        )
    
    def _save_metadata(
        self,
        ckpt_dir: Path,
        step: int,
        parallel_state: ParallelState,
        metadata: Optional[Dict[str, Any]],
    ):
        """Save checkpoint metadata"""
        meta = {
            'step': step,
            'parallel_config': {
                'data_parallel_size': parallel_state.data_parallel_size,
                'tensor_parallel_size': parallel_state.tensor_parallel_size,
                'pipeline_parallel_size': parallel_state.pipeline_parallel_size,
                'expert_parallel_size': parallel_state.expert_parallel_size,
            },
        }
        
        if metadata is not None:
            meta.update(metadata)
        
        meta_path = ckpt_dir / 'metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)


def save_distributed_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save distributed checkpoint (convenience function)"""
    ckpt_manager = DistributedCheckpoint(checkpoint_dir)
    ckpt_manager.save_checkpoint(model, optimizer, scheduler, step, metadata)


def load_distributed_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Load distributed checkpoint (convenience function)"""
    ckpt_manager = DistributedCheckpoint(checkpoint_dir)
    return ckpt_manager.load_checkpoint(model, optimizer, scheduler, step)
