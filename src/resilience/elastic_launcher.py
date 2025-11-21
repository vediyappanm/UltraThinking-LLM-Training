"""
Elastic & Fault-Tolerant Training
Keep large training jobs alive even if nodes fail or are preempted
Enables 24/7 long-run stability with zero manual restarts
"""
import torch
import torch.distributed as dist
import os
import json
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import signal
import sys

logger = logging.getLogger(__name__)


@dataclass
class ElasticConfig:
    """Configuration for elastic training"""
    # Checkpointing
    checkpoint_dir: str = "./elastic_checkpoints"
    checkpoint_interval: int = 100  # Steps between checkpoints
    
    # Fault tolerance
    max_restarts: int = 10
    restart_delay: int = 30  # Seconds
    health_check_interval: int = 60
    
    # Elastic scaling
    min_nodes: int = 1
    max_nodes: int = 100
    enable_elastic_scaling: bool = True
    
    # State management
    save_optimizer_state: bool = True
    save_rng_state: bool = True
    save_dataloader_state: bool = True


class ElasticCheckpoint:
    """
    Elastic checkpoint manager
    Handles saving/loading of all training state for recovery
    """
    
    def __init__(self, config: ElasticConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoint history
        self.checkpoint_history = []
        
        logger.info(f"Elastic checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        dataloader_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Save complete training state for elastic recovery
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler
            step: Current training step
            epoch: Current epoch
            dataloader_state: DataLoader state (for resumption)
            metadata: Additional metadata
        """
        checkpoint_path = self.checkpoint_dir / f"elastic_step_{step}.pt"
        
        # Gather all state
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self._get_model_state(model),
        }
        
        # Optimizer state
        if self.config.save_optimizer_state and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # RNG state
        if self.config.save_rng_state:
            checkpoint['rng_state'] = {
                'torch': torch.get_rng_state(),
                'numpy': None,  # Add numpy state if needed
            }
            if torch.cuda.is_available():
                checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
        
        # DataLoader state
        if self.config.save_dataloader_state and dataloader_state is not None:
            checkpoint['dataloader_state'] = dataloader_state
        
        # Metadata
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Distributed info
        if dist.is_initialized():
            checkpoint['world_size'] = dist.get_world_size()
            checkpoint['rank'] = dist.get_rank()
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        
        # Update history
        self.checkpoint_history.append({
            'step': step,
            'path': str(checkpoint_path),
            'timestamp': time.time(),
        })
        
        # Cleanup old checkpoints (keep last 3)
        self._cleanup_old_checkpoints(keep=3)
        
        logger.info(f"Saved elastic checkpoint at step {step}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint for recovery
        
        Args:
            model: Model to load into
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
            checkpoint_path: Specific checkpoint to load (or latest)
        
        Returns:
            Checkpoint metadata
        """
        # Find checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found for recovery")
        
        logger.info(f"Loading elastic checkpoint: {checkpoint_path}")
        
        # Load
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore RNG state
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            if torch.cuda.is_available() and 'cuda' in checkpoint['rng_state']:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
        
        logger.info(f"Recovered from step {checkpoint['step']}, epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def _get_model_state(self, model: torch.nn.Module) -> Dict:
        """Get model state dict (handle DDP/FSDP)"""
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("elastic_step_*.pt"))
        if not checkpoints:
            return None
        return str(checkpoints[-1])
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints, keep only recent ones"""
        checkpoints = sorted(self.checkpoint_dir.glob("elastic_step_*.pt"))
        if len(checkpoints) > keep:
            for ckpt in checkpoints[:-keep]:
                ckpt.unlink()


class ElasticTrainer:
    """
    Elastic training coordinator
    Handles fault tolerance, recovery, and elastic scaling
    """
    
    def __init__(
        self,
        config: ElasticConfig,
        train_fn: Callable,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.train_fn = train_fn
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Checkpoint manager
        self.checkpoint_mgr = ElasticCheckpoint(config)
        
        # State tracking
        self.current_step = 0
        self.current_epoch = 0
        self.restart_count = 0
        self.is_recovering = False
        
        # Register signal handlers
        self._register_signal_handlers()
        
        logger.info("Elastic trainer initialized")
    
    def train(self, num_steps: int, start_step: int = 0):
        """
        Run elastic training with fault tolerance
        
        Args:
            num_steps: Total training steps
            start_step: Starting step (for recovery)
        """
        self.current_step = start_step
        
        try:
            while self.current_step < num_steps:
                # Health check
                if self.current_step % self.config.health_check_interval == 0:
                    self._health_check()
                
                # Training step
                try:
                    self.train_fn(self.current_step)
                    self.current_step += 1
                    
                    # Checkpoint
                    if self.current_step % self.config.checkpoint_interval == 0:
                        self.checkpoint_mgr.save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.current_step,
                            self.current_epoch,
                        )
                
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    self._handle_failure()
        
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self.checkpoint_mgr.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.current_step,
                self.current_epoch,
            )
            raise
    
    def recover(self) -> bool:
        """
        Recover from failure
        
        Returns:
            success: Whether recovery succeeded
        """
        if self.restart_count >= self.config.max_restarts:
            logger.error(f"Max restarts ({self.config.max_restarts}) exceeded")
            return False
        
        logger.info(f"Attempting recovery (restart {self.restart_count + 1})")
        
        try:
            # Wait before restart
            time.sleep(self.config.restart_delay)
            
            # Load checkpoint
            checkpoint = self.checkpoint_mgr.load_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
            )
            
            # Update state
            self.current_step = checkpoint['step']
            self.current_epoch = checkpoint['epoch']
            self.restart_count += 1
            self.is_recovering = True
            
            logger.info("Recovery successful")
            return True
        
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def _handle_failure(self):
        """Handle training failure"""
        logger.warning("Handling training failure...")
        
        # Try to recover
        if self.recover():
            logger.info("Resuming training after recovery")
        else:
            logger.error("Recovery failed, exiting")
            sys.exit(1)
    
    def _health_check(self):
        """Check training health"""
        # Check GPU health
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.error(f"GPU health check failed: {e}")
                raise
        
        # Check distributed health
        if dist.is_initialized():
            try:
                # Simple all-reduce to verify communication
                tensor = torch.tensor([1.0], device='cuda' if torch.cuda.is_available() else 'cpu')
                dist.all_reduce(tensor)
            except Exception as e:
                logger.error(f"Distributed health check failed: {e}")
                raise
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, saving checkpoint...")
            self.checkpoint_mgr.save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.current_step,
                self.current_epoch,
            )
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


class TorchElasticLauncher:
    """
    Integration with PyTorch Elastic (torchrun)
    Handles dynamic node scaling
    """
    
    def __init__(self, config: ElasticConfig):
        self.config = config
        
        # Check if running under elastic
        self.is_elastic = os.environ.get('TORCHELASTIC_RUN_ID') is not None
        
        if self.is_elastic:
            logger.info("Running under PyTorch Elastic")
            self.run_id = os.environ.get('TORCHELASTIC_RUN_ID')
            self.restart_count = int(os.environ.get('TORCHELASTIC_RESTART_COUNT', 0))
        else:
            logger.info("Not running under PyTorch Elastic")
    
    def should_recover(self) -> bool:
        """Check if we should recover from checkpoint"""
        return self.is_elastic and self.restart_count > 0
    
    def get_world_info(self) -> Dict[str, int]:
        """Get current world size and rank"""
        if dist.is_initialized():
            return {
                'world_size': dist.get_world_size(),
                'rank': dist.get_rank(),
                'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
            }
        return {'world_size': 1, 'rank': 0, 'local_rank': 0}


def create_elastic_trainer(
    config: ElasticConfig,
    train_fn: Callable,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> ElasticTrainer:
    """
    Factory function to create elastic trainer
    
    Args:
        config: Elastic configuration
        train_fn: Training function
        model: Model to train
        optimizer: Optimizer
        scheduler: LR scheduler
    
    Returns:
        ElasticTrainer instance
    """
    return ElasticTrainer(config, train_fn, model, optimizer, scheduler)
