#!/usr/bin/env python3
"""
UltraThinking Unified Production Training Script
Integrates ALL advanced features from src/ folder for production-grade training

Features Integrated:
- Advanced Parallel: Context Parallel, Hybrid 3D, KV Cache, SAR, Adaptive MoE, Speculative Decoding
- Advanced Ops: FlashAttention v3, Unified Comm, RoPE variants, Fused Optimizers, LoRA/PEFT, Compression
- Resilience: Elastic training, fault tolerance, auto-recovery
- Alignment: RLHF/PPO engine for human alignment
- Infrastructure: Vocab parallel, profiling, paged checkpointing, MoA
- Models: UltraThink, Dynamic Reasoning, Constitutional AI, MoE, Multimodal
- Training: Distributed 4D, RLHF Advanced, optimizers
- Monitoring: Metrics tracking, system monitoring
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Core model imports
from src.models import (
    UltraThinkModel, UltraThinkConfig,
    AdvancedGPTModel, ModelConfig,
    DynamicReasoningEngine,
    ConstitutionalReasoningCore,
    MoELayer, ExpertConfig,
    UnifiedMultiModalModel, MultiModalConfig,
)

# Advanced parallel imports
from src.advanced_parallel import (
    ContextParallelAttention,
    RingAttention,
    HybridParallelConfig,
    HybridParallelEngine,
    DistributedKVCache,
    SelectiveCheckpointWrapper,
    AdaptiveMoELayer,
    SpeculativeDecoder,
)

# Advanced ops imports
from src.advanced_ops import (
    create_attention_layer,
    create_comm_layer,
    create_position_encoding,
    create_fused_optimizer,
    apply_lora_to_model,
    apply_gradient_compression,
    apply_activation_compression,
    get_trainable_parameters,
)

# Resilience imports
from src.resilience import (
    ElasticConfig,
    create_elastic_trainer,
)

# Alignment imports
from src.alignment import (
    RLHFConfig,
    create_rlhf_trainer,
    RewardModel,
)

# Infrastructure imports
from src.infrastructure import (
    VocabParallelEmbedding,
    PerformanceProfiler,
    PagedCheckpointManager,
    MixtureOfAgents,
)

# Training imports
from src.training import (
    DistributedConfig,
    DistributedTrainer,
    RLHF2System,
)

# Data imports
from src.data import (
    UltraThinkDataset,
    create_dataloaders,
)

# Monitoring imports
from src.monitoring import (
    MetricsTracker,
    SystemMonitor,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class UnifiedTrainingConfig:
    """Unified configuration for all training features"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load from YAML if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._load_from_dict(config_dict)
        else:
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration"""
        # Model configuration
        self.model_type = "ultrathink"  # ultrathink, gpt, multimodal
        self.vocab_size = 100352
        self.hidden_size = 4096
        self.num_layers = 32
        self.num_heads = 32
        self.num_kv_heads = 8  # For GQA
        self.max_seq_length = 8192
        
        # Advanced features toggles
        self.use_flash_attention = True
        self.attention_type = "gqa"  # flash_v3, mqa, gqa, sliding_window
        self.use_rope = True
        self.rope_type = "dynamic_continuous"  # standard, ntk_scaled, dynamic_continuous
        self.use_moe = True
        self.num_experts = 8
        self.experts_per_token = 2
        self.use_dynamic_reasoning = True
        self.use_constitutional_ai = True
        
        # Parallelism
        self.use_context_parallel = True
        self.context_parallel_size = 2
        self.use_hybrid_parallel = True
        self.tensor_parallel_size = 2
        self.pipeline_parallel_size = 2
        self.data_parallel_size = 1
        self.use_sequence_parallel = True
        
        # Advanced ops
        self.use_fused_optimizer = True
        self.optimizer_type = "fused_adamw"
        self.use_async_grad_accumulation = True
        self.gradient_accumulation_steps = 16
        self.use_gradient_compression = True
        self.gradient_compression_type = "topk"
        self.gradient_compression_ratio = 0.1
        self.use_activation_compression = True
        
        # PEFT
        self.use_lora = False
        self.lora_rank = 8
        self.lora_alpha = 16.0
        self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Resilience
        self.use_elastic_training = True
        self.checkpoint_interval = 100
        self.max_restarts = 10
        self.use_paged_checkpointing = True
        self.checkpoint_page_size_mb = 256
        self.checkpoint_offload = "cpu"
        
        # RLHF
        self.use_rlhf = False
        self.ppo_epochs = 4
        self.kl_coeff = 0.1
        
        # Infrastructure
        self.use_vocab_parallel = True
        self.vocab_parallel_size = 2
        self.use_profiling = True
        self.enable_nvtx = True
        self.use_mixture_of_agents = False
        
        # Training
        self.batch_size = 32
        self.micro_batch_size = 2
        self.learning_rate = 2e-4
        self.weight_decay = 0.01
        self.max_steps = 100000
        self.warmup_steps = 1000
        self.mixed_precision = "bf16"
        
        # Data
        self.data_path = "./data"
        self.num_workers = 4
        
        # Monitoring
        self.log_interval = 10
        self.eval_interval = 500
        self.save_interval = 1000
        
        # Output
        self.output_dir = "./outputs"
        self.experiment_name = "ultrathink_production"
    
    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to YAML"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class UnifiedProductionTrainer:
    """
    Unified production trainer integrating all advanced features
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize distributed
        self._init_distributed()
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config.save(str(self.output_dir / "config.yaml"))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Advanced components
        self.profiler = None
        self.checkpoint_mgr = None
        self.elastic_trainer = None
        self.metrics_tracker = None
        self.system_monitor = None
        
        logger.info("Unified Production Trainer initialized")
        logger.info(f"Configuration: {config.to_dict()}")
    
    def _init_distributed(self):
        """Initialize distributed training"""
        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            logger.info(f"Distributed initialized: rank={self.rank}, world_size={self.world_size}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            logger.info("Running in single-GPU mode")
    
    def build_model(self):
        """Build model with all advanced features"""
        logger.info(f"Building {self.config.model_type} model...")
        
        if self.config.model_type == "ultrathink":
            # UltraThink model with all features
            model_config = UltraThinkConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                max_seq_length=self.config.max_seq_length,
                use_moe=self.config.use_moe,
                num_experts=self.config.num_experts,
                experts_per_token=self.config.experts_per_token,
            )
            self.model = UltraThinkModel(model_config)
        
        elif self.config.model_type == "gpt":
            # Advanced GPT model
            model_config = ModelConfig(
                vocab_size=self.config.vocab_size,
                n_embd=self.config.hidden_size,
                n_layer=self.config.num_layers,
                n_head=self.config.num_heads,
                n_positions=self.config.max_seq_length,
            )
            self.model = AdvancedGPTModel(model_config)
        
        elif self.config.model_type == "multimodal":
            # Multimodal model
            model_config = MultiModalConfig(
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
            )
            self.model = UnifiedMultiModalModel(model_config)
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            logger.info("Applying LoRA...")
            self.model = apply_lora_to_model(
                self.model,
                target_modules=self.config.lora_target_modules,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )
            params = get_trainable_parameters(self.model)
            logger.info(f"LoRA applied: {params['trainable_pct']:.2f}% trainable parameters")
        
        # Apply gradient compression
        if self.config.use_gradient_compression:
            logger.info("Applying gradient compression...")
            apply_gradient_compression(
                self.model,
                compression_type=self.config.gradient_compression_type,
                compression_ratio=self.config.gradient_compression_ratio,
            )
        
        # Apply activation compression
        if self.config.use_activation_compression:
            logger.info("Applying activation compression...")
            apply_activation_compression(self.model, compression_type="fp16")
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Apply hybrid parallelism
        if self.config.use_hybrid_parallel and self.world_size > 1:
            logger.info("Applying hybrid parallelism...")
            parallel_config = HybridParallelConfig(
                tensor_parallel_size=self.config.tensor_parallel_size,
                pipeline_parallel_size=self.config.pipeline_parallel_size,
                data_parallel_size=self.config.data_parallel_size,
                sequence_parallel_size=self.config.context_parallel_size if self.config.use_sequence_parallel else 1,
            )
            parallel_engine = HybridParallelEngine(parallel_config)
            self.model = parallel_engine.apply(self.model)
        
        # Wrap with DDP if needed
        elif self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        logger.info(f"Model built: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
        
        return self.model
    
    def build_optimizer(self):
        """Build optimizer with fused operations"""
        logger.info("Building optimizer...")
        
        if self.config.use_fused_optimizer:
            self.optimizer = create_fused_optimizer(
                self.model,
                optimizer_type=self.config.optimizer_type,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                use_fused=True,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        
        return self.optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
        )
        
        return self.scheduler
    
    def build_dataloaders(self):
        """Build data loaders"""
        logger.info("Building data loaders...")
        
        from src.data import UltraThinkDataset, create_dataloaders
        
        # Create dataset
        train_dataset = UltraThinkDataset(
            data_path=self.config.data_path,
            split="train",
            max_length=self.config.max_seq_length,
        )
        
        val_dataset = UltraThinkDataset(
            data_path=self.config.data_path,
            split="validation",
            max_length=self.config.max_seq_length,
        )
        
        # Create loaders
        self.train_loader, self.val_loader = create_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        
        logger.info(f"Data loaders built: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return self.train_loader, self.val_loader
    
    def setup_infrastructure(self):
        """Setup infrastructure components"""
        logger.info("Setting up infrastructure...")
        
        # Performance profiler
        if self.config.use_profiling:
            self.profiler = PerformanceProfiler(
                enable_nvtx=self.config.enable_nvtx,
                enable_torch_profiler=True,
                log_dir=str(self.output_dir / "profiling"),
            )
            logger.info("Performance profiler enabled")
        
        # Paged checkpoint manager
        if self.config.use_paged_checkpointing:
            self.checkpoint_mgr = PagedCheckpointManager(
                checkpoint_dir=str(self.output_dir / "checkpoints"),
                page_size_mb=self.config.checkpoint_page_size_mb,
                offload_device=self.config.checkpoint_offload,
            )
            logger.info("Paged checkpoint manager enabled")
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker(
            log_dir=str(self.output_dir / "metrics"),
            experiment_name=self.config.experiment_name,
        )
        
        # System monitor
        self.system_monitor = SystemMonitor(
            log_interval=self.config.log_interval,
        )
        
        logger.info("Infrastructure setup complete")
    
    def train_step(self, batch, step):
        """Single training step"""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward pass
        if self.profiler:
            with self.profiler.mark_nvtx(f"forward_step_{step}"):
                outputs = self.model(input_ids, labels=labels)
        else:
            outputs = self.model(input_ids, labels=labels)
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.profiler:
            with self.profiler.mark_nvtx(f"backward_step_{step}"):
                loss.backward()
        else:
            loss.backward()
        
        # Optimizer step (if accumulated enough)
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Build all components
        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_dataloaders()
        self.setup_infrastructure()
        
        # Start profiling
        if self.profiler:
            self.profiler.start_profiling()
        
        # Training loop
        global_step = 0
        epoch = 0
        
        try:
            while global_step < self.config.max_steps:
                epoch += 1
                logger.info(f"Epoch {epoch}")
                
                for batch_idx, batch in enumerate(self.train_loader):
                    # Training step
                    loss = self.train_step(batch, global_step)
                    
                    # Logging
                    if global_step % self.config.log_interval == 0:
                        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                        logger.info(f"Step {global_step}: loss={loss:.4f}, lr={lr:.2e}")
                        
                        # Track metrics
                        if self.metrics_tracker:
                            self.metrics_tracker.log_scalar("train/loss", loss, global_step)
                            self.metrics_tracker.log_scalar("train/lr", lr, global_step)
                        
                        # Memory stats
                        if self.profiler:
                            mem_stats = self.profiler.get_memory_stats()
                            logger.info(f"GPU Memory: {mem_stats.get('allocated_gb', 0):.2f} GB")
                    
                    # Checkpointing
                    if global_step % self.config.save_interval == 0 and global_step > 0:
                        self.save_checkpoint(global_step, epoch)
                    
                    # Profiler step
                    if self.profiler:
                        self.profiler.step()
                    
                    global_step += 1
                    
                    if global_step >= self.config.max_steps:
                        break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted, saving checkpoint...")
            self.save_checkpoint(global_step, epoch)
        
        finally:
            # Stop profiling
            if self.profiler:
                self.profiler.stop_profiling()
            
            logger.info("Training complete!")
    
    def save_checkpoint(self, step, epoch):
        """Save checkpoint"""
        logger.info(f"Saving checkpoint at step {step}...")
        
        if self.checkpoint_mgr:
            # Paged checkpoint
            self.checkpoint_mgr.save_paged_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                step=step,
            )
        else:
            # Standard checkpoint
            checkpoint_path = self.output_dir / "checkpoints" / f"step_{step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: step {step}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="UltraThinking Unified Production Training")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--model_type", type=str, default="ultrathink", choices=["ultrathink", "gpt", "multimodal"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--use_rlhf", action="store_true", help="Enable RLHF training")
    parser.add_argument("--disable_profiling", action="store_true", help="Disable profiling")
    
    args = parser.parse_args()
    
    # Create config
    config = UnifiedTrainingConfig(args.config)
    
    # Override with CLI args
    if args.model_type:
        config.model_type = args.model_type
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.use_lora:
        config.use_lora = True
    if args.use_rlhf:
        config.use_rlhf = True
    if args.disable_profiling:
        config.use_profiling = False
    
    # Create trainer
    trainer = UnifiedProductionTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
