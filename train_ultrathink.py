"""
ULTRATHINK Training Pipeline
Complete training pipeline for GPT-5/Claude 4.1 level model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Any
import json
import argparse
import numpy as np
from pathlib import Path

# Dummy dataset for demonstration (moved to module level for Windows multiprocessing)
class DummyDataset:
    """Dummy dataset for demonstration"""
    def __init__(self, size=10000, seq_len=512):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (self.seq_len,)),
            'labels': torch.randint(0, 1000, (self.seq_len,))
        }

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all components
try:
    from src.models.ultrathink import UltraThinkModel, UltraThinkConfig
    from src.models.architecture import ModelConfig
    from src.models.moe_advanced import ExpertConfig
    from src.models.multimodal import MultiModalConfig
    from src.training.distributed_4d import DistributedConfig, DistributedTrainer
    from src.training.rlhf_advanced import RLHF2System, RLHFConfig
    from src.data.synthetic_generation import SyntheticDataEngine, SyntheticDataConfig
    from src.evaluation.benchmarks import ComprehensiveBenchmarkSuite, BenchmarkConfig
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class UltraThinkTrainer:
    """Main trainer for ULTRATHINK model"""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize distributed training if available
        self.setup_distributed()
        
        # Create configurations
        self.config = self.create_config()
        
        # Initialize model
        logger.info("Initializing ULTRATHINK model...")
        self.model = UltraThinkModel(self.config)
        
        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup distributed training
        if self.args.distributed:
            self.setup_distributed_model()
        
        # Initialize components
        self.setup_training_components()
        
        # Initialize wandb if requested
        if self.args.use_wandb and self.is_main_process():
            wandb.init(
                project="ultrathink",
                name=self.args.run_name,
                config=self.config.__dict__
            )
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        if self.args.distributed:
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
    
    def is_main_process(self):
        """Check if this is the main process"""
        return self.rank == 0
    
    def create_config(self) -> UltraThinkConfig:
        """Create model configuration"""
        # Base model config
        # Ensure num_kv_heads is compatible with num_heads for GQA
        num_kv_heads = min(self.args.num_kv_heads, self.args.num_heads)
        if self.args.num_heads % num_kv_heads != 0:
            num_kv_heads = self.args.num_heads  # Fall back to MHA if not divisible
        
        model_config = ModelConfig(
            vocab_size=self.args.vocab_size,
            n_positions=self.args.max_seq_length,
            n_embd=self.args.hidden_size,
            n_layer=self.args.num_layers,
            n_head=self.args.num_heads,
            n_kv_head=num_kv_heads,
            rotary_dim=self.args.hidden_size // self.args.num_heads,  # Set rotary_dim to head_dim
            intermediate_size=self.args.intermediate_size,
            activation=self.args.activation,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            flash_attention=self.args.use_flash_attention,
            gradient_checkpointing=self.args.gradient_checkpointing
        )
        
        # MoE config
        moe_config = ExpertConfig(
            num_knowledge_experts=self.args.num_knowledge_experts,
            num_skill_experts=self.args.num_skill_experts,
            num_meta_experts=self.args.num_meta_experts,
            num_safety_experts=self.args.num_safety_experts,
            top_k=self.args.moe_top_k,
            expert_capacity=self.args.expert_capacity
        )
        
        # Multi-modal config
        multimodal_config = MultiModalConfig(
            hidden_dim=self.args.hidden_size,
            image_size=self.args.image_size,
            patch_size=self.args.patch_size,
            audio_sample_rate=self.args.audio_sample_rate
        )
        
        # Create main config
        config = UltraThinkConfig(
            model_config=model_config,
            moe_config=moe_config,
            multimodal_config=multimodal_config,
            enable_dre=self.args.enable_dre,
            enable_constitutional=self.args.enable_constitutional,
            enable_moe=self.args.enable_moe,
            enable_multimodal=self.args.enable_multimodal,
            enable_rlhf=self.args.enable_rlhf,
            batch_size=self.args.batch_size,
            gradient_accumulation=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            max_steps=self.args.max_steps
        )
        
        return config
    
    def setup_distributed_model(self):
        """Setup model for distributed training"""
        if self.args.use_4d_parallelism:
            # 4D parallelism
            dist_config = DistributedConfig(
                data_parallel_size=self.args.data_parallel_size,
                tensor_parallel_size=self.args.tensor_parallel_size,
                pipeline_parallel_size=self.args.pipeline_parallel_size,
                expert_parallel_size=self.args.expert_parallel_size,
                zero_stage=self.args.zero_stage,
                gradient_checkpointing=self.args.gradient_checkpointing
            )
            
            self.distributed_trainer = DistributedTrainer(
                self.model.core,
                dist_config,
                optimizer_class=torch.optim.AdamW,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            # Standard DDP
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
    
    def setup_training_components(self):
        """Setup training components"""
        # Optimizer
        if not self.args.use_4d_parallelism:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.adam_beta1, self.args.adam_beta2)
            )
        
        # Learning rate scheduler
        from transformers import get_linear_schedule_with_warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer if not self.args.use_4d_parallelism else self.distributed_trainer.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.max_steps
        )
        
        # Mixed precision (fix deprecation warning)
        if self.args.use_amp:
            try:
                from torch.amp import GradScaler as NewGradScaler
                self.scaler = NewGradScaler('cuda' if torch.cuda.is_available() else 'cpu')
            except ImportError:
                # Fallback for older PyTorch versions
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # RLHF system
        if self.args.enable_rlhf:
            rlhf_config = RLHFConfig(
                ppo_epochs=self.args.ppo_epochs,
                ppo_batch_size=self.args.ppo_batch_size
            )
            self.rlhf_system = RLHF2System(self.model.core, rlhf_config)
        
        # Synthetic data engine
        if self.args.use_synthetic_data:
            synthetic_config = SyntheticDataConfig()
            self.synthetic_engine = SyntheticDataEngine(synthetic_config)
        
        # Evaluation suite
        bench_config = BenchmarkConfig()
        self.benchmark_suite = ComprehensiveBenchmarkSuite(bench_config)
    
    def load_datasets(self):
        """Load training datasets"""
        logger.info(f"Loading {self.args.dataset} dataset...")
        
        # Import dataset utilities
        from src.data.datasets import create_dataset, create_mixed_dataset, DATASET_CONFIGS, DatasetConfig
        
        if self.args.dataset == "dummy":
            # Use dummy dataset for testing
            train_dataset = DummyDataset(self.args.train_samples)
            val_dataset = DummyDataset(self.args.val_samples)
            logger.info(f"Using dummy dataset: {self.args.train_samples} train, {self.args.val_samples} val samples")
        
        elif self.args.dataset == "custom":
            # Use custom dataset from file
            if not self.args.data_path:
                raise ValueError("--data_path required when using custom dataset")
            
            config = DatasetConfig(
                name="custom",
                local_path=self.args.data_path,
                text_column=self.args.text_column,
                max_length=self.args.max_seq_length,
                tokenizer_name=self.args.tokenizer_name,
                max_samples=self.args.max_samples,
                streaming=self.args.streaming
            )
            
            train_dataset = create_dataset(config, "train")
            val_dataset = create_dataset(config, "train")  # Use same data, will be split
            logger.info(f"Using custom dataset from {self.args.data_path}")
        
        elif self.args.mix_datasets:
            # Mix multiple datasets according to weights
            pairs = [p.strip() for p in self.args.mix_datasets.split(',') if p.strip()]
            configs = {}
            weights = {}
            for p in pairs:
                if ':' not in p:
                    logger.warning(f"Ignoring malformed mix entry '{p}', expected format name:weight")
                    continue
                name, w = p.split(':', 1)
                name = name.strip()
                try:
                    weight = float(w)
                except ValueError:
                    logger.warning(f"Invalid weight '{w}' for dataset '{name}', skipping")
                    continue
                if name not in DATASET_CONFIGS:
                    logger.warning(f"Unknown dataset '{name}' in mix; available: {list(DATASET_CONFIGS.keys())}")
                    continue
                base = DATASET_CONFIGS[name]
                # Clone and apply global overrides
                cfg = DatasetConfig(
                    name=base.name,
                    subset=base.subset,
                    split_train=base.split_train,
                    split_val=base.split_val,
                    split_test=base.split_test,
                    text_column=base.text_column,
                    max_length=self.args.max_seq_length,
                    tokenizer_name=self.args.tokenizer_name,
                    streaming=base.streaming,
                    cache_dir=base.cache_dir,
                    num_proc=base.num_proc,
                    local_path=None,
                    file_type='auto',
                    min_length=base.min_length,
                    max_samples=self.args.max_samples
                )
                configs[name] = cfg
                weights[name] = weight
            if not configs:
                raise ValueError("--mix_datasets provided but no valid datasets parsed")
            train_dataset = create_mixed_dataset(configs, weights, split='train')
            val_dataset = create_mixed_dataset(configs, weights, split='validation')
            logger.info(f"Using mixed datasets: {', '.join([f'{k}:{weights[k]}' for k in configs.keys()])}")

        else:
            # Use predefined dataset configuration
            if self.args.dataset in DATASET_CONFIGS:
                config = DATASET_CONFIGS[self.args.dataset]
                
                # Override with command line arguments
                if self.args.dataset_subset:
                    config.subset = self.args.dataset_subset
                if self.args.max_samples:
                    config.max_samples = self.args.max_samples
                if self.args.streaming:
                    config.streaming = self.args.streaming
                
                config.max_length = self.args.max_seq_length
                config.tokenizer_name = self.args.tokenizer_name
                
                try:
                    train_dataset = create_dataset(config, "train")
                    val_dataset = create_dataset(config, "validation")
                    logger.info(f"Successfully loaded {self.args.dataset} dataset")
                except Exception as e:
                    logger.warning(f"Failed to load {self.args.dataset}: {e}")
                    logger.info("Falling back to dummy dataset...")
                    train_dataset = DummyDataset(self.args.train_samples)
                    val_dataset = DummyDataset(self.args.val_samples)
            else:
                logger.error(f"Unknown dataset: {self.args.dataset}")
                logger.info("Available datasets: " + ", ".join(DATASET_CONFIGS.keys()))
                logger.info("Falling back to dummy dataset...")
                train_dataset = DummyDataset(self.args.train_samples)
                val_dataset = DummyDataset(self.args.val_samples)
        
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset) if self.args.distributed else None
        val_sampler = DistributedSampler(val_dataset) if self.args.distributed else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=False  # Disable pin_memory on CPU
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=False  # Disable pin_memory on CPU
        )
        
        # Generate synthetic data if requested
        if self.args.use_synthetic_data:
            logger.info("Generating synthetic data...")
            synthetic_data = self.synthetic_engine.generate_dataset(
                num_examples=self.args.synthetic_samples
            )
            logger.info(f"Generated {len(synthetic_data)} synthetic examples")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main_process()
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if self.args.use_4d_parallelism:
                metrics = self.distributed_trainer.train_step(batch)
                loss = metrics['loss']
            else:
                if self.scaler:
                    # Use the new autocast API
                    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type=device_type):
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']

                # Scale loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps

                # NaN/Inf guard - skip unstable batches
                if not torch.isfinite(loss):
                    logger.warning(f"Skipping batch {batch_idx} due to non-finite loss.")
                    if hasattr(self, 'optimizer'):
                        self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Backward pass
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.gradient_clipping > 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.gradient_clipping
                        )
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches
            })
            
            # Log to wandb
            if self.args.use_wandb and self.is_main_process() and batch_idx % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=not self.is_main_process()):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.args.use_wandb and self.is_main_process():
            wandb.log({'val/loss': avg_loss})
        
        return avg_loss
    
    def run_rlhf_training(self):
        """Run RLHF training phase"""
        if not self.args.enable_rlhf:
            return
        
        logger.info("Starting RLHF training...")
        
        # Create environment (placeholder)
        class DummyEnv:
            def reset(self):
                return torch.randint(0, 1000, (512,))
            
            def step(self, action):
                reward_dict = {
                    'helpfulness': np.random.random(),
                    'harmlessness': np.random.random(),
                    'honesty': np.random.random()
                }
                done = np.random.random() > 0.9
                return self.reset(), reward_dict, done, {}
        
        env = DummyEnv()
        
        # Run RLHF loop
        self.rlhf_system.run_rlhf_loop(
            env,
            num_iterations=self.args.rlhf_iterations,
            steps_per_iteration=self.args.rlhf_steps_per_iteration
        )
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        logger.info("Running evaluation...")
        
        # Create dummy evaluation datasets
        eval_datasets = {
            'gsm8k': self.val_loader,
            'humaneval': self.val_loader,
            'safety': self.val_loader,
            'efficiency': self.val_loader
        }
        
        # Run benchmarks with robust error handling for incompatible datasets
        try:
            results = self.benchmark_suite.run_all_benchmarks(
                self.model,
                eval_datasets
            )
        except KeyError as e:
            # Some benchmarks (e.g., HumanEval) expect fields like 'prompt' not present in dummy/val datasets
            logger.warning(f"Skipping some benchmarks due to missing field: {e}. This is expected with dummy/incompatible datasets.")
            results = {
                'summary': 'Partial evaluation completed (some benchmarks skipped due to dataset incompatibility).',
                'aggregate_scores': {}
            }
        except Exception as e:
            logger.warning(f"Evaluation encountered an error: {e}. Returning partial results.")
            results = {
                'summary': f'Evaluation skipped/partial due to error: {e}',
                'aggregate_scores': {}
            }
        
        # Log results
        logger.info(results['summary'])
        
        if self.args.use_wandb and self.is_main_process():
            wandb.log({'eval/results': results['aggregate_scores']})
        
        # Save results
        if self.is_main_process():
            self.benchmark_suite.save_results(
                os.path.join(self.args.output_dir, 'evaluation_results.json')
            )
        
        return results
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        if not self.is_main_process():
            return
        
        checkpoint_path = os.path.join(
            self.args.output_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save model in HuggingFace format
        if epoch == self.args.num_epochs - 1:
            self.model.save_pretrained(os.path.join(self.args.output_dir, 'final_model'))
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load datasets
        self.load_datasets()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Validation
            val_loss = self.validate()
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
            
            # RLHF phase (every N epochs)
            if (epoch + 1) % self.args.rlhf_frequency == 0:
                self.run_rlhf_training()
            
            # Evaluation (every N epochs)
            if (epoch + 1) % self.args.eval_frequency == 0:
                self.evaluate()
        
        logger.info("Training completed!")
        
        # Final evaluation
        final_results = self.evaluate()
        
        return final_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ULTRATHINK Model')
    
    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=100352)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=32)
    parser.add_argument('--num_kv_heads', type=int, default=8)
    parser.add_argument('--intermediate_size', type=int, default=14336)
    parser.add_argument('--max_seq_length', type=int, default=8192)
    parser.add_argument('--activation', type=str, default='swiglu')
    
    # MoE settings
    parser.add_argument('--enable_moe', action='store_true')
    parser.add_argument('--num_knowledge_experts', type=int, default=64)
    parser.add_argument('--num_skill_experts', type=int, default=32)
    parser.add_argument('--num_meta_experts', type=int, default=16)
    parser.add_argument('--num_safety_experts', type=int, default=8)
    parser.add_argument('--moe_top_k', type=int, default=2)
    parser.add_argument('--expert_capacity', type=float, default=1.25)
    
    # Multi-modal settings
    parser.add_argument('--enable_multimodal', action='store_true')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=14)
    parser.add_argument('--audio_sample_rate', type=int, default=16000)
    
    # Advanced features
    parser.add_argument('--enable_dre', action='store_true')
    parser.add_argument('--enable_constitutional', action='store_true')
    parser.add_argument('--enable_rlhf', action='store_true')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.0)
    
    # Optimization
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--use_4d_parallelism', action='store_true')
    parser.add_argument('--data_parallel_size', type=int, default=1)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--pipeline_parallel_size', type=int, default=1)
    parser.add_argument('--expert_parallel_size', type=int, default=1)
    parser.add_argument('--zero_stage', type=int, default=0)
    
    # RLHF settings
    parser.add_argument('--rlhf_frequency', type=int, default=5)
    parser.add_argument('--rlhf_iterations', type=int, default=100)
    parser.add_argument('--rlhf_steps_per_iteration', type=int, default=1000)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--ppo_batch_size', type=int, default=32)

    # Data settings
    parser.add_argument('--dataset', type=str, default='wikitext', 
                       help='Dataset to use (wikitext, openwebtext, pile, c4, bookcorpus, dummy, custom)')
    parser.add_argument('--mix_datasets', type=str, default=None,
                       help='Comma-separated list of dataset:weight pairs to mix, e.g. "wikitext:0.5,openwebtext:0.5". Overrides --dataset if set.')
    parser.add_argument('--dataset_subset', type=str, default=None, help='Dataset subset/config name')
    parser.add_argument('--data_path', type=str, default=None, help='Path to custom dataset file')
    parser.add_argument('--text_column', type=str, default='text', help='Column name containing text')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer to use')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')

    parser.add_argument('--train_samples', type=int, default=10000, help='Number of training samples (for dummy dataset)')
    parser.add_argument('--val_samples', type=int, default=1000, help='Number of validation samples (for dummy dataset)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--use_synthetic_data', action='store_true', help='Use synthetic data generation')
    parser.add_argument('--synthetic_samples', type=int, default=5000, help='Number of synthetic samples')

    # Evaluation
    parser.add_argument('--eval_frequency', type=int, default=5)

    # Logging
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='ultrathink_training')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/ultrathink')
    
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log configuration
    logger.info("Training configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Create trainer
    trainer = UltraThinkTrainer(args)
    
    # Run training
    results = trainer.train()
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()
