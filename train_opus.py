#!/usr/bin/env python3
"""
Claude Opus 4 Scale Training System
Complete implementation with all modern features
"""

import os
import sys
import yaml
import json
import time
import math
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.architecture import AdvancedGPTModel, ModelConfig, TransformerBlock
from training.optimizers import get_optimizer, get_scheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Fallback character tokenizer"""
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos.get(t, '') for t in tokens])


class AdvancedTokenizer:
    """Advanced tokenizer with tiktoken support"""
    def __init__(self, tokenizer_type: str = "tiktoken", tokenizer_name: str = "gpt2", fallback_text: str = None):
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type == "tiktoken" and TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(tokenizer_name)
                self.vocab_size = self.tokenizer.n_vocab
                logger.info(f"Loaded tiktoken tokenizer: {tokenizer_name}, vocab_size: {self.vocab_size}")
            except Exception as e:
                logger.warning(f"Failed to load tiktoken: {e}, falling back to char tokenizer")
                self.tokenizer = SimpleTokenizer(fallback_text or "abcdefghijklmnopqrstuvwxyz")
                self.vocab_size = self.tokenizer.vocab_size
        else:
            self.tokenizer = SimpleTokenizer(fallback_text or "abcdefghijklmnopqrstuvwxyz")
            self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class TextDataset(Dataset):
    """Simple text dataset for training"""
    def __init__(self, tokens: List[int], max_length: int):
        self.tokens = tokens
        self.max_length = max_length

    def __len__(self):
        return max(0, len(self.tokens) - self.max_length)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.max_length + 1]
        return {
            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
            'labels': torch.tensor(chunk[1:], dtype=torch.long),
        }


class OpusTrainer:
    """Advanced trainer for Claude Opus scale models"""
    
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Setup
        self.setup_device()
        self.setup_directories()
        self.setup_logging()
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
    def load_config(self, config_path: str = None):
        """Load configuration from YAML files"""
        # Load model configs
        model_configs_path = "config/model_configs.yaml"
        training_configs_path = "config/training_configs.yaml"
        
        model_configs = {}
        training_configs = {}
        
        if os.path.exists(model_configs_path):
            with open(model_configs_path, 'r') as f:
                model_configs = yaml.safe_load(f)
        
        if os.path.exists(training_configs_path):
            with open(training_configs_path, 'r') as f:
                training_configs = yaml.safe_load(f)
        
        # Use debug config by default
        config_name = "debug"
        if config_name in training_configs:
            config = training_configs[config_name]
            model_config_name = config.get('model_config', 'small')
            if model_config_name in model_configs:
                config['model_config_dict'] = model_configs[model_config_name]
            return config
        
        # Fallback config
        return {
            'model_config': 'small',
            'model_config_dict': {
                'vocab_size': 50304,
                'n_positions': 2048,
                'n_embd': 768,
                'n_layer': 12,
                'n_head': 12,
                'n_kv_head': 4,
                'rotary_dim': 64,
                'intermediate_size': 3072,
                'activation': 'swiglu',
                'norm_type': 'rmsnorm',
                'norm_eps': 1e-5,
                'dropout': 0.0,
                'attention_dropout': 0.0,
                'residual_dropout': 0.1,
                'embed_dropout': 0.1,
                'tie_word_embeddings': True,
                'use_cache': True,
                'attention_bias': False,
                'mlp_bias': False,
                'flash_attention': True,
                'sliding_window': None,
                'rope_theta': 10000.0,
                'gradient_checkpointing': False,
                'max_position_embeddings': 2048
            },
            'data': {
                'dataset_path': 'input.txt',
                'tokenizer_type': 'tiktoken',
                'tokenizer_name': 'gpt2',
                'max_length': 1024,
                'validation_split': 0.1
            },
            'optim': {
                'optimizer': 'lion',
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'beta1': 0.95,
                'beta2': 0.98,
                'eps': 1e-8,
                'clip_grad_norm': 1.0,
                'warmup_steps': 100,
                'lr_scheduler': 'cosine',
                'total_steps': 1000
            },
            'training': {
                'batch_size': 4,
                'micro_batch_size': 2,
                'gradient_accumulation_steps': 2,
                'mixed_precision': 'bf16',
                'compile_model': False,
                'checkpoint_interval': 100,
                'eval_interval': 50,
                'log_interval': 10,
                'max_eval_batches': 10,
                'seed': 42,
                'save_total_limit': 3
            },
            'experiment': {
                'experiment_name': 'opus_training',
                'output_dir': 'outputs',
                'use_wandb': False,
                'sample_every': 100,
                'sample_length': 100,
                'sample_temperature': 0.8,
                'sample_top_k': 50,
                'sample_top_p': 0.9
            }
        }
    
    def setup_device(self):
        """Setup device and distributed training"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        seed = self.config['training']['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = self.config['experiment']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Weights & Biases
        if self.config['experiment']['use_wandb'] and WANDB_AVAILABLE:
            wandb.init(
                project=self.config['experiment'].get('wandb_project', 'opus-training'),
                name=self.config['experiment']['experiment_name'],
                config=self.config,
                tags=self.config['experiment'].get('wandb_tags', [])
            )
            self.wandb = wandb
        else:
            self.wandb = None
    
    def setup_data(self):
        """Setup data loading"""
        data_config = self.config['data']
        
        # Load text data
        dataset_path = data_config['dataset_path']
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} not found, creating dummy data")
            with open(dataset_path, 'w') as f:
                f.write("This is a sample text for training. " * 1000)
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Setup tokenizer
        self.tokenizer = AdvancedTokenizer(
            tokenizer_type=data_config['tokenizer_type'],
            tokenizer_name=data_config['tokenizer_name'],
            fallback_text=text
        )
        
        # Tokenize and split data
        tokens = self.tokenizer.encode(text)
        split_idx = int(len(tokens) * (1 - data_config['validation_split']))
        
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        
        # Create datasets
        max_length = data_config['max_length']
        self.train_dataset = TextDataset(train_tokens, max_length)
        self.val_dataset = TextDataset(val_tokens, max_length)
        
        # Create dataloaders
        batch_size = self.config['training']['micro_batch_size'] or self.config['training']['batch_size']
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Loaded dataset: {len(train_tokens)} train tokens, {len(val_tokens)} val tokens")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
    
    def setup_model(self):
        """Setup model"""
        model_config_dict = self.config['model_config_dict']
        model_config_dict['vocab_size'] = self.tokenizer.vocab_size
        
        model_config = ModelConfig(**model_config_dict)
        self.model = AdvancedGPTModel(model_config)
        self.model = self.model.to(self.device)
        
        # Model compilation
        if self.config['training'].get('compile_model', False) and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        optim_config = self.config['optim']
        
        # Create optimizer config object
        class OptimConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        optim_cfg = OptimConfig(**optim_config)
        
        self.optimizer = get_optimizer(self.model, optim_cfg)
        self.scheduler = get_scheduler(self.optimizer, optim_cfg)
        
        # Mixed precision scaler
        if self.config['training']['mixed_precision'] in ['fp16', 'bf16'] and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        max_batches = self.config['training']['max_eval_batches']
        
        for batch_idx, batch in enumerate(self.val_dataloader):
            if batch_idx >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))
        
        self.model.train()
        return {'eval_loss': avg_loss, 'eval_perplexity': perplexity}
    
    @torch.no_grad()
    def generate_sample(self):
        """Generate text sample"""
        self.model.eval()
        
        # Start with random token
        start_tokens = torch.randint(0, min(1000, self.tokenizer.vocab_size), (1, 1), device=self.device)
        generated = start_tokens.clone()
        
        sample_length = self.config['experiment']['sample_length']
        temperature = self.config['experiment']['sample_temperature']
        top_k = self.config['experiment']['sample_top_k']
        top_p = self.config['experiment']['sample_top_p']
        
        for _ in range(sample_length):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(input_ids=generated, use_cache=False)
                logits = outputs['logits']
            
            # Apply sampling
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            # Keep context window manageable
            if generated.size(1) > self.config['data']['max_length']:
                generated = generated[:, -self.config['data']['max_length']:]
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        self.model.train()
        return generated_text
    
    def save_checkpoint(self, checkpoint_dir: str = None):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        # Cleanup old checkpoints
        self.cleanup_checkpoints()
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints"""
        save_total_limit = self.config['training']['save_total_limit']
        if save_total_limit <= 0:
            return
        
        checkpoints = []
        for item in os.listdir(self.output_dir):
            if item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[-1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        checkpoints.sort(reverse=True)
        
        for step, checkpoint_name in checkpoints[save_total_limit:]:
            checkpoint_path = os.path.join(self.output_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_name}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        self.model.train()
        
        # Training state
        accumulation_steps = 0
        running_loss = 0
        total_steps = self.config['optim']['total_steps']
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        
        # Progress tracking
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_steps, initial=self.global_step, desc="Training")
        except ImportError:
            progress_bar = None
        
        train_iter = iter(self.train_dataloader)
        
        while self.global_step < total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss'] / gradient_accumulation_steps
            else:
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            running_loss += loss.item()
            accumulation_steps += 1
            
            # Gradient accumulation step
            if accumulation_steps >= gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                clip_grad_norm = self.config['optim']['clip_grad_norm']
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                accumulation_steps = 0
                
                # Logging
                log_interval = self.config['training']['log_interval']
                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(f"Step {self.global_step:,} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    
                    # TensorBoard
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                    
                    # Wandb
                    if self.wandb:
                        self.wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/step': self.global_step,
                        })
                    
                    running_loss = 0
                
                # Evaluation
                eval_interval = self.config['training']['eval_interval']
                if self.global_step % eval_interval == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    
                    logger.info(f"Eval | Loss: {eval_metrics['eval_loss']:.4f} | PPL: {eval_metrics['eval_perplexity']:.2f}")
                    
                    # TensorBoard
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{key}', value, self.global_step)
                    
                    # Wandb
                    if self.wandb:
                        self.wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()})
                    
                    # Save best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(os.path.join(self.output_dir, "best_model"))
                
                # Generate sample
                sample_every = self.config['experiment']['sample_every']
                if self.global_step % sample_every == 0 and self.global_step > 0:
                    sample_text = self.generate_sample()
                    logger.info(f"\n--- SAMPLE ---\n{sample_text}\n--------------")
                    
                    if self.wandb:
                        self.wandb.log({'sample': sample_text, 'step': self.global_step})
                
                # Save checkpoint
                checkpoint_interval = self.config['training']['checkpoint_interval']
                if self.global_step % checkpoint_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                if progress_bar:
                    progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        # Final save
        self.save_checkpoint(os.path.join(self.output_dir, "final_model"))
        
        # Cleanup
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()
        
        logger.info("Training completed!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced GPT Training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model", type=str, default="small", help="Model size")
    parser.add_argument("--data", type=str, default="input.txt", help="Dataset path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = OpusTrainer(config_path=args.config)
    
    # Override config with CLI args
    if args.data:
        trainer.config['data']['dataset_path'] = args.data
    if args.output_dir:
        trainer.config['experiment']['output_dir'] = args.output_dir
    if args.wandb:
        trainer.config['experiment']['use_wandb'] = True
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
