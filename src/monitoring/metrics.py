"""Real-time training metrics and monitoring"""
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Track comprehensive training metrics"""
    
    # Loss metrics
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    
    # Performance metrics
    tokens_per_second: List[float] = field(default_factory=list)
    samples_per_second: List[float] = field(default_factory=list)
    gpu_memory_allocated: List[float] = field(default_factory=list)
    gpu_memory_reserved: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    
    # Gradient metrics
    grad_norm: List[float] = field(default_factory=list)
    param_norm: List[float] = field(default_factory=list)
    grad_variance: List[float] = field(default_factory=list)
    
    # Learning rate
    lr_history: List[float] = field(default_factory=list)
    
    # Time tracking
    step_times: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Step counter
    global_step: int = 0
    
    def log_step(
        self,
        loss: float,
        lr: float,
        model: torch.nn.Module,
        batch_size: int = 1,
        seq_length: int = 512,
        step_time: Optional[float] = None
    ):
        """Log metrics for current step"""
        self.global_step += 1
        self.train_loss.append(loss)
        self.lr_history.append(lr)
        
        if step_time is not None:
            self.step_times.append(step_time)
            # Calculate throughput
            tokens = batch_size * seq_length
            self.tokens_per_second.append(tokens / step_time)
            self.samples_per_second.append(batch_size / step_time)
        
        # GPU metrics
        if torch.cuda.is_available():
            self.gpu_memory_allocated.append(
                torch.cuda.memory_allocated() / 1e9  # GB
            )
            self.gpu_memory_reserved.append(
                torch.cuda.memory_reserved() / 1e9
            )
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.append(util.gpu)
            except (ImportError, Exception):
                pass
        
        # Gradient norms
        try:
            total_norm = 0.0
            param_norm = 0.0
            grad_list = []
            
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = p.grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    grad_list.append(grad_norm)
                
                if p.data is not None:
                    param_norm += p.data.norm(2).item() ** 2
            
            self.grad_norm.append(total_norm ** 0.5)
            self.param_norm.append(param_norm ** 0.5)
            
            if grad_list:
                self.grad_variance.append(np.var(grad_list))
        except Exception as e:
            logger.debug(f"Failed to compute gradient metrics: {e}")
    
    def log_validation(self, val_loss: float):
        """Log validation metrics"""
        self.val_loss.append(val_loss)
    
    def get_summary(self, window: int = 100) -> Dict[str, float]:
        """Get summary statistics over recent window"""
        def get_recent_mean(data: List[float], window: int) -> float:
            if not data:
                return 0.0
            recent = data[-window:]
            return sum(recent) / len(recent) if recent else 0.0
        
        summary = {
            'avg_train_loss': get_recent_mean(self.train_loss, window),
            'avg_lr': get_recent_mean(self.lr_history, window),
            'avg_tokens_per_sec': get_recent_mean(self.tokens_per_second, window),
            'avg_grad_norm': get_recent_mean(self.grad_norm, window),
            'avg_gpu_memory_gb': get_recent_mean(self.gpu_memory_allocated, window),
            'global_step': self.global_step
        }
        
        if self.val_loss:
            summary['latest_val_loss'] = self.val_loss[-1]
        
        return summary
    
    def export_tensorboard(self, writer, step: Optional[int] = None):
        """Export to TensorBoard"""
        if step is None:
            step = self.global_step
        
        if self.train_loss:
            writer.add_scalar('Loss/train', self.train_loss[-1], step)
        
        if self.val_loss:
            writer.add_scalar('Loss/validation', self.val_loss[-1], step)
        
        if self.lr_history:
            writer.add_scalar('Learning_Rate', self.lr_history[-1], step)
        
        if self.tokens_per_second:
            writer.add_scalar('Performance/tokens_per_second', self.tokens_per_second[-1], step)
        
        if self.gpu_memory_allocated:
            writer.add_scalar('GPU/memory_allocated_gb', self.gpu_memory_allocated[-1], step)
        
        if self.grad_norm:
            writer.add_scalar('Gradients/norm', self.grad_norm[-1], step)
        
        if self.param_norm:
            writer.add_scalar('Parameters/norm', self.param_norm[-1], step)
    
    def export_wandb(self, step: Optional[int] = None):
        """Export to Weights & Biases"""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed, skipping export")
            return
        
        if step is None:
            step = self.global_step
        
        metrics_dict = {}
        
        if self.train_loss:
            metrics_dict['train/loss'] = self.train_loss[-1]
        
        if self.val_loss:
            metrics_dict['val/loss'] = self.val_loss[-1]
        
        if self.lr_history:
            metrics_dict['train/learning_rate'] = self.lr_history[-1]
        
        if self.tokens_per_second:
            metrics_dict['perf/tokens_per_second'] = self.tokens_per_second[-1]
        
        if self.gpu_memory_allocated:
            metrics_dict['gpu/memory_gb'] = self.gpu_memory_allocated[-1]
        
        if self.grad_norm:
            metrics_dict['train/grad_norm'] = self.grad_norm[-1]
        
        if metrics_dict:
            wandb.log(metrics_dict, step=step)


class MetricsLogger:
    """Enhanced metrics logger with rolling statistics"""
    
    def __init__(self, window_size: int = 100, log_freq: int = 10):
        self.window_size = window_size
        self.log_freq = log_freq
        self.metrics = TrainingMetrics()
        
        # Rolling windows for smoothing
        self.loss_window = deque(maxlen=window_size)
        self.lr_window = deque(maxlen=window_size)
        
        self.step_count = 0
        self.last_log_time = time.time()
    
    def log(
        self,
        loss: float,
        lr: float,
        model: torch.nn.Module,
        batch_size: int = 1,
        seq_length: int = 512
    ):
        """Log metrics and print periodically"""
        current_time = time.time()
        step_time = current_time - self.last_log_time
        self.last_log_time = current_time
        
        # Update metrics
        self.metrics.log_step(loss, lr, model, batch_size, seq_length, step_time)
        self.loss_window.append(loss)
        self.lr_window.append(lr)
        self.step_count += 1
        
        # Log periodically
        if self.step_count % self.log_freq == 0:
            avg_loss = sum(self.loss_window) / len(self.loss_window)
            avg_lr = sum(self.lr_window) / len(self.lr_window)
            
            log_msg = (
                f"Step {self.step_count} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {avg_lr:.2e} | "
            )
            
            if self.metrics.tokens_per_second:
                log_msg += f"Tokens/s: {self.metrics.tokens_per_second[-1]:.0f} | "
            
            if self.metrics.gpu_memory_allocated:
                log_msg += f"GPU Mem: {self.metrics.gpu_memory_allocated[-1]:.2f}GB"
            
            logger.info(log_msg)
    
    def get_metrics(self) -> TrainingMetrics:
        """Get the underlying metrics object"""
        return self.metrics
