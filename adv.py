
import os
import math
import time
import json
import random
import logging
import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from contextlib import contextmanager, nullcontext
from functools import partial
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    import tiktoken
    import sentencepiece as spm
    from transformers import AutoTokenizer
    ADVANCED_TOKENIZERS = True
except ImportError:
    ADVANCED_TOKENIZERS = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import numpy as np
    import datasets
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== Configuration System ====================
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50304  # GPT-2 vocab size (divisible by 64)
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None  # For Grouped Query Attention
    rotary_dim: Optional[int] = None  # RoPE dimensions
    intermediate_size: Optional[int] = None  # FFN intermediate size
    activation: str = "swiglu"  # swiglu, gelu, relu
    norm_type: str = "rmsnorm"  # rmsnorm, layernorm
    norm_eps: float = 1e-5
    dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.1
    embed_dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_cache: bool = True
    rope_scaling: Optional[Dict] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    flash_attention: bool = True
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.rotary_dim is None:
            self.rotary_dim = self.n_embd // self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = int(self.n_embd * 8/3) if self.activation == "swiglu" else self.n_embd * 4

@dataclass 
class DataConfig:
    """Data configuration"""
    dataset_path: str = "data/train.txt"
    tokenizer_path: Optional[str] = None
    tokenizer_type: str = "char"  # char, tiktoken, sentencepiece, huggingface
    max_length: int = 2048
    streaming: bool = False
    num_workers: int = 4
    shuffle_buffer_size: int = 10000
    validation_split: float = 0.1
    pack_sequences: bool = True
    pad_token_id: Optional[int] = None
    preprocessing_num_workers: Optional[int] = None

@dataclass
class OptimConfig:
    """Optimization configuration"""
    optimizer: str = "adamw"  # adamw, lion, sam_adamw
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    gradient_centralization: bool = False
    clip_grad_norm: float = 1.0
    warmup_steps: int = 2000
    lr_scheduler: str = "cosine"  # cosine, linear, polynomial
    lr_scheduler_kwargs: Dict = field(default_factory=dict)
    total_steps: int = 100000
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 8
    micro_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # fp16, bf16, fp32
    compile_model: bool = False
    use_fsdp: bool = False
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10
    max_eval_batches: int = 100
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str = "advanced_gpt"
    output_dir: str = "outputs"
    logging_dir: Optional[str] = None
    use_wandb: bool = False
    wandb_project: str = "advanced-gpt"
    wandb_tags: List[str] = field(default_factory=list)
    sample_every: int = 1000
    sample_length: int = 200
    sample_temperature: float = 0.8
    sample_top_k: int = 50
    sample_top_p: float = 0.9
    
@dataclass
class Config:
    """Main configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig) 
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

# ==================== Modern Model Components ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        self._update_cos_sin_cache(x, seq_len)
        return self._cos_cached[:, :, :seq_len, ...], self._sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply rotary positional embedding to query and key tensors."""
    if position_ids is not None:
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention mechanism"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.num_kv_heads = config.n_kv_head
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = RotaryPositionalEmbedding(
            config.rotary_dim, max_position_embeddings=config.n_positions
        ) if config.rotary_dim else None
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key/value heads to match query heads"""
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        past_key_value = (key_states, value_states) if use_cache else None
        return attn_output, past_key_value

class TransformerBlock(nn.Module):
    """Advanced Transformer Block"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Normalization
        if config.norm_type == "rmsnorm":
            self.input_layernorm = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.post_attention_layernorm = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        
        # Attention
        self.self_attn = GroupedQueryAttention(config)
        
        # MLP
        if config.activation == "swiglu":
            self.mlp = SwiGLU(config.n_embd, config.intermediate_size, bias=config.mlp_bias)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.intermediate_size, bias=config.mlp_bias),
                nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                nn.Linear(config.intermediate_size, config.n_embd, bias=config.mlp_bias)
            )
        
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_output, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value,
        )
        
        # Residual connection
        hidden_states = residual + self.residual_dropout(attn_output)
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.residual_dropout(mlp_output)
        
        return hidden_states, present_key_value

class AdvancedGPTModel(nn.Module):
    """Advanced GPT Model with modern improvements"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        if not hasattr(config, 'rotary_dim') or config.rotary_dim is None:
            self.embed_positions = nn.Embedding(config.n_positions, config.n_embd)
        else:
            self.embed_positions = None
            
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final norm
        if config.norm_type == "rmsnorm":
            self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.norm = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        
        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using scaled initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=True,
    ):
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        
        batch_size, seq_length = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        if self.embed_positions is not None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            position_embeds = self.embed_positions(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
            
        hidden_states = self.embed_dropout(hidden_states)

        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=input_ids.device),
                diagonal=1
            )

        # Transform through layers
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)
        present_key_values = []
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                present_key_values.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': hidden_states,
        }

# ==================== Advanced Optimizers ====================

class Lion(torch.optim.Optimizer):
    """Lion optimizer implementation"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                update = exp_avg.sign()
                p.add_(update, alpha=-group['lr'])

                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def get_optimizer(model, config: OptimConfig):
    """Get optimizer based on configuration"""
    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'bias' in name or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    if config.optimizer.lower() == "adamw":
        optimizer = AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer.lower() == "lion":
        optimizer = Lion(
            optim_groups,
            lr=config.learning_rate * 0.3,  # Lion typically needs lower LR
            betas=(config.beta1, config.beta2),
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer

# ==================== Advanced Data Loading ====================

class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large text files"""
    def __init__(self, file_paths, tokenizer, max_length, shuffle_buffer_size=1000):
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        buffer = []
        
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                current_tokens = []
                
                for line in f:
                    line_tokens = self.tokenizer.encode(line.strip())
                    current_tokens.extend(line_tokens)
                    
                    # Yield complete sequences
                    while len(current_tokens) >= self.max_length + 1:
                        sequence = current_tokens[:self.max_length + 1]
                        buffer.append({
                            'input_ids': torch.tensor(sequence[:-1]),
                            'labels': torch.tensor(sequence[1:])
                        })
                        current_tokens = current_tokens[self.max_length:]
                        
                        # Shuffle buffer
                        if len(buffer) >= self.shuffle_buffer_size:
                            random.shuffle(buffer)
                            for item in buffer:
                                yield item
                            buffer = []
                
                # Handle remaining tokens
                if len(current_tokens) > 1:
                    buffer.append({
                        'input_ids': torch.tensor(current_tokens[:-1]),
                        'labels': torch.tensor(current_tokens[1:])
                    })

        # Yield remaining items
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item

class AdvancedTokenizer:
    """Advanced tokenizer supporting multiple backends"""
    def __init__(self, tokenizer_path=None, tokenizer_type="char", vocab_size=None):
        self.tokenizer_type = tokenizer_type
        self.tokenizer_path = tokenizer_path
        
        if tokenizer_type == "tiktoken" and ADVANCED_TOKENIZERS:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
        elif tokenizer_type == "huggingface" and ADVANCED_TOKENIZERS and tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.vocab_size = len(self.tokenizer)
        else:
            # Fallback to character tokenizer
            self.tokenizer = None
            self.vocab_size = vocab_size or 256

    def encode(self, text):
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # Simple char tokenizer
            return [ord(c) for c in text if ord(c) < self.vocab_size]

    def decode(self, tokens):
        if self.tokenizer:
            return self.tokenizer.decode(tokens)
        else:
            return ''.join([chr(t) for t in tokens if 0 <= t < 256])

# ==================== Training Infrastructure ====================

class AdvancedTrainer:
    """Advanced training class with all modern features"""
    
    def __init__(self, config: Config):
        self.config = config
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Setup directories
        self.setup_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Setup model and training components
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()
        
        # Setup distributed training if needed
        self.setup_distributed()
        
    def setup_directories(self):
        """Setup output directories"""
        os.makedirs(self.config.experiment.output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(self.config.experiment.output_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({
                'model': self.config.model.__dict__,
                'data': self.config.data.__dict__,
                'optim': self.config.optim.__dict__,
                'training': self.config.training.__dict__,
                'experiment': self.config.experiment.__dict__,
            }, f, default_flow_style=False)
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        log_dir = self.config.experiment.logging_dir or os.path.join(self.config.experiment.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Weights & Biases
        if self.config.experiment.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.experiment.wandb_project,
                name=self.config.experiment.experiment_name,
                config=self._config_to_dict(),
                tags=self.config.experiment.wandb_tags,
            )
            self.wandb = wandb
        else:
            self.wandb = None
            
    def _config_to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            **self.config.model.__dict__,
            **self.config.data.__dict__,
            **self.config.optim.__dict__,
            **self.config.training.__dict__,
            **self.config.experiment.__dict__,
        }
    
    def setup_model(self):
        """Setup model with compilation and FSDP if needed"""
        self.model = AdvancedGPTModel(self.config.model)
        
        # Model compilation
        if self.config.training.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
        
        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # FSDP setup
        if self.config.training.use_fsdp and torch.cuda.device_count() > 1:
            self.setup_fsdp()
        
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def setup_fsdp(self):
        """Setup Fully Sharded Data Parallel"""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            MixedPrecision,
            BackwardPrefetch,
            CPUOffload,
        )
        
        # Mixed precision policy
        if self.config.training.mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.config.training.mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None
        
        # Auto wrap policy
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerBlock},
        )
        
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
        )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = get_optimizer(self.model, self.config.optim)
        
        # Learning rate scheduler
        if self.config.optim.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.optim.total_steps - self.config.optim.warmup_steps,
                **self.config.optim.lr_scheduler_kwargs
            )
        elif self.config.optim.lr_scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.optim.total_steps - self.config.optim.warmup_steps,
                **self.config.optim.lr_scheduler_kwargs
            )
        else:
            self.scheduler = None
        
        # Warmup scheduler
        if self.config.optim.warmup_steps > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.config.optim.warmup_steps
            )
            
            if self.scheduler:
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, self.scheduler],
                    milestones=[self.config.optim.warmup_steps]
                )
            else:
                self.scheduler = warmup_scheduler
        
        # Gradient scaler for mixed precision
        if self.config.training.mixed_precision in ["fp16", "bf16"]:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def setup_data(self):
        """Setup data loading"""
        self.tokenizer = AdvancedTokenizer(
            tokenizer_path=self.config.data.tokenizer_path,
            tokenizer_type=self.config.data.tokenizer_type,
            vocab_size=self.config.model.vocab_size
        )
        
        # Update model vocab size if needed
        if self.tokenizer.vocab_size != self.config.model.vocab_size:
            logger.warning(f"Updating vocab size from {self.config.model.vocab_size} to {self.tokenizer.vocab_size}")
            self.config.model.vocab_size = self.tokenizer.vocab_size
            # Resize embeddings
            old_embeddings = self.model.embed_tokens
            new_embeddings = nn.Embedding(self.tokenizer.vocab_size, self.config.model.n_embd)
            new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
            self.model.embed_tokens = new_embeddings
        
        if self.config.data.streaming:
            train_dataset = StreamingTextDataset(
                file_paths=self.config.data.dataset_path,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_length,
                shuffle_buffer_size=self.config.data.shuffle_buffer_size
            )
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.training.micro_batch_size or self.config.training.batch_size,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.training.dataloader_pin_memory,
            )
        else:
            # Load dataset into memory
            with open(self.config.data.dataset_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = self.tokenizer.encode(text)
            split_idx = int(len(tokens) * (1 - self.config.data.validation_split))
            
            train_tokens = tokens[:split_idx]
            val_tokens = tokens[split_idx:]
            
            self.train_dataset = TokenDataset(train_tokens, self.config.data.max_length)
            self.val_dataset = TokenDataset(val_tokens, self.config.data.max_length)
            
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.training.micro_batch_size or self.config.training.batch_size,
                shuffle=True,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.training.dataloader_pin_memory,
                drop_last=self.config.training.dataloader_drop_last,
            )
            
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.training.micro_batch_size or self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.training.dataloader_pin_memory,
            )
    
    def setup_distributed(self):
        """Setup distributed training if needed"""
        if torch.cuda.device_count() > 1 and not self.config.training.use_fsdp:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    def save_checkpoint(self, checkpoint_dir=None):
        """Save model checkpoint"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.config.experiment.output_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Prepare state dict
        model_state = self.model.state_dict()
        if isinstance(self.model, (nn.DataParallel, FSDP)):
            model_state = self.model.module.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': self._config_to_dict(),
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save tokenizer info
        tokenizer_info = {
            'type': self.tokenizer.tokenizer_type,
            'path': self.tokenizer.tokenizer_path,
            'vocab_size': self.tokenizer.vocab_size,
        }
        with open(os.path.join(checkpoint_dir, "tokenizer.json"), 'w') as f:
            json.dump(tokenizer_info, f)
        
        logger.info(f"Checkpoint saved at {checkpoint_dir}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit"""
        if self.config.training.save_total_limit <= 0:
            return
        
        checkpoints = []
        for item in os.listdir(self.config.experiment.output_dir):
            if item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[-1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        checkpoints.sort(reverse=True)  # Most recent first
        
        for step, checkpoint_name in checkpoints[self.config.training.save_total_limit:]:
            checkpoint_path = os.path.join(self.config.experiment.output_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, (nn.DataParallel, FSDP)):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from step {self.global_step}")
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        if not hasattr(self, 'val_dataloader'):
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.val_dataloader):
            if batch_idx >= self.config.training.max_eval_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
        }
    
    @torch.no_grad()
    def generate_sample(self):
        """Generate text sample for monitoring"""
        self.model.eval()
        
        # Start with a random token or BOS
        start_tokens = torch.randint(0, min(1000, self.tokenizer.vocab_size), (1, 1), device=self.device)
        
        generated = start_tokens.clone()
        
        for _ in range(self.config.experiment.sample_length):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(input_ids=generated, use_cache=False)
                logits = outputs['logits']
            
            # Apply sampling strategy
            next_token_logits = logits[0, -1, :] / self.config.experiment.sample_temperature
            
            # Top-k filtering
            if self.config.experiment.sample_top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, self.config.experiment.sample_top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if self.config.experiment.sample_top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > self.config.experiment.sample_top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            # Keep only recent context to avoid memory issues
            if generated.size(1) > self.config.data.max_length:
                generated = generated[:, -self.config.data.max_length:]
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        self.model.train()
        return generated_text
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self.load_checkpoint(self.config.training.resume_from_checkpoint)
        
        self.model.train()
        
        # Training state
        accumulation_steps = 0
        running_loss = 0
        
        # Progress tracking
        from tqdm import tqdm
        progress_bar = tqdm(
            total=self.config.optim.total_steps,
            initial=self.global_step,
            desc="Training"
        )
        
        # Training loop
        train_iter = iter(self.train_dataloader)
        
        while self.global_step < self.config.optim.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / self.config.training.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            running_loss += loss.item()
            accumulation_steps += 1
            
            # Gradient accumulation step
            if accumulation_steps >= self.config.training.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                if self.config.optim.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optim.clip_grad_norm
                    )
                
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
                if self.global_step % self.config.training.log_interval == 0:
                    avg_loss = running_loss / self.config.training.log_interval
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    # Console logging
                    logger.info(
                        f"Step {self.global_step:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Step time: {time.time():.2f}s"
                    )
                    
                    # TensorBoard logging
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/learning_rate', lr, self.global_step)
                    
                    # Wandb logging
                    if self.wandb:
                        self.wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/step': self.global_step,
                        })
                    
                    running_loss = 0
                
                # Evaluation
                if self.global_step % self.config.training.eval_interval == 0 and self.global_step > 0:
                    eval_metrics = self.evaluate()
                    
                    if eval_metrics:
                        logger.info(f"Eval | Loss: {eval_metrics['eval_loss']:.4f} | PPL: {eval_metrics['eval_perplexity']:.2f}")
                        
                        # TensorBoard logging
                        for key, value in eval_metrics.items():
                            self.writer.add_scalar(f'eval/{key}', value, self.global_step)
                        
                        # Wandb logging
                        if self.wandb:
                            self.wandb.log({f'eval/{k}': v for k, v in eval_metrics.items()})
                        
                        # Save best model
                        if eval_metrics['eval_loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(
                                os.path.join(self.config.experiment.output_dir, "best_model")
                            )
                
                # Generate sample
                if self.global_step % self.config.experiment.sample_every == 0 and self.global_step > 0:
                    sample_text = self.generate_sample()
                    logger.info(f"\n--- SAMPLE ---\n{sample_text}\n--------------")
                    
                    if self.wandb:
                        self.wandb.log({'sample': sample_text, 'step': self.global_step})
                
                # Save checkpoint
                if self.global_step % self.config.training.checkpoint_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()
                
                self.global_step += 1
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Final save
        self.save_checkpoint(os.path.join(self.config.experiment.output_dir, "final_model"))
        
        # Cleanup
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()
        
        logger.info("Training completed!")

class TokenDataset(Dataset):
    """Simple token dataset for non-streaming training"""
    def __init__(self, tokens, max_length):
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

# ==================== Main Execution ====================

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced GPT Training")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model.n_layer", type=int, help="Number of layers")
    parser.add_argument("--model.n_embd", type=int, help="Embedding dimension")
    parser.add_argument("--model.n_head", type=int, help="Number of attention heads")
    parser.add_argument("--data.dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--training.batch_size", type=int, help="Batch size")
    parser.add_argument("--optim.learning_rate", type=float, help="Learning rate")
    parser.add_argument("--experiment.output_dir", type=str, help="Output directory")
    parser.add_argument("--experiment.use_wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    for arg, value in vars(args).items():
        if value is not None and '.' in arg:
            section, key = arg.split('.', 1)
            if hasattr(config, section):
                setattr(getattr(config, section), key, value)
    
    # Set random seed
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    
    # Create trainer and start training
    trainer = AdvancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()