"""
Advanced Transformer Architecture Components
State-of-the-art implementations for Claude Opus 4 scale training
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False


@dataclass
class ModelConfig:
    """Advanced model configuration for Claude Opus 4 scale models"""
    vocab_size: int = 100352
    n_positions: int = 8192
    n_embd: int = 4096
    n_layer: int = 64
    n_head: int = 32
    n_kv_head: int = 8
    rotary_dim: int = 128
    intermediate_size: int = 14336
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    residual_dropout: float = 0.05
    embed_dropout: float = 0.05
    tie_word_embeddings: bool = False
    use_cache: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    flash_attention: bool = True
    sliding_window: Optional[int] = 4096
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict] = None
    gradient_checkpointing: bool = True
    max_position_embeddings: int = 8192


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding with dynamic scaling"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
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


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.n_embd
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with Flash Attention support"""
    
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
        )
        
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
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for generation
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)

        # Flash Attention if available
        if FLASH_ATTENTION_AVAILABLE and self.config.flash_attention:
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2), 
                value_states.transpose(1, 2),
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                causal=True
            ).transpose(1, 2)
        else:
            # Standard attention
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
            self.input_layernorm = RMSNorm(config.n_embd, eps=float(config.norm_eps))
            self.post_attention_layernorm = RMSNorm(config.n_embd, eps=float(config.norm_eps))
        else:
            self.input_layernorm = nn.LayerNorm(config.n_embd, eps=float(config.norm_eps))
            self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=float(config.norm_eps))
        
        # Attention
        self.self_attn = GroupedQueryAttention(config)
        
        # MLP
        if config.activation == "swiglu":
            self.mlp = SwiGLU(config)
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
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final norm
        if config.norm_type == "rmsnorm":
            self.norm = RMSNorm(config.n_embd, eps=float(config.norm_eps))
        else:
            self.norm = nn.LayerNorm(config.n_embd, eps=float(config.norm_eps))
        
        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None
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

    def forward(self, input_ids, attention_mask=None, past_key_values=None, labels=None, use_cache=None):
        batch_size, seq_length = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
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
            if self.config.gradient_checkpointing and self.training:
                hidden_states, present_key_value = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_cache,
                    past_key_value,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_key_value = layer(
                    hidden_states, attention_mask, use_cache, past_key_value
                )
            
            if use_cache:
                present_key_values.append(present_key_value)

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

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': hidden_states,
        }
