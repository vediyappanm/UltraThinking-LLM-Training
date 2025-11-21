"""
Rotary + NTK-Scaled + Continuous RoPE
Dynamically scalable positional encodings for 4K → 1M token contexts
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class RotaryEmbedding(nn.Module):
    """
    Standard Rotary Position Embedding (RoPE)
    Base implementation for positional encoding
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
        """Update cosine and sine cache"""
        if seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            
            # Compute position indices
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq)
            
            # Concatenate for full rotation
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute cos and sin
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            seq_len: Sequence length
        
        Returns:
            cos, sin: Cosine and sine embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        self._update_cos_sin_cache(x, seq_len)
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


class NTKScaledRotaryEmbedding(RotaryEmbedding):
    """
    NTK-Scaled Rotary Embedding
    Dynamically scales base frequency for longer contexts
    Used in GPT-4-Turbo, Gemini 1.5
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        
        # Compute NTK-scaled base
        if ntk_alpha > 1.0:
            # NTK scaling: base' = base * alpha^(d/(d-2))
            scaled_base = base * (ntk_alpha ** (dim / (dim - 2)))
        else:
            scaled_base = base
        
        super().__init__(dim, max_position_embeddings, scaled_base, device)
        
        logger.info(f"NTK-Scaled RoPE: base={base:.0f} -> {scaled_base:.0f}, alpha={ntk_alpha}")
    
    def _update_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
        """Update cache with NTK scaling"""
        # Apply scaling factor to sequence length
        effective_seq_len = int(seq_len / self.scaling_factor)
        
        if effective_seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = effective_seq_len
            
            # Compute scaled positions
            t = torch.arange(effective_seq_len, device=x.device, dtype=self.inv_freq.dtype)
            t = t * self.scaling_factor
            
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)


class DynamicNTKScalingRoPE(RotaryEmbedding):
    """
    Dynamic NTK Scaling RoPE
    Automatically adjusts scaling based on sequence length
    Enables training at multiple context sizes
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_type: str = "dynamic",  # dynamic, linear, yarn
        device: Optional[torch.device] = None,
    ):
        super().__init__(dim, max_position_embeddings, base, device)
        self.scaling_type = scaling_type
        self.original_max_position_embeddings = max_position_embeddings
        
        logger.info(f"Dynamic NTK RoPE: type={scaling_type}, max_pos={max_position_embeddings}")
    
    def _compute_dynamic_scale(self, seq_len: int) -> float:
        """Compute dynamic scaling factor based on sequence length"""
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
        
        if self.scaling_type == "linear":
            # Linear scaling
            return seq_len / self.original_max_position_embeddings
        
        elif self.scaling_type == "dynamic":
            # Dynamic NTK scaling
            scale = seq_len / self.original_max_position_embeddings
            alpha = scale ** (self.dim / (self.dim - 2))
            return alpha
        
        elif self.scaling_type == "yarn":
            # YaRN scaling (Yet another RoPE extensioN)
            scale = seq_len / self.original_max_position_embeddings
            # Interpolate low frequencies, extrapolate high frequencies
            return scale
        
        return 1.0
    
    def _update_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
        """Update cache with dynamic scaling"""
        scale = self._compute_dynamic_scale(seq_len)
        
        if seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            
            # Compute scaled inverse frequencies
            if scale > 1.0:
                scaled_inv_freq = self.inv_freq / scale
            else:
                scaled_inv_freq = self.inv_freq
            
            t = torch.arange(seq_len, device=x.device, dtype=scaled_inv_freq.dtype)
            freqs = torch.outer(t, scaled_inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)


class ContinuousRoPE(nn.Module):
    """
    Continuous RoPE with learnable interpolation
    Smoothly adapts to any sequence length
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        learnable_interpolation: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.learnable_interpolation = learnable_interpolation
        
        # Base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Learnable interpolation weights
        if learnable_interpolation:
            self.interpolation_scale = nn.Parameter(torch.ones(1))
            self.interpolation_offset = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("interpolation_scale", torch.ones(1))
            self.register_buffer("interpolation_offset", torch.zeros(1))
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with continuous interpolation
        
        Args:
            x: Input tensor
            seq_len: Sequence length
        
        Returns:
            cos, sin: Continuous position embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Continuous position indices
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # Apply learnable interpolation
        t = t * self.interpolation_scale + self.interpolation_offset
        
        # Normalize to [0, max_position_embeddings]
        if seq_len > self.max_position_embeddings:
            t = t * (self.max_position_embeddings / seq_len)
        
        # Compute frequencies
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key
    
    Args:
        q: Query tensor [batch, seq, heads, dim]
        k: Key tensor [batch, seq, heads, dim]
        cos: Cosine embeddings [seq, dim]
        sin: Sine embeddings [seq, dim]
    
    Returns:
        q_embed, k_embed: Rotated query and key
    """
    # Expand cos/sin to match q/k shape
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def create_position_encoding(
    encoding_type: str,
    dim: int,
    max_position_embeddings: int = 2048,
    base: float = 10000.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create position encodings
    
    Args:
        encoding_type: standard, ntk_scaled, dynamic_continuous
        dim: Embedding dimension
        max_position_embeddings: Maximum sequence length
        base: Base frequency
        **kwargs: Additional arguments
    
    Returns:
        Position encoding module
    """
    if encoding_type == "standard":
        return RotaryEmbedding(dim, max_position_embeddings, base)
    
    elif encoding_type == "ntk_scaled":
        ntk_alpha = kwargs.get("ntk_alpha", 1.0)
        scaling_factor = kwargs.get("scaling_factor", 1.0)
        return NTKScaledRotaryEmbedding(dim, max_position_embeddings, base, scaling_factor, ntk_alpha)
    
    elif encoding_type == "dynamic_continuous":
        scaling_type = kwargs.get("scaling_type", "dynamic")
        return DynamicNTKScalingRoPE(dim, max_position_embeddings, base, scaling_type)
    
    elif encoding_type == "continuous":
        learnable = kwargs.get("learnable_interpolation", True)
        return ContinuousRoPE(dim, max_position_embeddings, base, learnable)
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
