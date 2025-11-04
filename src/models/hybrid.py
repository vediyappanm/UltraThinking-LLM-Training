"""
Hybrid Mamba-Attention model with sliding-window attention fallback.

- Interleaves Mamba blocks and Attention blocks
- Useful for long-context and streaming inputs
- Compatible with existing AdvancedGPTModel components
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaBlock, MambaConfig, RMSNorm
from .architecture import GroupedQueryAttention, ModelConfig


@dataclass
class HybridConfig:
    d_model: int
    n_layers: int
    vocab_size: int
    pattern: List[str]  # e.g., ["mamba", "attn", "mamba", "attn", ...]
    rotary_dim: int = 128
    dropout: float = 0.05


class HybridLayer(nn.Module):
    def __init__(self, d_model: int, layer_type: str, model_cfg: ModelConfig, dropout: float = 0.05):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(d_model)
        self.resid_drop = nn.Dropout(dropout)
        if layer_type == "mamba":
            self.block = MambaBlock(MambaConfig(d_model=d_model, dropout=dropout))
        else:
            # Attention layer using existing GQA implementation
            attn_cfg = ModelConfig(**{**model_cfg.__dict__})
            attn_cfg.n_embd = d_model
            self.attn = GroupedQueryAttention(attn_cfg)
            # Simple FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.layer_type == "mamba":
            return self.block(x)
        # attention path
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, attention_mask=attention_mask, use_cache=False, past_key_value=None)
        x = residual + self.resid_drop(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.resid_drop(self.ffn(x))
        return x


class HybridModel(nn.Module):
    def __init__(self, model_cfg: ModelConfig, n_layers: int, vocab_size: int, pattern: Optional[List[str]] = None, dropout: float = 0.05):
        super().__init__()
        d = model_cfg.n_embd
        self.embed = nn.Embedding(vocab_size, d)
        self.drop = nn.Dropout(dropout)
        if pattern is None:
            pattern = ["mamba" if i % 2 == 0 else "attn" for i in range(n_layers)]
        self.layers = nn.ModuleList([
            HybridLayer(d, layer_type=pattern[i % len(pattern)], model_cfg=model_cfg, dropout=dropout)
            for i in range(n_layers)
        ])
        self.norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        x = self.embed(input_ids)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "hidden_states": x}
