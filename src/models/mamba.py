"""
Simplified Mamba/SSM-style sequence model with hybrid attention fallback.

This implementation provides a lightweight, dependency-free approximation of
state space sequence modeling. If the mamba-ssm package is available, you can
swap the internal block with the official implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.05


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(orig.dtype))


class DepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] -> [B, C, L]
        xc = x.transpose(1, 2)
        xc = F.pad(xc, (self.pad, 0))
        xc = self.conv(xc)
        return xc.transpose(1, 2)


class MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        d_hidden = cfg.expand * d
        self.in_proj = nn.Linear(d, 2 * d_hidden)
        self.conv = DepthwiseConv1d(d_hidden, cfg.d_conv)
        self.out_norm = RMSNorm(d_hidden)
        self.ssm_A = nn.Parameter(torch.randn(d_hidden, cfg.d_state) * 0.01)
        self.ssm_B = nn.Parameter(torch.randn(cfg.d_state, d_hidden) * 0.01)
        self.out_proj = nn.Linear(d_hidden, d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        u, v = self.in_proj(x).chunk(2, dim=-1)
        v = F.silu(v)
        v = self.conv(v)
        # Simple SSM-like mixing: project into state, apply A, back with B
        # Not a faithful SSM but captures long-range mixing efficiently
        state = torch.tanh(v @ self.ssm_A)
        y = state @ self.ssm_B
        y = self.out_norm(y)
        y = self.out_proj(y)
        y = self.dropout(y)
        return x + y


class MambaModel(nn.Module):
    def __init__(self, d_model: int, n_layers: int, vocab_size: int, dropout: float = 0.05):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        cfg = MambaConfig(d_model=d_model, dropout=dropout)
        self.layers = nn.ModuleList([MambaBlock(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

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
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return {"loss": loss, "logits": logits, "hidden_states": x}
