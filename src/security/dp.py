"""
Differential Privacy training stubs; uses Opacus if available.
"""
from __future__ import annotations

from typing import Optional
import warnings

import torch


def make_dp_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader,
    target_epsilon: float = 5.0,
    max_grad_norm: float = 1.0,
    epochs: int = 1,
    delta: float = 1e-5,
):
    try:
        from opacus import PrivacyEngine  # type: ignore
    except Exception:
        warnings.warn("Opacus not installed; returning original optimizer without DP")
        return model, optimizer, data_loader, None

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )
    return model, optimizer, data_loader, privacy_engine
