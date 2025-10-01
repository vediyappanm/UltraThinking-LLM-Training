import torch
from typing import Iterable


def build_optimizer(model: torch.nn.Module, args) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with common defaults.
    If a fused optimizer is available and requested later, hook it up here.
    """
    kwargs = dict(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
    )
    # Use fused AdamW if available and requested
    use_fused = bool(getattr(args, "fused_adam", False)) and torch.cuda.is_available()
    try:
        if use_fused:
            # PyTorch AdamW may accept fused=True on recent versions
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        # Fallback to non-fused if unsupported
        pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def clip_grads(parameters: Iterable[torch.nn.Parameter], scaler, max_norm: float) -> None:
    """Gradient clipping that is AMP-aware (unscales first if using a scaler)."""
    if max_norm is None or max_norm <= 0:
        return
    if scaler is not None:
        scaler.unscale_(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)
