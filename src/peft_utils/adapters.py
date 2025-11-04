from __future__ import annotations

from typing import Optional

import torch

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def apply_lora(model: torch.nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: Optional[list] = None):
    if not PEFT_AVAILABLE:
        raise ImportError("peft not installed")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules or ["q_proj","v_proj","k_proj","o_proj"]) 
    return get_peft_model(model, cfg)


def apply_qlora(model: torch.nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: Optional[list] = None):
    if not PEFT_AVAILABLE:
        raise ImportError("peft not installed")
    try:
        import bitsandbytes as bnb  # type: ignore
    except Exception:
        raise ImportError("bitsandbytes not installed for QLoRA")
    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules or ["q_proj","v_proj","k_proj","o_proj"]) 
    return get_peft_model(model, cfg)


def apply_dora(model: torch.nn.Module, *args, **kwargs):
    # DoRA (weight decomposition) not directly in peft stable; fallback to standard LoRA
    return apply_lora(model, *args, **kwargs)


def apply_adalora(model: torch.nn.Module, *args, **kwargs):
    # AdaLoRA dynamic rank scheduling would require extra scheduler; fallback to LoRA config for now
    return apply_lora(model, *args, **kwargs)
