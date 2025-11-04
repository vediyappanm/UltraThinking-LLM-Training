"""
Export utilities: ONNX, TensorRT, GGUF (via existing), and vLLM integration stubs.
"""
from __future__ import annotations

from typing import Optional
import warnings

import torch

from src.models.quantization import ModelQuantizer


def export_onnx(model, tokenizer, output_path: str = "model.onnx", opset: int = 17):
    try:
        import onnx  # type: ignore
    except Exception:
        warnings.warn("onnx not installed; skipping export")
        return False
    model.eval()
    dummy = tokenizer("Hello", return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
    torch.onnx.export(model, (dummy,), output_path, opset_version=opset, input_names=["input_ids"], output_names=["logits"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}})
    return True


def export_tensorrt(model, output_path: str = "engine.plan"):
    warnings.warn("TensorRT export stub; integrate TensorRT-LLM for full support")
    return False


def integrate_vllm(model_path: str):
    warnings.warn("vLLM integration stub; use vllm.LLM(model=model_path)")
    return None
