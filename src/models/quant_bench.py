"""
Quantization benchmarking and dual-checkpoint utilities.
Builds on src.models.quantization to:
- Save both full-precision and quantized checkpoints
- Benchmark speed, memory, and accuracy (perplexity) on sample datasets
- Ensure compatibility with LLaMA/Mistral/Phi families via HF Auto classes
"""
from __future__ import annotations

import time
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .quantization import ModelQuantizer


def save_dual_checkpoints(
    model_name_or_path: str,
    method: str = "gptq",
    bits: int = 4,
    output_dir_full: str = "./output/full_precision",
    output_dir_quant: str = "./output/quantized",
) -> Dict[str, str]:
    """Save both full-precision and quantized checkpoints.

    Returns mapping with paths.
    """
    # Save full precision
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base_model.save_pretrained(output_dir_full)
    tokenizer.save_pretrained(output_dir_full)

    # Quantize and save
    quantizer = ModelQuantizer(model_name_or_path)
    q_model = quantizer.quantize(method=method, bits=bits, output_path=output_dir_quant)

    return {"full": output_dir_full, "quant": output_dir_quant}


def perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "validation",
    max_samples: int = 256,
    context_length: int = 512,
) -> float:
    """Compute perplexity on a small dataset subset for quick accuracy check."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split)
    texts = [x["text"] for x in ds if x["text"] and len(x["text"]) > 0][:max_samples]

    nll = 0.0
    n_tokens = 0
    device = next(model.parameters()).device
    model.eval()

    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=context_length)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss.detach().float()
        # Approximate: loss is mean over tokens in sequence
        n = input_ids.numel()
        nll += loss.item() * n
        n_tokens += n

    ppl = torch.exp(torch.tensor(nll / max(n_tokens, 1))).item()
    return float(ppl)


def benchmark_accuracy_speed_memory(
    model_name_or_path: str,
    method: str = "gptq",
    bits: int = 4,
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    split: str = "validation",
    max_samples: int = 128,
    prompt: str = "UltraThinking is",
    gen_tokens: int = 64,
) -> Dict[str, Any]:
    """Benchmark speed, memory, and accuracy before and after quantization."""
    # Load FP model
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # Accuracy (perplexity)
    fp_ppl = perplexity(fp_model, tok, dataset_name, subset, split, max_samples)

    # Speed test
    inputs = tok(prompt, return_tensors="pt").to(next(fp_model.parameters()).device)
    _ = fp_model.generate(**inputs, max_new_tokens=8)
    t0 = time.time()
    _ = fp_model.generate(**inputs, max_new_tokens=gen_tokens)
    fp_time = time.time() - t0
    fp_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    # Quantize
    quantizer = ModelQuantizer(model_name_or_path)
    q_model = quantizer.quantize(method=method, bits=bits)

    # Accuracy (perplexity) quantized
    q_ppl = perplexity(q_model, tok, dataset_name, subset, split, max_samples)

    # Speed test quantized
    inputs = tok(prompt, return_tensors="pt").to(next(q_model.parameters()).device)
    _ = q_model.generate(**inputs, max_new_tokens=8)
    t0 = time.time()
    _ = q_model.generate(**inputs, max_new_tokens=gen_tokens)
    q_time = time.time() - t0
    q_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

    return {
        "fp": {"perplexity": fp_ppl, "gen_time_s": fp_time, "memory_mb": fp_mem},
        "quant": {"perplexity": q_ppl, "gen_time_s": q_time, "memory_mb": q_mem},
        "speedup": fp_time / q_time if q_time > 0 else None,
        "ppl_delta": q_ppl - fp_ppl,
        "memory_saving_mb": max(fp_mem - q_mem, 0),
    }
