#!/usr/bin/env python
"""
FastAPI server: inference, merge, quantize endpoints.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from src.models.merge import merge_checkpoints
from src.models.quant_bench import benchmark_accuracy_speed_memory

app = FastAPI()


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    max_new_tokens: int = 128


@app.post("/generate")
def generate(req: GenerateRequest):
    tok = AutoTokenizer.from_pretrained(req.model)
    model = AutoModelForCausalLM.from_pretrained(req.model, device_map="auto")
    inputs = tok(req.prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return {"text": txt}


class MergeRequest(BaseModel):
    checkpoints: List[str]
    method: str = "slerp"
    out: str = "merged.pt"


@app.post("/merge")
def merge(req: MergeRequest):
    merge_checkpoints(req.checkpoints, method=req.method, output_path=req.out)
    return {"status": "ok", "output": req.out}


class QuantBenchRequest(BaseModel):
    model: str
    method: str = "gptq"
    bits: int = 4


@app.post("/quant_bench")
def quant_bench(req: QuantBenchRequest):
    res = benchmark_accuracy_speed_memory(req.model, method=req.method, bits=req.bits)
    return res
