from __future__ import annotations

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def ab_test(models: List[str], prompts: List[str], max_new_tokens: int = 64) -> Dict[str, float]:
    assert len(models) == 2
    toks = [AutoTokenizer.from_pretrained(m) for m in models]
    mdls = [AutoModelForCausalLM.from_pretrained(m, device_map="auto") for m in models]
    scores = {models[0]: 0, models[1]: 0}
    for p in prompts:
        outs = []
        for tok, mdl in zip(toks, mdls):
            inputs = tok(p, return_tensors="pt").to(next(mdl.parameters()).device)
            with torch.no_grad():
                out = mdl.generate(**inputs, max_new_tokens=max_new_tokens)
            outs.append(tok.decode(out[0], skip_special_tokens=True))
        # Simple length-based proxy: prefer shorter answer (placeholder)
        winner = 0 if len(outs[0]) <= len(outs[1]) else 1
        scores[models[winner]] += 1
    # Normalize
    total = len(prompts)
    return {k: v / total for k, v in scores.items()}
