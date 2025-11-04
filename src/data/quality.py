from __future__ import annotations

from typing import Dict, Any
import torch


def score_text_quality(model, tokenizer, text: str, max_len: int = 512) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    # Lower loss -> higher quality
    loss = out.loss.detach().float().item()
    return float(1.0 / (1.0 + loss))
