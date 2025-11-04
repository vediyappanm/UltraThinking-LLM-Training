"""
Speculative decoding utilities (teacher + draft model).
"""
from __future__ import annotations

from typing import Dict, Any
import torch


def speculative_generate(
    teacher_model,
    draft_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    draft_steps: int = 4,
) -> str:
    device = next(teacher_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    teacher_model.eval(); draft_model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Draft proposes K tokens autoregressively
            proposal = input_ids.clone()
            for _ in range(draft_steps):
                logits = draft_model(proposal)["logits"][:, -1, :]
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                proposal = torch.cat([proposal, next_id], dim=1)

            # Teacher validates one step from proposal
            t_logits = teacher_model(proposal)["logits"]
            # Accept the first proposed token according to teacher
            accept_id = torch.argmax(t_logits[:, -draft_steps - 1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, accept_id], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
