from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


def orpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    # Odds Ratio Preference Optimization (no reference model)
    # L = - E[ log sigma( beta * (s_chosen - s_rejected) ) ]
    diff = policy_chosen_logps - policy_rejected_logps
    return -F.logsigmoid(beta * diff).mean()


def cpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # Contrastive Preference Optimization (InfoNCE-style)
    # L = - E[ log( exp(s_c / T) / (exp(s_c / T) + exp(s_r / T)) ) ]
    s_c = policy_chosen_logps / temperature
    s_r = policy_rejected_logps / temperature
    logits = torch.stack([s_c, s_r], dim=-1)
    targets = torch.zeros(logits.shape[:-1], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def compute_pair_logps(model, chosen_inputs: Dict[str, torch.Tensor], rejected_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        lp_chosen = model(**chosen_inputs)["logits"].log_softmax(-1).mean()
        lp_rejected = model(**rejected_inputs)["logits"].log_softmax(-1).mean()
    return {"chosen": lp_chosen, "rejected": lp_rejected}
