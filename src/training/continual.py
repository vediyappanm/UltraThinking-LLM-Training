"""
Continual learning utilities: EWC and Replay
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EWCConfig:
    lambda_ewc: float = 0.4
    fisher_samples: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EWC:
    """Elastic Weight Consolidation for continual learning.

    Usage:
        ewc = EWC(model)
        ewc.compute_fisher(dataloader)
        # during training on new task compute loss + ewc.penalty(model)
    """
    def __init__(self, model: nn.Module, config: Optional[EWCConfig] = None):
        self.model = model
        self.config = config or EWCConfig()
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p, device=p.device) for n, p in self.params.items()}

    @torch.no_grad()
    def compute_fisher(self, dataloader) -> None:
        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= self.config.fisher_samples:
                break
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            grads = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], retain_graph=False, allow_unused=True)
            i = 0
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                g = grads[i]
                i += 1
                if g is None:
                    continue
                self.fisher[n] += g.detach() ** 2
            count += 1
        # Average
        for n in self.fisher:
            self.fisher[n] /= max(count, 1)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.config.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.config.lambda_ewc * loss


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict[str, torch.Tensor]] = []

    def add(self, example: Dict[str, torch.Tensor]):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append({k: v.detach().cpu() for k, v in example.items()})

    def sample(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


class ContinualTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, ewc: Optional[EWC] = None, replay: Optional[ReplayBuffer] = None):
        self.model = model
        self.optimizer = optimizer
        self.ewc = ewc
        self.replay = replay

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = {k: v.to(next(self.model.parameters()).device) for k, v in batch.items()}
        outputs = self.model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if self.ewc is not None:
            loss = loss + self.ewc.penalty(self.model)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.replay is not None:
            self.replay.add(batch)
        return {"loss": float(loss.detach().cpu().item())}
