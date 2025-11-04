"""
AutoML / NAS Hooks using Optuna
"""
from __future__ import annotations

from typing import Dict, Any, Optional

import optuna
import torch

from src.models.ultrathink import UltraThinkModel, UltraThinkConfig
from src.models.architecture import ModelConfig


def objective(trial: optuna.Trial, dataset, eval_fn) -> float:
    # Search over architecture & hparams
    arch = trial.suggest_categorical("architecture", ["transformer", "mamba", "hybrid"])
    n_layer = trial.suggest_int("n_layer", 12, 48, step=12)
    n_embd = trial.suggest_categorical("n_embd", [1024, 2048, 4096])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    mcfg = ModelConfig(n_embd=n_embd, n_layer=n_layer)
    cfg = UltraThinkConfig(model_config=mcfg)
    cfg.architecture = arch

    model = UltraThinkModel(cfg)

    # Train for very few steps (proxy)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    steps = 10
    for _ in range(steps):
        batch = next(iter(dataset))
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        out = model(**batch)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Evaluate
    score = eval_fn(model)
    return score


def run_search(dataset, eval_fn, n_trials: int = 20, direction: str = "maximize") -> optuna.Study:
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: objective(trial, dataset, eval_fn), n_trials=n_trials)
    return study
