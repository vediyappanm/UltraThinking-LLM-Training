import os
import torch
from typing import Optional


def save_checkpoint(checkpoint_dir: str, epoch: int, model, optimizer, scheduler, val_loss: float, config) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> Optional[int]:
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=False)
    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload.get("epoch", 0))
