"""
Model merging utilities: SLERP, TIES, DARE

- SLERP: Spherical linear interpolation in parameter space
- TIES: Trim, Elect, and Merge (task-integrated ensembling)
- DARE: Drop-And-REscale merging

These methods operate on PyTorch state_dicts and handle common-module
alignment issues (e.g., missing keys) robustly.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
import torch


def _align_state_dicts(state_dicts: List[Dict[str, torch.Tensor]]) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
    """Align keys across multiple state dicts by intersecting keys.
    Returns aligned dicts and the list of shared keys.
    """
    if not state_dicts:
        raise ValueError("No state dicts provided")

    shared_keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        shared_keys &= set(sd.keys())

    shared_keys = sorted(list(shared_keys))
    aligned = [{k: sd[k] for k in shared_keys} for sd in state_dicts]
    return aligned, shared_keys


def slerp_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Spherical linear interpolation across N models.

    Args:
        state_dicts: list of state dicts with identical keys
        weights: optional convex weights; defaults to uniform
    """
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6):
        total = sum(weights)
        weights = [w / total for w in weights]

    aligned, keys = _align_state_dicts(state_dicts)
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        vecs = [sd[k].float() for sd in aligned]
        # Normalize vectors
        norms = [v.norm().clamp_min(eps) for v in vecs]
        unit = [v / n for v, n in zip(vecs, norms)]
        # Weighted sum on unit sphere, then renormalize with avg norm
        merged_dir = sum(w * u for w, u in zip(weights, unit))
        merged_dir = merged_dir / merged_dir.norm().clamp_min(eps)
        avg_norm = sum(w * n for w, n in zip(weights, norms))
        merged = merged_dir * avg_norm
        out[k] = merged.to(aligned[0][k].dtype)

    return out


def ties_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    top_fraction: float = 0.2,
    tie_strategy: str = "mean",
) -> Dict[str, torch.Tensor]:
    """TIES: Keep only top-magnitude updates per tensor and average.

    Args:
        top_fraction: fraction of elements (by magnitude) to keep from union of strong entries
        tie_strategy: how to combine selected entries: 'mean' or 'median'
    """
    assert 0 < top_fraction <= 1.0
    aligned, keys = _align_state_dicts(state_dicts)
    out: Dict[str, torch.Tensor] = {}

    for k in keys:
        tensors = [sd[k].float() for sd in aligned]
        stacked = torch.stack(tensors, dim=0)  # [N, ...]
        # Compute magnitude score across models
        score = stacked.abs().max(dim=0).values
        k_keep = max(1, int(score.numel() * top_fraction))
        thresh = torch.topk(score.flatten(), k_keep).values.min()
        mask = score >= thresh
        # Combine only on masked entries
        if tie_strategy == "median":
            merged = stacked.median(dim=0).values
        else:
            merged = stacked.mean(dim=0)
        merged = torch.where(mask, merged, torch.zeros_like(merged))
        out[k] = merged.to(aligned[0][k].dtype)

    return out


def dare_merge(
    state_dicts: List[Dict[str, torch.Tensor]],
    drop_fraction: float = 0.2,
    rescale: bool = True,
) -> Dict[str, torch.Tensor]:
    """DARE: Drop low-magnitude elements and rescale.

    Args:
        drop_fraction: fraction of smallest-magnitude elements to drop (set to 0)
        rescale: if True, scale remaining entries to preserve L2 norm
    """
    assert 0.0 <= drop_fraction < 1.0
    aligned, keys = _align_state_dicts(state_dicts)
    out: Dict[str, torch.Tensor] = {}

    # Start from simple average
    base: Dict[str, torch.Tensor] = {}
    for k in keys:
        base[k] = torch.stack([sd[k].float() for sd in aligned], dim=0).mean(dim=0)

    for k in keys:
        t = base[k]
        flat = t.flatten()
        k_drop = int(flat.numel() * drop_fraction)
        if k_drop > 0:
            idx = flat.abs().argsort()[:k_drop]
            flat[idx] = 0.0
        if rescale:
            # Preserve original norm
            target_norm = base[k].norm().clamp_min(1e-8)
            new_norm = flat.norm().clamp_min(1e-8)
            flat = flat * (target_norm / new_norm)
        out[k] = flat.view_as(t).to(aligned[0][k].dtype)

    return out


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def save_state_dict(state: Dict[str, torch.Tensor], path: str) -> None:
    torch.save(state, path)


def merge_checkpoints(
    checkpoint_paths: List[str],
    method: str = "slerp",
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """High-level API to merge checkpoints.

    Args:
        checkpoint_paths: list of .pt/.bin files (state_dict format)
        method: 'slerp' | 'ties' | 'dare'
        output_path: optional path to save merged state
    """
    sds = [load_state_dict(p) for p in checkpoint_paths]
    if method == "slerp":
        merged = slerp_merge(sds, weights=kwargs.get("weights"))
    elif method == "ties":
        merged = ties_merge(sds, top_fraction=kwargs.get("top_fraction", 0.2), tie_strategy=kwargs.get("tie_strategy", "mean"))
    elif method == "dare":
        merged = dare_merge(sds, drop_fraction=kwargs.get("drop_fraction", 0.2), rescale=kwargs.get("rescale", True))
    else:
        raise ValueError(f"Unknown merge method: {method}")

    if output_path:
        save_state_dict(merged, output_path)
    return merged
