from __future__ import annotations

from typing import Dict

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    class _T:
        @staticmethod
        def cuda():
            class _C:
                @staticmethod
                def is_available():
                    return False
            return _C
    torch = _T()  # type: ignore


def get_resource_stats() -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            stats.update(
                {
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "ram_used_gb": float((mem.total - mem.available) / (1024 ** 3)),
                    "ram_total_gb": float(mem.total / (1024 ** 3)),
                }
            )
        except Exception:
            pass
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        try:
            stats.update(
                {
                    "gpu_mem_alloc_gb": float(torch.cuda.memory_allocated() / (1024 ** 3)),
                    "gpu_mem_reserved_gb": float(torch.cuda.memory_reserved() / (1024 ** 3)),
                }
            )
        except Exception:
            pass
    return stats
