from __future__ import annotations

from typing import List, Dict, Any


def filter_by_length(records: List[Dict[str, Any]], min_len: int = 16, max_len: int = 4096, field: str = "text") -> List[Dict[str, Any]]:
    out = []
    for r in records:
        t = r.get(field, "")
        if isinstance(t, str) and min_len <= len(t) <= max_len:
            out.append(r)
    return out
