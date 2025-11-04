from __future__ import annotations

from typing import Iterator, Dict, Any
from datasets import load_dataset


def stream_text(dataset_name: str, subset: str = None, split: str = "train") -> Iterator[Dict[str, Any]]:
    ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    for ex in ds:
        if ex.get("text"):
            yield ex
