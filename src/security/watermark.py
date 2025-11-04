"""
Lightweight text watermarking utilities.
"""
from __future__ import annotations

from typing import List
import hashlib


def embed_watermark(text: str, key: str) -> str:
    # Simple scheme: append invisible zero-width spaces encoding hash bits
    h = hashlib.sha256((key + text).encode()).hexdigest()
    bits = bin(int(h, 16))[2:][:64]
    zwsp = "\u200b"  # zero-width space
    return text + " " + zwsp.join(["" if b == "0" else zwsp for b in bits])


def detect_watermark(text: str, key: str) -> float:
    # Estimate presence by matching expected pattern length
    zwsp = "\u200b"
    count = text.count(zwsp)
    return min(1.0, count / 64.0)
