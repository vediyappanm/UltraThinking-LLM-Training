"""
Checkpoint encryption utilities (AES via cryptography.fernet)
"""
from __future__ import annotations

from typing import Dict, Any
import json

try:
    from cryptography.fernet import Fernet  # type: ignore
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False


def generate_key() -> bytes:
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography not installed")
    return Fernet.generate_key()


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography not installed")
    return Fernet(key).encrypt(data)


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography not installed")
    return Fernet(key).decrypt(token)


def save_encrypted_checkpoint(state: Dict[str, Any], path: str, key: bytes) -> None:
    import torch
    raw = torch.save(state, path + ".tmp", _use_new_zipfile_serialization=True)
    with open(path + ".tmp", "rb") as f:
        data = f.read()
    enc = encrypt_bytes(data, key)
    with open(path, "wb") as f:
        f.write(enc)


def load_encrypted_checkpoint(path: str, key: bytes) -> Dict[str, Any]:
    import torch
    data = decrypt_bytes(open(path, "rb").read(), key)
    tmp = path + ".dec"
    with open(tmp, "wb") as f:
        f.write(data)
    state = torch.load(tmp, map_location="cpu")
    return state
