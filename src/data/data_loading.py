"""
Advanced Data Loading and Preprocessing for Large Scale Training
Supports streaming, quality filtering, and efficient tokenization
"""

import os
import json
import random
import hashlib
import re
from typing import Dict, List, Optional, Union, Iterator, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np

# Optional libs
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    import datasets
    from datasets import load_dataset, Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = "data/train.txt"  # file or directory, or HF dataset name when use_hf=True
    tokenizer_type: str = "tiktoken"       # tiktoken | sentencepiece | char
    tokenizer_name: str = "gpt2"          # tiktoken encoding name
    sp_model_path: Optional[str] = None    # sentencepiece .model path if tokenizer_type=sentencepiece
    max_length: int = 2048
    streaming: bool = True
    num_workers: int = 4
    validation_split: float = 0.05
    pack_sequences: bool = True
    shuffle_buffer_size: int = 10000
    preprocessing_num_workers: int = 8
    quality_filtering: bool = False
    deduplication: bool = False
    min_length: int = 10
    max_length_filter: int = 100000
    language_filter: Optional[str] = None  # simple ASCII/latin filter if set to 'en'
    seed: int = 42
    batch_size: int = 4
    use_hf: bool = False                   # set true to load a HF dataset by name in dataset_path
    hf_text_column: Optional[str] = None   # text column when using HF datasets


# -----------------------------------------------------------------------------
# Tokenizer
# -----------------------------------------------------------------------------

class AdvancedTokenizer:
    """Advanced tokenizer supporting multiple backends.
    Provides encode() and decode(), and exposes vocab_size, eos_token_id, pad_token_id.
    """

    def __init__(self, tokenizer_type: str = "tiktoken", tokenizer_name: str = "gpt2", sp_model_path: Optional[str] = None, vocab_size: Optional[int] = None):
        self.tokenizer_type = tokenizer_type
        self.tokenizer_name = tokenizer_name
        self.pad_token_id = None
        self.eos_token_id = None

        if tokenizer_type == "tiktoken" and TIKTOKEN_AVAILABLE:
            enc = tiktoken.get_encoding(tokenizer_name)
            self.tokenizer = enc
            self.vocab_size = enc.n_vocab
            # there is no canonical eos in raw encoders; use newline as pseudo eos
            self.eos_token_id = enc.encode("\n")[0]
        elif tokenizer_type == "sentencepiece" and SENTENCEPIECE_AVAILABLE and sp_model_path and os.path.exists(sp_model_path):
            self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
            self.tokenizer = None
            self.vocab_size = self.sp.get_piece_size()
            # Try to infer eos/pad
            self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else None
            self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else None
        else:
            # Simple character-level fallback (byte-level limited to 256)
            self.tokenizer = None
            self.sp = None
            self.vocab_size = vocab_size or 256
            self.eos_token_id = ord("\n") if self.vocab_size > ord("\n") else None

    def encode(self, text: str) -> List[int]:
        if self.tokenizer is not None and self.tokenizer_type == "tiktoken":
            return self.tokenizer.encode(text)
        if self.sp is not None:
            return self.sp.encode(text, out_type=int)
        # char/byte fallback
        return [ord(c) for c in text if 0 <= ord(c) < self.vocab_size]

    def decode(self, tokens: List[int]) -> str:
        if self.tokenizer is not None and self.tokenizer_type == "tiktoken":
            return self.tokenizer.decode(tokens)
        if self.sp is not None:
            return self.sp.decode(tokens)
        # char/byte fallback
        return ''.join(chr(t) for t in tokens if 0 <= t < 256)


# -----------------------------------------------------------------------------
# Quality filtering and utilities
# -----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_ASCII_RE = re.compile(r"^[\x09\x0A\x0D\x20-\x7E]+$")


def normalize_text(s: str) -> str:
    s = s.replace("\u200b", " ").replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s


def passes_language_filter(s: str, lang: Optional[str]) -> bool:
    if not lang:
        return True
    # very simple English filter: largely ASCII printable range
    if lang.lower() == 'en':
        return bool(_ASCII_RE.match(s))
    return True


def length_ok(s: str, min_len: int, max_len: int) -> bool:
    n = len(s)
    return (n >= min_len) and (n <= max_len)


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Streaming dataset that reads raw text lines, tokenizes, and yields fixed-length causal LM examples.

    Yields dicts with keys: 'input_ids', 'labels'
    """

    def __init__(
        self,
        sources: Union[str, List[str]],
        tokenizer: AdvancedTokenizer,
        max_length: int,
        shuffle_buffer_size: int = 10000,
        quality_filtering: bool = False,
        language_filter: Optional[str] = None,
        min_length: int = 0,
        max_length_filter: int = 10_000_000,
        deduplicate: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.sources = sources if isinstance(sources, list) else [sources]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.quality_filtering = quality_filtering
        self.language_filter = language_filter
        self.min_length = min_length
        self.max_length_filter = max_length_filter
        self.deduplicate = deduplicate
        self.seed = seed

    def _iter_lines(self) -> Iterator[str]:
        random.seed(self.seed)
        for src in self.sources:
            p = Path(src)
            if p.is_file():
                files = [p]
            elif p.is_dir():
                files = [q for q in p.rglob("*.txt")]
            else:
                continue
            # deterministic order but can shuffle within file later
            for f in files:
                try:
                    with f.open('r', encoding='utf-8', errors='ignore') as fh:
                        for line in fh:
                            yield line.rstrip("\n")
                except Exception:
                    continue

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[Dict[str, torch.Tensor]] = []
        seen_hashes = set()
        token_buf: List[int] = []

        for raw in self._iter_lines():
            text = normalize_text(raw)
            if self.quality_filtering:
                if not length_ok(text, self.min_length, self.max_length_filter):
                    continue
                if not passes_language_filter(text, self.language_filter):
                    continue
            if self.deduplicate:
                h = hashlib.md5(text.encode('utf-8')).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

            ids = self.tokenizer.encode(text)
            if not ids:
                continue
            # append eos if available
            if self.tokenizer.eos_token_id is not None:
                ids.append(self.tokenizer.eos_token_id)

            token_buf.extend(ids)

            # emit fixed-length chunks
            while len(token_buf) >= self.max_length + 1:
                seq = token_buf[: self.max_length + 1]
                item = {
                    'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
                    'labels': torch.tensor(seq[1:], dtype=torch.long),
                }
                buffer.append(item)
                token_buf = token_buf[self.max_length:]

                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    for it in buffer:
                        yield it
                    buffer.clear()

        if len(token_buf) > 1:
            seq = token_buf[: self.max_length + 1]
            if len(seq) > 1:
                yield {
                    'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
                    'labels': torch.tensor(seq[1:], dtype=torch.long),
                }
        # flush buffer
        if buffer:
            random.shuffle(buffer)
            for it in buffer:
                yield it


class PackedTextDataset(Dataset):
    """Pack a list of token ids into fixed-length training examples."""

    def __init__(self, token_ids: List[int], max_length: int):
        self.token_ids = token_ids
        self.max_length = max_length

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.max_length)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.token_ids[idx: idx + self.max_length + 1]
        return {
            'input_ids': torch.tensor(chunk[:-1], dtype=torch.long),
            'labels': torch.tensor(chunk[1:], dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# High level loader
# -----------------------------------------------------------------------------

def _load_all_text_from_local(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding='utf-8', errors='ignore')
    if p.is_dir():
        texts = []
        for f in p.rglob('*.txt'):
            try:
                texts.append(f.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                continue
        return "\n".join(texts)
    return ""


def _tokenize_in_threads(texts: List[str], tokenizer: AdvancedTokenizer, workers: int) -> List[int]:
    # tokenize many chunks and concatenate
    def _enc(t: str) -> List[int]:
        ids = tokenizer.encode(t)
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        return ids
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        parts = list(ex.map(_enc, texts))
    flat: List[int] = []
    for p in parts:
        flat.extend(p)
    return flat


def create_dataloaders(config: DataConfig, tokenizer: Optional[AdvancedTokenizer] = None) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Create train/val dataloaders based on the provided config.

    Returns (train_loader, val_loader, info_dict)
    """
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Build tokenizer if not provided
    tok = tokenizer or AdvancedTokenizer(
        tokenizer_type=config.tokenizer_type,
        tokenizer_name=config.tokenizer_name,
        sp_model_path=config.sp_model_path,
    )

    # Streaming path
    if config.streaming:
        if config.use_hf and HF_DATASETS_AVAILABLE:
            # HF streaming dataset (text column required)
            hf = load_dataset(config.dataset_path, split='train', streaming=True)
            text_col = config.hf_text_column
            assert text_col is not None, "hf_text_column must be provided when use_hf=True"
            # write to temp file-like stream by iterating; then reuse StreamingTextDataset over a buffer of lines
            # Simpler: build an IterableDataset that wraps HF streaming
            class HFStream(IterableDataset):
                def __iter__(self_inner):
                    buffer: List[Dict[str, torch.Tensor]] = []
                    token_buf: List[int] = []
                    for ex in hf:
                        raw = ex[text_col]
                        text = normalize_text(raw)
                        if config.quality_filtering:
                            if not length_ok(text, config.min_length, config.max_length_filter):
                                continue
                            if not passes_language_filter(text, config.language_filter):
                                continue
                        ids = tok.encode(text)
                        if not ids:
                            continue
                        if tok.eos_token_id is not None:
                            ids.append(tok.eos_token_id)
                        token_buf.extend(ids)
                        while len(token_buf) >= config.max_length + 1:
                            seq = token_buf[: config.max_length + 1]
                            item = {
                                'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
                                'labels': torch.tensor(seq[1:], dtype=torch.long),
                            }
                            buffer.append(item)
                            token_buf = token_buf[config.max_length:]
                            if len(buffer) >= config.shuffle_buffer_size:
                                random.shuffle(buffer)
                                for it in buffer:
                                    yield it
                                buffer.clear()
                    if len(token_buf) > 1:
                        seq = token_buf[: config.max_length + 1]
                        if len(seq) > 1:
                            yield {
                                'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
                                'labels': torch.tensor(seq[1:], dtype=torch.long),
                            }
                    if buffer:
                        random.shuffle(buffer)
                        for it in buffer:
                            yield it
            train_iter = HFStream()
            # For simplicity we don't split HF streaming; users can pass validation separately.
            train_loader = DataLoader(train_iter, batch_size=config.batch_size, num_workers=0)
            val_loader = DataLoader([], batch_size=config.batch_size)
            return train_loader, val_loader, {"vocab_size": tok.vocab_size, "tokenizer_type": tok.tokenizer_type}
        else:
            stream_ds = StreamingTextDataset(
                sources=config.dataset_path,
                tokenizer=tok,
                max_length=config.max_length,
                shuffle_buffer_size=config.shuffle_buffer_size,
                quality_filtering=config.quality_filtering,
                language_filter=config.language_filter,
                min_length=config.min_length,
                max_length_filter=config.max_length_filter,
                deduplicate=config.deduplication,
                seed=config.seed,
            )
            # Note: IterableDataset shouldn't use num_workers>0 unless guaranteed safe; keep 0 here
            train_loader = DataLoader(stream_ds, batch_size=config.batch_size, num_workers=0)
            # No natural split for streaming; provide empty val loader
            val_loader = DataLoader([], batch_size=config.batch_size)
            return train_loader, val_loader, {"vocab_size": tok.vocab_size, "tokenizer_type": tok.tokenizer_type}

    # Non-streaming: read all, tokenize, split
    if config.use_hf and HF_DATASETS_AVAILABLE:
        ds = load_dataset(config.dataset_path, split='train')
        text_col = config.hf_text_column
        assert text_col is not None, "hf_text_column must be provided when use_hf=True"
        texts = [normalize_text(r[text_col]) for r in ds]
    else:
        raw_text = _load_all_text_from_local(config.dataset_path)
        if not raw_text:
            # create tiny dummy data
            raw_text = "This is a sample text for training. " * 1000
        # simple paragraph split to enable parallel tokenization
        texts = [t for t in re.split(r"\n\n+", raw_text) if t.strip()]

    if config.quality_filtering:
        texts = [t for t in texts if length_ok(t, config.min_length, config.max_length_filter) and passes_language_filter(t, config.language_filter)]

    if config.deduplication:
        seen = set()
        deduped = []
        for t in texts:
            h = hashlib.md5(t.encode('utf-8')).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            deduped.append(t)
        texts = deduped

    token_ids = _tokenize_in_threads(texts, tok, workers=config.preprocessing_num_workers)

    # Split tokens into train/val by proportion
    split_idx = int(len(token_ids) * (1 - config.validation_split))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_ds = PackedTextDataset(train_tokens, config.max_length)
    val_ds = PackedTextDataset(val_tokens, config.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(0, config.num_workers // 2),
        pin_memory=True,
    )

    info = {
        "vocab_size": tok.vocab_size,
        "tokenizer_type": tok.tokenizer_type,
        "num_train_tokens": len(train_tokens),
        "num_val_tokens": len(val_tokens),
        "num_train_examples": len(train_ds),
        "num_val_examples": len(val_ds),
    }
    return train_loader, val_loader, info
