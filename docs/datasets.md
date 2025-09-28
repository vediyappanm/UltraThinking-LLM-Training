# Datasets: Switching, Custom Data, Mixing, Streaming

This guide explains how to select built-in datasets, use your own data, mix multiple datasets, and stream large corpora.

## Built-in datasets
Use `--dataset <name>` and (optionally) `--dataset_subset`.

Supported names (see `src/data/datasets.py:DATASET_CONFIGS`):
- `wikitext` (default subset: `wikitext-2-raw-v1`)
- `wikitext-103` (streaming)
- `openwebtext` (Skylion007 mirror, streaming)
- `slim-pajama` (streaming)
- `pile`, `pile-unc` (streaming)
- `c4` (subset: `en`, streaming)
- `bookcorpus` (open variant, streaming)
- `oscar` (subset: `unshuffled_deduplicated_en`, streaming)
- `wikipedia` (subset: `20231101.en`, streaming)
- `dummy` (small in-memory samples)

Example:
```bash
python train_ultrathink.py \
  --dataset c4 \
  --dataset_subset en \
  --streaming \
  --tokenizer_name gpt2 \
  --max_seq_length 512
```

## Custom dataset from local file(s)
Use `--dataset custom --data_path <path>`.

Accepted formats via datasets: JSON/JSONL, txt, parquet. For small local files, line-delimited JSON works well:

`data.jsonl`:
```json
{"text": "First sample"}
{"text": "Second sample"}
```

Command:
```bash
python train_ultrathink.py \
  --dataset custom \
  --data_path /path/to/data.jsonl \
  --text_column text \
  --tokenizer_name gpt2 \
  --max_seq_length 512
```

Notes:
- Set `--max_samples` to cap examples when iterating quickly.
- For remote URLs or globs, `TextDataset` auto-selects a datasets builder and can stream.

## Mixing multiple datasets
Use `--mix_datasets` to blend datasets with weights. This overrides `--dataset`.

Example (50/50 wikitext/openwebtext):
```bash
python train_ultrathink.py \
  --mix_datasets "wikitext:0.5,openwebtext:0.5" \
  --tokenizer_name gpt2 \
  --max_seq_length 512 \
  --streaming
```

Implementation references:
- Mixing logic: `src/data/datasets.py:MixedDataset`
- Parsing and creation: `train_ultrathink.py:UltraThinkTrainer.load_datasets`

## Streaming vs local loading
- `--streaming` uses HF datasets streaming: low memory footprint, good for very large corpora.
- Non-streaming loads examples into memory (or iterates but indexes normally). Use with small datasets.

## Tokenization parameters
- `--tokenizer_name` (default: gpt2)
- `--max_seq_length` controls `padding='max_length'` and truncation.
- Labels are masked on padding positions to avoid loss on padded tokens.

## Troubleshooting
- If you see non-finite loss at start:
  - Lower LR (e.g., `5e-5`), keep `--max_seq_length 512` initially.
  - Use `--amp_warmup_steps 200` and `--dre_warmup_steps 500`.
- C4 deprecation warning: prefer `--dataset c4 --dataset_subset en` which maps to `allenai/c4` under the hood in some environments.
