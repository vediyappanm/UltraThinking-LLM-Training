# FAQ

- Q: I get non-finite losses on step 0-10.
  - A: Lower LR, use AMP warmup, keep seq length 512, and ensure attention mask is causal+key-side only.

- Q: How do I resume training?
  - A: Use `--resume_checkpoint <path_to_checkpoint.pt>`.

- Q: Do I need FlashAttention?
  - A: Optional. The code falls back to PyTorch SDPA which is stable and memory-efficient.
