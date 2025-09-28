# ULTRATHINK (Quick Start)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/colab.ipynb)

ULTRATHINK is a practical, open-source training stack for advanced LLMs with stability-first defaults and modular components.

- **Train small first**: verify learning on a tiny dataset, then scale safely.
- **Stable by default**: SDPA/FlashAttention, AMP/DRE warmups, safe attention masks.
- **Modular**: Dynamic Reasoning (DRE), Mixture-of-Experts (MoE), Multimodal, RLHF.

## Architecture (High-level)

```mermaid
flowchart TD
    A[Text / Multimodal Data] --> B[Data Pipeline\nHF Datasets/Streaming]
    B --> C[Tokenizer]
    C --> D[Backbone Transformer]
    D -->|Optional| E[DRE Router\nDynamic Reasoning]
    D -->|Optional| F[MoE Blocks]
    D -->|Optional| G[Multimodal Encoders]
    E --> H[Model Outputs]
    F --> H
    G --> H
    D --> H

    subgraph Training
        D <--> I[Optimizer + Scheduler]
        I --> J[AMP / Grad Accum / Grad Clip]
        J --> D
    end

## Quick Links
+- Docs index: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/README.md
+- Getting started: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/getting_started.md
+- Train small & monitor: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/training_small.md
+- Datasets (switching, custom, mixing): https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/datasets.md
+- Train in Google Colab: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/colab.md
+- Full training (DDP/4D/RLHF): https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/training_full.md
+- Development guide: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/development.md
+- Evaluation: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/evaluation.md
+- FAQ: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/docs/faq.md

## Small real-data run (stable defaults)
```bash
export TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --enable_dre --dre_warmup_steps 500 \
  --amp_warmup_steps 200 \
  --tokenizer_name gpt2 \
  --vocab_size 50257 \
  --hidden_size 384 --num_layers 4 --num_heads 6 --num_kv_heads 6 \
  --intermediate_size 1536 --max_seq_length 512 \
  --batch_size 1 --gradient_accumulation_steps 64 \
  --learning_rate 5e-5 --weight_decay 0.1 \
  --warmup_steps 2000 \
  --use_amp --gradient_checkpointing \
  --eval_frequency 1 \
  --output_dir ./outputs/ultrathink_c4_seq512_sdpa_warmup
```

## Project structure (short)
```
deep/
├─ train_ultrathink.py        # Training entrypoint
├─ app_gradio.py              # Demo UI (if applicable)
├─ src/
│  ├─ models/                 # Model, attention, MoE, DRE, multimodal
│  ├─ data/                   # Dataset configs & loaders
│  ├─ training/               # Distributed + RLHF utilities
│  └─ evaluation/             # Benchmark suite
└─ docs/                      # Documentation guides
```

## Contributing & License
- See CONTRIBUTING: https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/CONTRIBUTING.md
- License (MIT): https://github.com/vediyappanm/UltraThinking-LLM-Training/blob/main/deep/LICENSE

