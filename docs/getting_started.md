# Getting Started

## Install

```bash
git clone <repository-url>
cd deep
pip install -r requirements.txt
```

Optional (recommended on supported GPUs):
```bash
pip install flash-attn --no-build-isolation
```

## Quick Sanity Run (dummy data)
Use the built-in dummy dataset to verify the environment.
```bash
python train_ultrathink.py \
  --dataset dummy \
  --train_samples 2000 \
  --val_samples 200 \
  --vocab_size 50257 \
  --hidden_size 384 --num_layers 4 --num_heads 6 --num_kv_heads 6 \
  --intermediate_size 1536 --max_seq_length 256 \
  --batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --use_amp --gradient_checkpointing \
  --num_epochs 1 \
  --output_dir ./outputs/sanity_dummy
```

Expected: the training loop runs and reports finite losses.

## Troubleshooting
- If you see non-finite losses on first steps, lower LR and/or disable AMP briefly.
- Ensure `transformers`, `datasets`, and `torch` versions meet `requirements.txt`.
