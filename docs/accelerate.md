# Accelerate (DDP) Training

`accelerate` provides a simple launcher for multi-GPU training.

## Setup
```bash
pip install accelerate
accelerate config  # follow prompts, set mixed precision if desired
```

## Launch
```bash
accelerate launch train_ultrathink.py \
  --distributed \
  --dataset dummy --train_samples 4000 --val_samples 400 \
  --vocab_size 50257 --hidden_size 384 --num_layers 4 --num_heads 6 --num_kv_heads 6 \
  --intermediate_size 1536 --max_seq_length 512 \
  --batch_size 1 --gradient_accumulation_steps 64 \
  --learning_rate 5e-5 --use_amp --gradient_checkpointing
```

Tip: you can also keep `--distributed` off and let Accelerate manage process groups based on your config. Ensure dataloaders and samplers are set up accordingly.
