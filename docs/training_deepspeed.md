# DeepSpeed Training

## Prereqs
- Install: `pip install deepspeed`
- GPU(s) with NVLink/PCIe and recent CUDA drivers

## Quick start (ZeRO-2)
```bash
# Example config provided
deepspeed --num_gpus 2 train_ultrathink.py \
  --deepspeed deepspeed_config_zero2.json \
  --dataset dummy --train_samples 4000 --val_samples 400 \
  --vocab_size 50257 --hidden_size 384 --num_layers 4 --num_heads 6 --num_kv_heads 6 \
  --intermediate_size 1536 --max_seq_length 512 \
  --batch_size 1 --gradient_accumulation_steps 64 \
  --learning_rate 5e-5 --use_amp --gradient_checkpointing
```

## Notes
- Adjust `train_batch_size` and `gradient_accumulation_steps` in the JSON.
- For ZeRO-3, set `"stage": 3` and ensure the system has fast storage for optimizer/param state offload if used.
