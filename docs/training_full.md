# Full Training Pipeline

Covers distributed data parallel (DDP), 4D parallelism, and RLHF.

## DDP
```bash
torchrun --nproc_per_node=8 train_ultrathink.py \
  --distributed \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --use_amp --gradient_checkpointing
```

## 4D Parallelism (if supported)
```bash
torchrun --nproc_per_node=16 train_ultrathink.py \
  --distributed --use_4d_parallelism \
  --data_parallel_size 2 --tensor_parallel_size 2 \
  --pipeline_parallel_size 2 --expert_parallel_size 2 \
  --zero_stage 3
```

## RLHF 2.0 (placeholder)
```bash
python train_ultrathink.py \
  --enable_rlhf \
  --rlhf_iterations 100 \
  --rlhf_steps_per_iteration 1000 \
  --ppo_epochs 4 --ppo_batch_size 32
```

Notes:
- Some RLHF datasets/pipelines are placeholders and need real environments and reward models.
