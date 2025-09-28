# Training on a Small Dataset and Monitoring Learning

This guide helps you validate learning on a tiny or medium dataset before scaling up.

## Recommended hyperparameters (stable start)
- seq length: 512
- LR: 5e-5 to 1e-4
- AMP warmup: first 200 optimizer steps (disable autocast)
- DRE warmup: first 500 optimizer steps (disable DRE)
- Grad accumulation: adjust to fit memory

## Example (C4 streaming, small model)
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

## Monitoring tips
- The printed loss is averaged; look for a gradual downward trend over thousands of steps.
- If you enable Weights & Biases: `--use_wandb --run_name <name>` to visualize curves.
- If early instability appears: lower LR, keep seq length at 512 longer, or increase warmup steps.

## Switch to a tiny local/custom dataset
```bash
python train_ultrathink.py \
  --dataset custom \
  --data_path /path/to/data.jsonl \
  --text_column text \
  --max_samples 50000 \
  --tokenizer_name gpt2 \
  --max_seq_length 512 \
  --batch_size 2 --gradient_accumulation_steps 32 \
  --learning_rate 5e-5 --use_amp --gradient_checkpointing
```

Data format for `.jsonl`:
```json
{"text": "First sample..."}
{"text": "Second sample..."}
```
