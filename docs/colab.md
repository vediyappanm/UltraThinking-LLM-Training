# Train in Google Colab

This guide shows how to train ULTRATHINK on a small setup in Colab with stable defaults.

## 1) Connect GPU runtime
- Runtime → Change runtime type → Hardware accelerator: GPU (T4/A100 preferred)

## 2) Clone repo and install deps
```bash
!git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
%cd UltraThinking-LLM-Training/deep

# Pin versions known to work well in Colab
!pip install --upgrade pip
!pip install torch --index-url https://download.pytorch.org/whl/cu121
!pip install "transformers>=4.41.0" "datasets==2.14.7" accelerate einops tqdm wandb

# Optional (FlashAttention). Skip if it fails on your GPU/driver.
# !pip install flash-attn --no-build-isolation
```

## 3) Environment tweaks for stability
```bash
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## 4) Sanity run on dummy data (quick)
```bash
!python train_ultrathink.py \
  --dataset dummy --train_samples 2000 --val_samples 200 \
  --vocab_size 50257 --hidden_size 384 --num_layers 4 --num_heads 6 --num_kv_heads 6 \
  --intermediate_size 1536 --max_seq_length 256 \
  --batch_size 4 --gradient_accumulation_steps 8 \
  --learning_rate 5e-4 --use_amp --gradient_checkpointing \
  --num_epochs 1 \
  --output_dir ./outputs/sanity_dummy
```

## 5) Small real-data run (C4 streaming)
```bash
!python train_ultrathink.py \
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

## 6) Monitoring
- Watch the printed average loss to trend downward.
- Optional: add `--use_wandb --run_name ultrathink_colab` to log to Weights & Biases.

## 7) Switching datasets
See `docs/datasets.md` for details. Examples:
```bash
# Wikipedia streaming
!python train_ultrathink.py --dataset wikipedia --dataset_subset 20231101.en --streaming --tokenizer_name gpt2

# Custom JSONL
!python train_ultrathink.py --dataset custom --data_path /content/data.jsonl --text_column text --tokenizer_name gpt2

# Mixed datasets (50/50 wikitext + openwebtext)
!python train_ultrathink.py --mix_datasets "wikitext:0.5,openwebtext:0.5" --tokenizer_name gpt2 --streaming
```

## Tips
- If you hit non-finite losses early, lower LR (`2e-5`), keep `--max_seq_length 512`, or increase warmups.
- FlashAttention may not install on all Colab GPUs; SDPA fallback is already enabled in the model.
