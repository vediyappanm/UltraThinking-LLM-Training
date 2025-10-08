# Training on Small Datasets

This guide shows you how to effectively train and monitor ULTRATHINK models on small datasets. Perfect for validating your setup, experimenting with hyperparameters, or working with limited compute resources.

## 🎯 Why Start Small?

- **Fast iteration** - Quickly test configurations and debug issues
- **Resource efficient** - Train on consumer GPUs or even CPU
- **Validate learning** - Ensure your model learns before scaling up
- **Cost effective** - Minimize cloud compute costs during experimentation

## ⚙️ Recommended Hyperparameters

For stable training on small datasets:

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| **Sequence Length** | 512 | Start here, increase gradually |
| **Learning Rate** | 5e-5 to 1e-4 | Lower is safer for small data |
| **AMP Warmup** | 200 steps | Disable autocast initially |
| **DRE Warmup** | 500 steps | Disable DRE at start |
| **Grad Accumulation** | Adjust to fit memory | Higher = more stable gradients |

## 🚀 Example: C4 Streaming (Small Model)

This example trains a small model on the C4 dataset with streaming:

```bash
# Set environment variables for stability
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

**Model specs:** ~50M parameters, fits in 4-8GB GPU memory

## 📊 Monitoring Your Training

### Console Output
The printed loss is averaged over recent steps. Look for:
- **Gradual downward trend** over thousands of steps
- **Stable values** (no sudden spikes or NaN)
- **Consistent throughput** (tokens/sec)

### Weights & Biases Integration
Enable W&B for beautiful visualizations:
```bash
python train_ultrathink.py \
  --use_wandb --run_name my_experiment \
  # ... other args
```

### Troubleshooting Early Instability
If you see unstable losses in the first few hundred steps:
- ✅ Lower learning rate (try 1e-5)
- ✅ Keep sequence length at 512 longer
- ✅ Increase warmup steps (3000-5000)
- ✅ Disable AMP temporarily

## 📁 Using Custom/Local Datasets

Train on your own data with the custom dataset option:

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

### Data Format

Your `.jsonl` file should have one JSON object per line:

```json
{"text": "First sample text here..."}
{"text": "Second sample text here..."}
{"text": "Third sample text here..."}
```

**Tips:**
- Each text should be a complete thought or paragraph
- Aim for 50-100 tokens per sample on average
- Remove any special formatting or control characters
- Ensure UTF-8 encoding

## 🎓 Best Practices

1. **Start with a baseline** - Train without MoE/DRE first
2. **Monitor closely** - Check loss curves every few hundred steps
3. **Save checkpoints** - Use `--save_frequency` to save regularly
4. **Validate often** - Set `--eval_frequency 1` for frequent validation
5. **Document experiments** - Keep notes on what works

## 📚 Next Steps

- **[Advanced Training](training_full.md)** - Enable MoE, DRE, and Constitutional AI
- **[DeepSpeed](training_deepspeed.md)** - Scale to larger models
- **[Evaluation](evaluation.md)** - Benchmark your trained models
- **[Join Discord](https://discord.gg/ek2x9Rmk)** - Share your results!

---

💬 **Questions?** Join our [Discord](https://discord.gg/ek2x9Rmk) or check the [FAQ](faq.md)
