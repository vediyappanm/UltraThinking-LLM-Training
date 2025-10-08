# Getting Started with ULTRATHINK

Welcome! This guide will get you up and running with ULTRATHINK in just 5 minutes. You'll learn how to install the framework and run your first training experiment.

## 📋 Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

## 🔧 Installation

**1. Clone the repository:**
```bash
git clone https://github.com/vediyappanm/UltraThinking-LLM-Training.git
cd UltraThinking-LLM-Training/deep
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Optional - Install Flash Attention (for supported GPUs):**
```bash
pip install flash-attn --no-build-isolation
```

> ⚠️ Flash Attention requires CUDA and may take several minutes to compile. The framework will automatically fall back to PyTorch SDPA if not available.

## 🚀 Your First Training Run

Let's verify your installation with a quick sanity check using dummy data:

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

**Expected output:** The training loop should run smoothly and report finite, decreasing losses.

## ✅ Verify Success

You should see output similar to:
```
Step 10/500 | Loss: 10.234 | LR: 5.0e-04 | Tokens/sec: 1234
Step 20/500 | Loss: 9.876 | LR: 5.0e-04 | Tokens/sec: 1245
...
```

If you see this, congratulations! 🎉 Your ULTRATHINK installation is working correctly.

## 🔧 Troubleshooting

**Issue: Non-finite losses (NaN/Inf) on first steps**
- Lower the learning rate: `--learning_rate 1e-4`
- Disable AMP temporarily: remove `--use_amp`
- Increase warmup steps: `--warmup_steps 500`

**Issue: Out of memory errors**
- Reduce batch size: `--batch_size 1`
- Enable gradient checkpointing: `--gradient_checkpointing`
- Reduce sequence length: `--max_seq_length 128`

**Issue: Import errors**
- Verify package versions match `requirements.txt`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## 📚 Next Steps

Now that you have ULTRATHINK running:

1. **[Train on Small Datasets](training_small.md)** - Learn best practices for real datasets
2. **[Try Google Colab](colab.md)** - Train with free GPU in your browser
3. **[Explore Advanced Features](training_full.md)** - MoE, DRE, and distributed training
4. **[Join our Discord](https://discord.gg/ek2x9Rmk)** - Get help and share your progress!

---

💬 **Need help?** Join our [Discord community](https://discord.gg/ek2x9Rmk) or check the [FAQ](faq.md)
