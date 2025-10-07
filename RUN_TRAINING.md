# Quick Start Guide - Running Training After Fixes

## ✅ Fixes Applied

The following critical fixes have been implemented to resolve the training hang issue:

1. **DataLoader Configuration** - Set `num_workers=0` for Windows compatibility
2. **Iterator Error Handling** - Proper try-catch for batch fetching
3. **Memory Cleanup** - Periodic CUDA cache clearing and garbage collection
4. **Heartbeat Logging** - Visual confirmation every 10 batches
5. **Validation Loop Fix** - Same error handling as training loop

## 🚀 How to Run Training

### Option 1: Quick Test (Recommended First)
Test the DataLoader configuration:
```bash
python test_dataloader.py
```

**Expected Output:**
```
============================================================
Testing DataLoader Configuration
============================================================
✓ DataLoader created successfully
✓ Iterator created successfully
✓ First batch fetched successfully
✓ Second batch fetched successfully (CRITICAL TEST)
✓ Batch 4 fetched successfully
...
✓ ALL TESTS PASSED - DataLoader is working correctly!
============================================================
```

### Option 2: Run Full Training
Once the test passes, run your actual training:
```bash
python train_ultrathink.py \
    --dataset dummy \
    --train_samples 1000 \
    --val_samples 100 \
    --batch_size 2 \
    --num_epochs 1 \
    --max_seq_length 512 \
    --gradient_accumulation_steps 1
```

## 📊 What You Should See

### ✅ Successful Training Output:
```
[DEBUG] Starting training loop, train_loader length estimate: 500
[DEBUG] gradient_accumulation_steps: 1

=== FIRST STEP MoE VERIFICATION ===
MoE info keys: ['aux_losses', 'level_outputs', 'num_experts_used', 'expert_utilization']

[step] step=1 loss=11.0130 ppl=60656.23 toks/s=267.9 moe=[...] dre=[...] grad=[...]
[step] step=2 loss=10.9845 ppl=58432.11 toks/s=285.3 moe=[...] dre=[...] grad=[...]
[step] step=3 loss=10.9612 ppl=57103.45 toks/s=291.7 moe=[...] dre=[...] grad=[...]
...
[HEARTBEAT] batch_idx=10, global_step=10, elapsed=25.3s
[step] step=11 loss=10.8234 ppl=50234.67 toks/s=295.1 moe=[...] dre=[...] grad=[...]
...
[DEBUG] Completed epoch - processed 500 batches
[train] avg_loss=10.5432 avg_ppl=37234.56 epoch_time=125.5s toks/s=4096.2
```

### ❌ Previous Issue (FIXED):
```
[step] step=1 loss=11.0130 ppl=60656.23 toks/s=267.9 moe=[...] dre=[...] grad=[...]
<HANGS HERE - NO MORE OUTPUT>
```

## 🔍 Monitoring Training Progress

### Key Indicators Training is Working:
1. ✅ **Step numbers increase**: `[step] step=1`, `[step] step=2`, etc.
2. ✅ **Heartbeat messages**: `[HEARTBEAT] batch_idx=10, global_step=10, elapsed=25.3s`
3. ✅ **Loss decreases**: Loss values should generally decrease over time
4. ✅ **Epoch completes**: `[DEBUG] Completed epoch - processed N batches`

### If You Still See Issues:
1. **Check GPU memory**: Run `nvidia-smi` in another terminal
2. **Reduce batch size**: Try `--batch_size 1`
3. **Reduce sequence length**: Try `--max_seq_length 256`
4. **Check logs**: Look for ERROR messages in the output

## 📈 Performance Notes

### Windows vs Linux:
- **Windows** (`num_workers=0`): ~200-300 tokens/sec (single-threaded data loading)
- **Linux** (`num_workers=4-6`): ~800-1200 tokens/sec (multi-process data loading)

### Optimization Tips:
1. **Increase batch_size**: If you have GPU memory, use larger batches
2. **Use gradient_accumulation**: Simulate larger batches without more memory
3. **Enable mixed precision**: Already enabled via AMP in the code
4. **Pre-process data**: Cache tokenized data to disk if using real datasets

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```bash
--batch_size 1 --gradient_accumulation_steps 4
```

### Issue: Training still hangs
**Solution:**
1. Check if you have other Python processes running
2. Restart your terminal/IDE
3. Clear CUDA cache: `torch.cuda.empty_cache()`
4. Reboot your system

### Issue: Loss is NaN
**Solution:**
- Reduce learning rate: `--learning_rate 1e-5`
- Check gradient clipping: `--gradient_clipping 1.0`
- Enable AMP warmup (already in code)

### Issue: Very slow training
**Expected:** On Windows with `num_workers=0`, training is slower than Linux
**Solutions:**
- Use smaller sequences: `--max_seq_length 256`
- Use larger batches: `--batch_size 4` (if GPU allows)
- Switch to Linux or WSL2 for production training

## 📝 Files Modified

1. **`train_ultrathink.py`**: DataLoader configuration (lines 455-483)
2. **`src/training/loop.py`**: Training loop iterator and error handling (lines 48-501)
3. **`src/training/loop.py`**: Validation loop fix (lines 540-577)

## 📚 Additional Resources

- **Full Fix Details**: See `TRAINING_HANG_FIX.md`
- **Project Structure**: See `PROJECT_STRUCTURE.md`
- **Training Guide**: See `TRAINING_GUIDE.md`
- **Advanced Training**: See `ADVANCED_TRAINING_GUIDE.md`

## ✨ Next Steps

Once training runs successfully:

1. **Monitor metrics**: Watch loss, perplexity, and MoE utilization
2. **Adjust hyperparameters**: Tune learning rate, batch size based on results
3. **Add real data**: Switch from dummy data to actual datasets
4. **Enable checkpointing**: Regularly save model checkpoints
5. **Run validation**: Monitor validation loss to prevent overfitting

## 🎯 Success Criteria

Training is working correctly when:
- ✅ Multiple steps complete without hanging
- ✅ Heartbeat messages appear regularly
- ✅ Loss values are finite and generally decreasing
- ✅ Full epoch completes successfully
- ✅ Validation runs without errors

---

**Need Help?** Check the error messages in the logs and refer to the troubleshooting section above.
