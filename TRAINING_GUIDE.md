# ULTRATHINK Training Guide

## Recent Improvements (2025-10-05)

### 1. Reduced Verbose Logging ✅

**Changed files:**
- `src/models/dynamic_reasoning.py` - DRE logs now at DEBUG level
- `src/models/ultrathink.py` - MoE logs now at DEBUG level  
- `src/training/loop.py` - Reduced per-batch debug prints

**What you'll see now:**
```
[step] step=0 loss=10.99 ppl=59234 toks/s=95.2 moe=[entropy=0.97,max_exp=50.0%] dre=[path=expert]
[step] step=1 loss=10.95 ppl=57543 toks/s=98.1 moe=[entropy=0.98,max_exp=48.5%] dre=[path=expert]
[step] step=2 loss=10.89 ppl=54210 toks/s=99.3 moe=[entropy=1.01,max_exp=47.2%] dre=[path=expert]
...
```

**What you won't see anymore:**
- Repeated "DRE: Using override path=expert" for every micro-batch
- "UltraThinkCore: reused persisted reasoning_path" messages
- "DEBUG MoE configured_layers" messages

### 2. How to See Full Debug Logs (If Needed)

Set logging level to DEBUG in your training script:
```python
import logging
logging.getLogger('src.models.dynamic_reasoning').setLevel(logging.DEBUG)
logging.getLogger('src.models.ultrathink').setLevel(logging.DEBUG)
```

Or via command line:
```bash
export ULTRATHINK_LOG_LEVEL=DEBUG
```

## Training Configuration for Loss Decrease

### Current Status (from your logs)
- ✅ MoE is working: `entropy=0.97`, `max_exp=50%` (healthy routing)
- ✅ DRE is working: Successfully forcing expert path
- ✅ No crashes or gradient explosions
- ⚠️ Only ~13 optimizer steps with 200 samples (too few to see learning)

### Recommended Configuration for Visible Loss Decrease

```bash
python train_ultrathink.py \
  --dataset c4 --dataset_subset en --streaming \
  --train_samples 5000 --val_samples 1000 \
  --tokenizer_name gpt2 --vocab_size 50257 \
  --hidden_size 512 --num_layers 6 --num_heads 8 --num_kv_heads 4 \
  --intermediate_size 2048 --max_seq_length 256 \
  --enable_moe --enable_dre --dre_force_path expert --dre_warmup_steps 500 \
  --num_knowledge_experts 4 --num_skill_experts 2 \
  --num_meta_experts 1 --num_safety_experts 1 \
  --moe_top_k 2 --expert_capacity 1.25 \
  --batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 --weight_decay 0.01 --warmup_steps 100 \
  --num_epochs 1 --use_amp --gradient_checkpointing \
  --load_balance_weight 0.001 --z_loss_weight 0.0001 \
  --importance_weight 0.001 --entropy_reg_weight 0.01 \
  --num_workers 0 --eval_frequency 50 --perf_log_interval 10 \
  --use_mlflow --run_name moe_5k_training \
  --output_dir ./outputs/moe_5k_training
```

**Key changes from your current run:**
- `train_samples: 200 → 5000` (312 optimizer steps instead of 13)
- `val_samples: 50 → 1000` (better validation statistics)
- `warmup_steps: 500 → 100` (reach full LR faster for small-scale run)
- `learning_rate: 3e-4 → 5e-5` (more stable for small model)
- `eval_frequency: 5 → 50` (don't eval so often at start)
- `perf_log_interval: 200 → 10` (see step logs every 10 steps)
- `num_workers: 2 → 0` (Windows compatibility)

### Expected Training Trajectory

**First 10 steps (warmup):**
```
step=0  loss=10.99 ppl=59234  ← Random initialization
step=1  loss=10.95 ppl=57543  ← Small decrease
step=2  loss=10.89 ppl=54210  ← Gradual improvement
...
step=10 loss=10.50 ppl=36316  ← Still high but decreasing
```

**After 50 steps:**
```
step=50 loss=9.80 ppl=18079  ← Clear downward trend
```

**After 100 steps:**
```
step=100 loss=9.20 ppl=9897  ← Sub-10K perplexity
```

**After 200 steps:**
```
step=200 loss=8.50 ppl=4914  ← Model is learning patterns
```

### MoE Health Indicators

**Good signs:**
- Entropy stays between 0.9-1.4 (balanced routing)
- Max expert % stays below 60% (no collapse)
- Auxiliary loss components stable
- Gradient norms < 50

**Warning signs:**
- Entropy drops below 0.5 (expert collapse)
- Max expert % consistently > 80% (over-concentration)
- Loss not decreasing after 50 steps (learning stalled)
- Gradient norms > 1000 (instability)

## Monitoring Your Training

### 1. Console Output
Clean, concise step logs every 10 steps showing:
- Loss and perplexity
- Token throughput
- MoE metrics (entropy, max expert %)
- DRE metrics (complexity, path)
- Gradient norms (after first accumulation step)

### 2. MLflow Dashboard
```bash
mlflow ui --backend-store-uri file:./mlruns
```
Then open http://localhost:5000 to see:
- Loss curves over time
- Expert utilization heatmaps
- Per-expert usage percentages
- Gradient norm trends

### 3. Key Metrics to Track

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Loss decrease | Steady downward | Flat for 50 steps | Increasing |
| Perplexity | < 10K by step 100 | Still > 20K | Exploding |
| MoE entropy | 0.9-1.4 | 0.5-0.8 | < 0.5 |
| Max expert % | < 60% | 60-80% | > 80% |
| Grad norm | < 50 | 50-500 | > 1000 |

## Troubleshooting

### Loss not decreasing after 100 steps
- **Try:** Reduce LR to `3e-5`
- **Try:** Increase `entropy_reg_weight` to `0.02` temporarily
- **Try:** Check if gradients are flowing (look for non-zero grad norms)

### Expert collapse (entropy < 0.5)
- **Try:** Increase `entropy_reg_weight` to `0.05`
- **Try:** Increase `load_balance_weight` to `0.005`
- **Try:** Add `--dre_warmup_steps 1000`

### Training too slow
- **Try:** Reduce `max_seq_length` to `128`
- **Try:** Turn off gradient checkpointing (if you have enough memory)
- **Try:** Increase `batch_size` to 2 (if memory allows)

### Out of memory
- **Try:** Enable gradient checkpointing
- **Try:** Reduce `hidden_size` to `384`
- **Try:** Reduce `num_knowledge_experts` to `2`

## Next Steps

1. ✅ **Run the 5K samples configuration** above
2. ⏳ **Monitor loss for 100+ steps**
3. ⏳ **Check MoE metrics remain healthy**
4. ⏳ **Validate that perplexity drops below 10K**
5. 📊 **Review MLflow dashboard for detailed metrics**
6. 🎯 **Scale up to 50K+ samples for serious training**

## Summary of Code Changes

### Files Modified
- `src/models/dynamic_reasoning.py` - Line 466, 685-688 (INFO → DEBUG)
- `src/models/ultrathink.py` - Lines 223-240, 282 (INFO → DEBUG)
- `src/training/loop.py` - Lines 49-51, 60-64, 160, 184-190 (Reduced prints)

### What Changed
- **Before:** Every micro-batch logged DRE and MoE debug messages
- **After:** Only step-level summaries shown (every 10-16 batches)
- **Result:** Clean, readable training logs focused on progress

### Compatibility
- ✅ All existing functionality preserved
- ✅ Full debug logs still available via logging level
- ✅ MLflow tracking unaffected
- ✅ Validation metrics unchanged
- ✅ Works on Windows, Linux, Colab

---

**Created:** 2025-10-05  
**Version:** 1.0  
**Status:** Ready for training
