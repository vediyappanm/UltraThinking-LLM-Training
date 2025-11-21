#!/usr/bin/env python3
"""
Quick diagnostic test - identifies where training hangs
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import time

print("=" * 60)
print("🔍 ULTRATHINK DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: PyTorch
print("\n[1/6] Testing PyTorch...")
print(f"  ✅ PyTorch version: {torch.__version__}")
print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✅ CUDA device: {torch.cuda.get_device_name(0)}")

# Test 2: Model import
print("\n[2/6] Testing model import...")
try:
    from src.models import UltraThinkModel, UltraThinkConfig, ModelConfig
    print("  ✅ Model imports OK")
except Exception as e:
    print(f"  ❌ Model import failed: {e}")
    sys.exit(1)

# Test 3: Create config
print("\n[3/6] Creating model config...")
try:
    base_config = ModelConfig(
        vocab_size=50000,
        n_embd=512,
        n_layer=4,
        n_head=8,
        n_positions=512,
    )
    
    config = UltraThinkConfig(
        model_config=base_config,
        enable_dre=False,
        enable_constitutional=False,
        enable_moe=False,
        enable_multimodal=False,
    )
    print("  ✅ Config created")
except Exception as e:
    print(f"  ❌ Config creation failed: {e}")
    sys.exit(1)

# Test 4: Create model
print("\n[4/6] Creating model...")
try:
    start = time.time()
    model = UltraThinkModel(config)
    model = model.cuda() if torch.cuda.is_available() else model
    elapsed = time.time() - start
    print(f"  ✅ Model created in {elapsed:.2f}s")
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ Model size: {num_params:.2f}M parameters")
except Exception as e:
    print(f"  ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n[5/6] Testing forward pass...")
try:
    batch_size = 2
    seq_len = 128
    vocab_size = 50000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    print(f"  ⏳ Running forward pass (batch={batch_size}, seq={seq_len})...")
    start = time.time()
    
    with torch.no_grad():
        output = model(input_ids)
    
    elapsed = time.time() - start
    print(f"  ✅ Forward pass completed in {elapsed:.2f}s")
    print(f"  ✅ Output shape: {output.logits.shape if hasattr(output, 'logits') else output[0].shape}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Training step
print("\n[6/6] Testing training step...")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        labels = labels.cuda()
    
    print(f"  ⏳ Running training step...")
    start = time.time()
    
    # Forward
    output = model(input_ids, labels=labels)
    loss = output.loss if hasattr(output, 'loss') else output[0]
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    elapsed = time.time() - start
    print(f"  ✅ Training step completed in {elapsed:.2f}s")
    print(f"  ✅ Loss: {loss.item():.4f}")
except Exception as e:
    print(f"  ❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🚀 Your system is ready for training!")
print("\nRun training with:")
print("  python train_minimal.py --max_steps 100 --batch_size 2")
print("=" * 60)
