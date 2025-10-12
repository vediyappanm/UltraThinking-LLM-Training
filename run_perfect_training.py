#!/usr/bin/env python3
"""
ULTRATHINK Perfect Training Launcher
Simplified launcher for the perfect training configuration
"""

import subprocess
import sys
import os
from pathlib import Path

# Perfect training configuration
PERFECT_CONFIG = {
    # Model Architecture
    "vocab_size": 50257,
    "hidden_size": 512,
    "num_layers": 6,
    "num_heads": 8,
    "num_kv_heads": 4,
    "intermediate_size": 2048,
    "max_seq_length": 256,
    "activation": "swiglu",
    
    # MoE Configuration (FIXED)
    "enable_moe": True,
    "num_knowledge_experts": 4,
    "num_skill_experts": 2,
    "num_meta_experts": 1,
    "num_safety_experts": 1,
    "moe_top_k": 2,
    "expert_capacity": 1.5,
    
    # Load Balancing Weights (CRITICAL FIXES)
    "load_balance_weight": 0.1,
    "z_loss_weight": 0.0001,
    "importance_weight": 0.05,
    
    # Training Hyperparameters (OPTIMIZED)
    "batch_size": 2,
    "gradient_accumulation_steps": 32,
    "learning_rate": 0.0001,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "warmup_steps": 1000,
    "max_steps": 100000,
    "num_epochs": 1,
    "gradient_clipping": 0.5,
    
    # Regularization (INCREASED)
    "dropout": 0.15,
    "attention_dropout": 0.15,
    
    # Optimization Features
    "gradient_checkpointing": True,
    "use_amp": True,
    "amp_warmup_steps": 500,
    
    # Dynamic Reasoning Engine
    "enable_dre": True,
    "dre_warmup_steps": 1000,
    
    # Dataset Configuration
    "dataset": "c4",
    "dataset_subset": "en",
    "tokenizer_name": "gpt2",
    "streaming": True,
    "train_samples": 10000,
    "val_samples": 1000,
    "num_workers": 2,
    
    # Logging & Monitoring (ENHANCED)
    "use_mlflow": True,
    "mlflow_tracking_uri": "file:./mlruns",
    "mlflow_experiment": "UltraThinking-LLM-Training",
    "run_name": "ultrathink_fixed_routing_v2",
    "perf_log_interval": 5,
    "eval_frequency": 50,
    
    # Output
    "output_dir": "./outputs/ultrathink_fixed",
}


def build_command(config, script="train_ultrathink.py"):
    """Build the training command from config dictionary"""
    cmd = [sys.executable, script]
    
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    return cmd


def check_environment():
    """Check if the environment is ready for training"""
    print("=" * 50)
    print("ULTRATHINK Perfect Training Launcher")
    print("=" * 50)
    print()
    
    # Check if train_ultrathink.py exists
    if not Path("train_ultrathink.py").exists():
        print("❌ Error: train_ultrathink.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ Warning: No GPU detected. Training will be slow on CPU.")
    except ImportError:
        print("⚠ Warning: PyTorch not installed. Cannot check GPU.")
    
    # Create output directory
    output_dir = Path(PERFECT_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    print()


def main():
    """Main entry point"""
    check_environment()
    
    # Build command
    cmd = build_command(PERFECT_CONFIG)
    
    print("Starting training with perfect configuration...")
    print()
    print("Configuration highlights:")
    print(f"  • Model: {PERFECT_CONFIG['hidden_size']}d, {PERFECT_CONFIG['num_layers']} layers")
    print(f"  • MoE: {PERFECT_CONFIG['num_knowledge_experts']} knowledge experts, top-k={PERFECT_CONFIG['moe_top_k']}")
    print(f"  • Batch: {PERFECT_CONFIG['batch_size']} × {PERFECT_CONFIG['gradient_accumulation_steps']} = {PERFECT_CONFIG['batch_size'] * PERFECT_CONFIG['gradient_accumulation_steps']} effective")
    print(f"  • Learning rate: {PERFECT_CONFIG['learning_rate']}")
    print(f"  • Load balance weight: {PERFECT_CONFIG['load_balance_weight']} (10x stronger)")
    print(f"  • Z-loss weight: {PERFECT_CONFIG['z_loss_weight']} (10x weaker)")
    print()
    print("Expected improvements by step 50-100:")
    print("  • Entropy: 0.52 → 0.8-1.2")
    print("  • Max Expert: 100% → 50-70%")
    print("  • Aux Loss: 8.0 → 2.0-4.0")
    print()
    print("-" * 50)
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 50)
        print("✓ Training completed successfully!")
        print("=" * 50)
        print()
        print(f"Output directory: {PERFECT_CONFIG['output_dir']}")
        print("MLflow logs: ./mlruns")
        print()
        print("To view training metrics:")
        print("  mlflow ui --backend-store-uri ./mlruns --port 5000")
        print()
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 50)
        print("❌ Training failed!")
        print("=" * 50)
        print()
        print("Check the logs above for error details.")
        return 1
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
