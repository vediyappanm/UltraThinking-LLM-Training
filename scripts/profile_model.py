"""Profile model performance and memory usage"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import argparse
from torch.profiler import profile, ProfilerActivity, record_function
from src.models.ultrathink import UltraThinkModel, UltraThinkConfig
from src.models.architecture import ModelConfig


def profile_model(model, input_shape=(2, 512), device='cuda', num_iters=10):
    """Profile model forward and backward pass"""
    print(f"Profiling model on {device}")
    print(f"Input shape: {input_shape}")
    print(f"Number of iterations: {num_iters}")
    print("=" * 60)
    
    model = model.to(device)
    model.train()
    
    vocab_size = model.config.model_config.vocab_size
    batch_size, seq_length = input_shape
    
    # Dummy input
    input_ids = torch.randint(0, vocab_size, input_shape, device=device)
    labels = torch.randint(0, vocab_size, input_shape, device=device)
    
    print("Warming up...")
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    print("Profiling...")
    # Profile
    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i in range(num_iters):
            with record_function(f"iteration_{i}"):
                output = model(input_ids=input_ids, labels=labels)
                loss = output['loss']
                loss.backward()
                
                if device == 'cuda':
                    torch.cuda.synchronize()
    
    # Print results
    print("\n" + "=" * 60)
    print("Top 20 Operations by Time")
    print("=" * 60)
    sort_key = "cuda_time_total" if device == 'cuda' else "cpu_time_total"
    print(prof.key_averages().table(
        sort_by=sort_key, row_limit=20
    ))
    
    # Memory stats
    if torch.cuda.is_available() and device == 'cuda':
        print("\n" + "=" * 60)
        print("GPU Memory Statistics")
        print("=" * 60)
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    
    # Timing stats
    print("\n" + "=" * 60)
    print("Performance Metrics")
    print("=" * 60)
    
    # Measure inference time
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()
    
    avg_time = (end - start) / 10
    tokens_per_sec = (batch_size * seq_length) / avg_time
    
    print(f"Average forward pass time: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {tokens_per_sec:.0f} tokens/second")
    print(f"Throughput: {batch_size / avg_time:.2f} samples/second")
    
    return prof


def build_test_model(size='tiny'):
    """Build a test model of specified size"""
    configs = {
        'tiny': {
            'n_embd': 256,
            'n_layer': 4,
            'n_head': 4,
            'intermediate_size': 1024
        },
        'small': {
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'intermediate_size': 3072
        },
        'medium': {
            'n_embd': 1024,
            'n_layer': 24,
            'n_head': 16,
            'intermediate_size': 4096
        }
    }
    
    model_params = configs.get(size, configs['tiny'])
    
    model_config = ModelConfig(
        vocab_size=50257,
        n_positions=512,
        n_embd=model_params['n_embd'],
        n_layer=model_params['n_layer'],
        n_head=model_params['n_head'],
        n_kv_head=model_params['n_head'] // 2,
        intermediate_size=model_params['intermediate_size'],
        flash_attention=False,  # For CPU compatibility
        gradient_checkpointing=False
    )
    
    config = UltraThinkConfig(
        model_config=model_config,
        enable_dre=False,
        enable_constitutional=False,
        enable_moe=False,
        enable_multimodal=False,
        enable_rlhf=False
    )
    
    return UltraThinkModel(config)


def main():
    parser = argparse.ArgumentParser(description='Profile ULTRATHINK model')
    parser.add_argument('--size', type=str, default='tiny', choices=['tiny', 'small', 'medium'],
                        help='Model size')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_iters', type=int, default=10, help='Number of profiling iterations')
    parser.add_argument('--export_trace', type=str, default=None,
                        help='Path to export Chrome trace')
    
    args = parser.parse_args()
    
    print("Building model...")
    model = build_test_model(args.size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model size: {args.size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter size: {total_params * 4 / 1e9:.2f} GB (float32)")
    print()
    
    # Profile
    prof = profile_model(
        model,
        input_shape=(args.batch_size, args.seq_length),
        device=args.device,
        num_iters=args.num_iters
    )
    
    # Export trace
    if args.export_trace:
        prof.export_chrome_trace(args.export_trace)
        print(f"\nTrace exported to: {args.export_trace}")
        print("View in Chrome at: chrome://tracing")


if __name__ == "__main__":
    main()
