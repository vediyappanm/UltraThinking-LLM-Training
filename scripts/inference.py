#!/usr/bin/env python3
"""
Inference Script for Trained Models
Supports interactive chat, batch generation, and API serving
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.architecture import AdvancedGPTModel, ModelConfig
from utils.generation import AdvancedGenerator, ControllableGenerator, GenerationConfig, create_generation_configs

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class SimpleTokenizer:
    """Fallback tokenizer"""
    def __init__(self, vocab_file=None):
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                data = json.load(f)
                self.chars = data.get('chars', list('abcdefghijklmnopqrstuvwxyz'))
        else:
            self.chars = list('abcdefghijklmnopqrstuvwxyz')
        
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '') for t in tokens])


def load_model_and_tokenizer(checkpoint_path, device='cuda'):
    """Load trained model and tokenizer"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    if os.path.isdir(checkpoint_path):
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer.json")
    else:
        model_path = checkpoint_path
        tokenizer_path = None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    config_dict = checkpoint.get('config', {})
    model_config_dict = config_dict.get('model_config_dict', {})
    
    if not model_config_dict:
        # Fallback config
        model_config_dict = {
            'vocab_size': 50304,
            'n_positions': 2048,
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'n_kv_head': 4,
            'rotary_dim': 64,
            'intermediate_size': 3072,
            'activation': 'swiglu',
            'norm_type': 'rmsnorm',
            'norm_eps': 1e-5,
            'dropout': 0.0,
            'attention_dropout': 0.0,
            'residual_dropout': 0.1,
            'embed_dropout': 0.1,
            'tie_word_embeddings': True,
            'use_cache': True,
            'attention_bias': False,
            'mlp_bias': False,
            'flash_attention': True,
            'gradient_checkpointing': False,
            'max_position_embeddings': 2048
        }
    
    # Create model
    model_config = ModelConfig(**model_config_dict)
    model = AdvancedGPTModel(model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)
        
        if tokenizer_info.get('type') == 'tiktoken' and TIKTOKEN_AVAILABLE:
            tokenizer = tiktoken.get_encoding('gpt2')
        else:
            tokenizer = SimpleTokenizer()
    else:
        # Try tiktoken
        if TIKTOKEN_AVAILABLE:
            tokenizer = tiktoken.get_encoding('gpt2')
        else:
            tokenizer = SimpleTokenizer()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    # Support tiktoken Encoding (n_vocab) and SimpleTokenizer (vocab_size/chars)
    tok_vs = None
    for attr in ('vocab_size', 'n_vocab'):
        if hasattr(tokenizer, attr):
            tok_vs = getattr(tokenizer, attr)
            break
    if tok_vs is None:
        tok_vs = len(getattr(tokenizer, 'chars', [])) or 'unknown'
    print(f"Tokenizer: {tok_vs} vocab size")
    
    return model, tokenizer


def interactive_chat(model, tokenizer, device, overrides=None):
    """Interactive chat interface"""
    generator = AdvancedGenerator(model, tokenizer, device)
    controllable = ControllableGenerator(model, tokenizer, device)
    configs = create_generation_configs()
    
    print("\n" + "="*50)
    print("🤖 CLAUDE OPUS 4 SCALE MODEL - INTERACTIVE CHAT")
    print("="*50)
    print("Commands:")
    print("  /help - Show this help")
    print("  /config <name> - Change generation config (creative, balanced, focused, etc.)")
    print("  /style <style> - Set style (formal, casual, creative, technical, etc.)")
    print("  /clear - Clear conversation history")
    print("  /quit - Exit chat")
    print("="*50)
    
    current_config = configs['balanced']
    # Apply CLI overrides if provided
    if overrides:
        if 'max_new_tokens' in overrides and overrides['max_new_tokens'] is not None:
            current_config.max_new_tokens = int(overrides['max_new_tokens'])
        if 'temperature' in overrides and overrides['temperature'] is not None:
            current_config.temperature = float(overrides['temperature'])
        if 'top_k' in overrides and overrides['top_k'] is not None:
            current_config.top_k = int(overrides['top_k'])
        if 'top_p' in overrides and overrides['top_p'] is not None:
            current_config.top_p = float(overrides['top_p'])
        if 'do_sample' in overrides and overrides['do_sample'] is not None:
            current_config.do_sample = bool(overrides['do_sample'])
    current_style = None
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                cmd_parts = user_input[1:].split()
                cmd = cmd_parts[0].lower()
                
                if cmd == 'quit':
                    print("👋 Goodbye!")
                    break
                elif cmd == 'help':
                    print("\nAvailable configs:", list(configs.keys()))
                    print("Available styles: formal, casual, creative, technical, humorous, serious")
                    continue
                elif cmd == 'config' and len(cmd_parts) > 1:
                    config_name = cmd_parts[1]
                    if config_name in configs:
                        current_config = configs[config_name]
                        print(f"✅ Config changed to: {config_name}")
                    else:
                        print(f"❌ Unknown config: {config_name}")
                    continue
                elif cmd == 'style' and len(cmd_parts) > 1:
                    current_style = cmd_parts[1]
                    print(f"✅ Style set to: {current_style}")
                    continue
                elif cmd == 'clear':
                    conversation_history = ""
                    print("🗑️ Conversation history cleared")
                    continue
                else:
                    print("❌ Unknown command. Type /help for available commands.")
                    continue
            
            # Prepare prompt
            conversation_history += f"\nHuman: {user_input}\nAssistant: "
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            
            def stream_callback(text):
                print(text, end="", flush=True)
            
            if current_style:
                response = controllable.generate_with_style(
                    conversation_history, current_style, current_config
                )
            else:
                response = generator.generate(
                    conversation_history, current_config, stream=True, callback=stream_callback
                )
            
            if not current_config.do_sample or current_style:
                print(response, end="", flush=True)
            
            # Update conversation history
            conversation_history += response
            print()  # New line
            
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def batch_generation(model, tokenizer, device, prompts_file, output_file, config_name='balanced'):
    """Batch generation from file"""
    generator = AdvancedGenerator(model, tokenizer, device)
    configs = create_generation_configs()
    config = configs.get(config_name, configs['balanced'])
    
    # Load prompts
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Generating responses for {len(prompts)} prompts...")
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        response = generator.generate(prompt, config)
        results.append({
            'prompt': prompt,
            'response': response,
            'config': config_name
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


def benchmark_model(model, tokenizer, device):
    """Run model benchmarks"""
    from utils.generation import BenchmarkGenerator
    from utils.evaluation import run_evaluation
    
    print("🔬 Running model benchmarks...")
    
    # Speed benchmark
    benchmark_gen = BenchmarkGenerator(model, tokenizer, device)
    test_prompts = [
        "Explain quantum computing",
        "Write a short story about AI",
        "Describe the benefits of renewable energy",
        "What is the future of space exploration?",
        "How does machine learning work?"
    ]
    
    speed_results = benchmark_gen.speed_benchmark(test_prompts)
    print("\n📊 Speed Benchmark Results:")
    print(f"  Tokens/second: {speed_results['tokens_per_second']:.2f}")
    print(f"  Average time per batch: {speed_results['avg_time_per_batch']:.2f}s")
    
    # Quality evaluation
    print("\n🎯 Running quality evaluation...")
    eval_results = run_evaluation(model, tokenizer, device)
    
    return {**speed_results, **eval_results}


def main():
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=['chat', 'batch', 'benchmark'], default='chat')
    parser.add_argument("--prompts", type=str, help="Input prompts file for batch mode")
    parser.add_argument("--output", type=str, help="Output file for batch mode")
    parser.add_argument("--config", type=str, default='balanced', help="Generation config")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    # Optional generation overrides
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--do_sample", type=int, default=None, help="1 to enable sampling, 0 to disable")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    
    # Run inference
    if args.mode == 'chat':
        overrides = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'do_sample': (None if args.do_sample is None else bool(args.do_sample)),
        }
        interactive_chat(model, tokenizer, device, overrides)
    elif args.mode == 'batch':
        if not args.prompts or not args.output:
            print("❌ Batch mode requires --prompts and --output arguments")
            return
        batch_generation(model, tokenizer, device, args.prompts, args.output, args.config)
    elif args.mode == 'benchmark':
        results = benchmark_model(model, tokenizer, device)
        print("\n📋 Benchmark Summary:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
