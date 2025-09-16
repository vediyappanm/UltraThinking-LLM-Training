"""
Advanced Text Generation Utilities
For inference, sampling, and interactive generation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import time
import json
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    min_length: int = 0
    no_repeat_ngram_size: int = 0


class AdvancedGenerator:
    """Advanced text generation with multiple sampling strategies"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None
    ) -> Union[str, List[str]]:
        """Generate text with advanced sampling"""
        if config is None:
            config = GenerationConfig()
        
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        original_length = input_ids.size(1)
        
        # Generation loop
        generated = input_ids.clone()
        generated_tokens = []
        
        for step in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(input_ids=generated, use_cache=False)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated[0], config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p filtering
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
            
            # Sample next token
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS
            if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
                break
            
            # Append token
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            generated_tokens.append(next_token.item())
            
            # Streaming callback
            if stream and callback:
                partial_text = self.tokenizer.decode(generated_tokens)
                callback(partial_text)
            
            # Check minimum length
            if step < config.min_length:
                continue
            
            # Early stopping conditions
            if config.early_stopping and self._should_stop_early(generated_tokens, config):
                break
        
        # Decode final result
        full_text = self.tokenizer.decode(generated[0].cpu().tolist())
        generated_text = full_text[len(prompt):]
        
        return generated_text
    
    def _apply_repetition_penalty(self, logits, input_ids, penalty):
        """Apply repetition penalty to logits"""
        score = torch.gather(logits, 0, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(0, input_ids, score)
        return logits
    
    def _top_k_filtering(self, logits, top_k):
        """Apply top-k filtering"""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits, top_p):
        """Apply top-p (nucleus) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _should_stop_early(self, tokens, config):
        """Check if generation should stop early"""
        if len(tokens) < 3:
            return False
        
        # Stop if repeating n-grams
        if config.no_repeat_ngram_size > 0:
            ngram_size = config.no_repeat_ngram_size
            if len(tokens) >= ngram_size * 2:
                recent_ngram = tokens[-ngram_size:]
                prev_ngram = tokens[-ngram_size*2:-ngram_size]
                if recent_ngram == prev_ngram:
                    return True
        
        return False
    
    def batch_generate(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        batch_size: int = 4
    ) -> List[str]:
        """Generate text for multiple prompts in batches"""
        if config is None:
            config = GenerationConfig()
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                result = self.generate(prompt, config)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def interactive_chat(self, system_prompt: str = "", config: GenerationConfig = None):
        """Interactive chat interface"""
        if config is None:
            config = GenerationConfig(max_new_tokens=256, temperature=0.7)
        
        print("=== Interactive Chat ===")
        print("Type 'quit' to exit, 'clear' to clear history")
        print("=" * 30)
        
        conversation_history = system_prompt
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                conversation_history = system_prompt
                print("History cleared.")
                continue
            
            # Add user input to conversation
            conversation_history += f"\nUser: {user_input}\nAssistant: "
            
            # Generate response
            print("Assistant: ", end="", flush=True)
            
            def stream_callback(text):
                print(text, end="", flush=True)
            
            response = self.generate(
                conversation_history,
                config,
                stream=True,
                callback=stream_callback
            )
            
            # Add response to history
            conversation_history += response
            print()  # New line after response


class ControllableGenerator:
    """Generator with controllable attributes (sentiment, style, etc.)"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.base_generator = AdvancedGenerator(model, tokenizer, device)
    
    def generate_with_style(
        self,
        prompt: str,
        style: str = "neutral",
        config: GenerationConfig = None
    ) -> str:
        """Generate text with specific style"""
        style_prompts = {
            "formal": "Write in a formal, professional tone: ",
            "casual": "Write in a casual, friendly tone: ",
            "creative": "Write creatively and imaginatively: ",
            "technical": "Write in a technical, precise manner: ",
            "humorous": "Write with humor and wit: ",
            "serious": "Write in a serious, thoughtful tone: "
        }
        
        style_prefix = style_prompts.get(style, "")
        full_prompt = style_prefix + prompt
        
        return self.base_generator.generate(full_prompt, config)
    
    def generate_with_length_control(
        self,
        prompt: str,
        target_length: str = "medium",
        config: GenerationConfig = None
    ) -> str:
        """Generate text with controlled length"""
        if config is None:
            config = GenerationConfig()
        
        length_configs = {
            "short": GenerationConfig(max_new_tokens=50, **config.__dict__),
            "medium": GenerationConfig(max_new_tokens=150, **config.__dict__),
            "long": GenerationConfig(max_new_tokens=400, **config.__dict__)
        }
        
        target_config = length_configs.get(target_length, config)
        return self.base_generator.generate(prompt, target_config)
    
    def generate_with_constraints(
        self,
        prompt: str,
        must_include: List[str] = None,
        must_avoid: List[str] = None,
        config: GenerationConfig = None
    ) -> str:
        """Generate text with inclusion/exclusion constraints"""
        if config is None:
            config = GenerationConfig()
        
        # Simple constraint implementation
        constraint_prompt = prompt
        
        if must_include:
            constraint_prompt += f" (Must include: {', '.join(must_include)})"
        
        if must_avoid:
            constraint_prompt += f" (Avoid mentioning: {', '.join(must_avoid)})"
        
        return self.base_generator.generate(constraint_prompt, config)


class BenchmarkGenerator:
    """Generator for running generation benchmarks"""
    
    def __init__(self, model, tokenizer, device):
        self.generator = AdvancedGenerator(model, tokenizer, device)
    
    def speed_benchmark(
        self,
        prompts: List[str],
        config: GenerationConfig = None,
        num_runs: int = 3
    ) -> Dict[str, float]:
        """Benchmark generation speed"""
        if config is None:
            config = GenerationConfig(max_new_tokens=100)
        
        times = []
        total_tokens = 0
        
        for run in range(num_runs):
            start_time = time.time()
            
            for prompt in prompts:
                result = self.generator.generate(prompt, config)
                total_tokens += len(self.generator.tokenizer.encode(result))
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        tokens_per_second = (total_tokens / num_runs) / avg_time
        
        return {
            "avg_time_per_batch": avg_time,
            "tokens_per_second": tokens_per_second,
            "total_tokens_generated": total_tokens // num_runs,
            "num_prompts": len(prompts)
        }
    
    def quality_benchmark(
        self,
        prompts: List[str],
        configs: List[GenerationConfig]
    ) -> Dict[str, List[str]]:
        """Compare generation quality across different configs"""
        results = {}
        
        for i, config in enumerate(configs):
            config_name = f"config_{i}"
            results[config_name] = []
            
            for prompt in prompts:
                result = self.generator.generate(prompt, config)
                results[config_name].append(result)
        
        return results


def create_generation_configs() -> Dict[str, GenerationConfig]:
    """Create predefined generation configurations"""
    return {
        "creative": GenerationConfig(
            temperature=0.9,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        "balanced": GenerationConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0
        ),
        "focused": GenerationConfig(
            temperature=0.3,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.0
        ),
        "deterministic": GenerationConfig(
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.0
        ),
        "diverse": GenerationConfig(
            temperature=1.0,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.2
        )
    }


def demo_generation(model, tokenizer, device):
    """Demonstrate various generation capabilities"""
    generator = AdvancedGenerator(model, tokenizer, device)
    controllable = ControllableGenerator(model, tokenizer, device)
    configs = create_generation_configs()
    
    print("=== Generation Demo ===")
    
    # Basic generation
    prompt = "The future of artificial intelligence"
    print(f"\nPrompt: {prompt}")
    
    for name, config in configs.items():
        print(f"\n{name.upper()} generation:")
        result = generator.generate(prompt, config)
        print(result[:200] + "..." if len(result) > 200 else result)
    
    # Style-controlled generation
    print("\n=== Style-Controlled Generation ===")
    styles = ["formal", "casual", "creative", "technical"]
    
    for style in styles:
        print(f"\n{style.upper()} style:")
        result = controllable.generate_with_style(prompt, style, configs["balanced"])
        print(result[:150] + "..." if len(result) > 150 else result)
    
    # Length-controlled generation
    print("\n=== Length-Controlled Generation ===")
    lengths = ["short", "medium", "long"]
    
    for length in lengths:
        print(f"\n{length.upper()} length:")
        result = controllable.generate_with_length_control(prompt, length)
        print(f"({len(result)} chars) {result}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Example usage
    print("Generation utilities loaded successfully!")
    print("Use demo_generation(model, tokenizer, device) to see examples.")
