"""
Advanced Evaluation and Benchmarking Utilities
For comprehensive model assessment and comparison
"""

import os
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    max_eval_samples: int = 1000
    batch_size: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0


class PerplexityEvaluator:
    """Evaluate model perplexity on various datasets"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def evaluate_perplexity(self, texts: List[str], max_length: int = 2048) -> Dict[str, float]:
        """Calculate perplexity on a list of texts"""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) < 2:
                continue
                
            # Split into chunks if too long
            chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
            
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                    
                input_ids = torch.tensor([chunk[:-1]], device=self.device)
                labels = torch.tensor([chunk[1:]], device=self.device)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item() * len(chunk[1:])
                total_tokens += len(chunk[1:])
        
        if total_tokens == 0:
            return {'perplexity': float('inf'), 'loss': float('inf')}
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
        
        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'total_tokens': total_tokens
        }


class GenerationEvaluator:
    """Evaluate text generation quality"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def generate_text(
        self, 
        prompt: str, 
        config: EvaluationConfig
    ) -> str:
        """Generate text from a prompt"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        # Generate
        generated = input_ids.clone()
        
        for _ in range(config.max_new_tokens):
            # Forward pass
            outputs = self.model(input_ids=generated, use_cache=False)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            if config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
            
            # Stop if we hit EOS or max length
            if generated.size(1) >= input_ids.size(1) + config.max_new_tokens:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        # Extract only the new part
        prompt_length = len(prompt)
        return generated_text[prompt_length:]
    
    def evaluate_generation_quality(
        self, 
        prompts: List[str], 
        references: Optional[List[str]] = None,
        config: EvaluationConfig = None
    ) -> Dict[str, Any]:
        """Evaluate generation quality with various metrics"""
        if config is None:
            config = EvaluationConfig()
        
        results = {
            'generations': [],
            'metrics': {}
        }
        
        # Generate responses
        for prompt in prompts:
            generation = self.generate_text(prompt, config)
            results['generations'].append({
                'prompt': prompt,
                'generation': generation
            })
        
        # Calculate metrics if references provided
        if references and len(references) == len(prompts):
            generations = [r['generation'] for r in results['generations']]
            
            # BLEU score
            if BLEU_AVAILABLE:
                bleu_scores = []
                for gen, ref in zip(generations, references):
                    bleu = sacrebleu.sentence_bleu(gen, [ref])
                    bleu_scores.append(bleu.score)
                results['metrics']['bleu'] = np.mean(bleu_scores)
            
            # ROUGE scores
            if ROUGE_AVAILABLE:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
                
                for gen, ref in zip(generations, references):
                    scores = scorer.score(ref, gen)
                    for key in rouge_scores:
                        rouge_scores[key].append(scores[key].fmeasure)
                
                for key in rouge_scores:
                    results['metrics'][key] = np.mean(rouge_scores[key])
            
            # Length statistics
            gen_lengths = [len(gen.split()) for gen in generations]
            ref_lengths = [len(ref.split()) for ref in references]
            
            results['metrics']['avg_gen_length'] = np.mean(gen_lengths)
            results['metrics']['avg_ref_length'] = np.mean(ref_lengths)
            results['metrics']['length_ratio'] = np.mean(gen_lengths) / np.mean(ref_lengths)
        
        return results


class BenchmarkEvaluator:
    """Run standardized benchmarks"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.perplexity_evaluator = PerplexityEvaluator(model, tokenizer, device)
        self.generation_evaluator = GenerationEvaluator(model, tokenizer, device)
    
    def run_hellaswag_eval(self, dataset_path: str = None) -> Dict[str, float]:
        """Evaluate on HellaSwag dataset (common sense reasoning)"""
        # Simplified HellaSwag evaluation
        # In practice, you'd load the actual dataset
        
        examples = [
            {
                "context": "A woman is outside with a bucket and a dog. The dog is running around trying to avoid a bath. She",
                "choices": [
                    "rinses the bucket off with a hose and fills it with soap.",
                    "uses a hose to keep filling the bucket with water.",
                    "gets the dog wet, then it runs away again.",
                    "gets into the bucket."
                ],
                "correct": 2
            },
            # Add more examples...
        ]
        
        correct = 0
        total = 0
        
        for example in examples:
            context = example["context"]
            choices = example["choices"]
            correct_idx = example["correct"]
            
            # Calculate likelihood for each choice
            choice_scores = []
            for choice in choices:
                full_text = context + " " + choice
                tokens = self.tokenizer.encode(full_text)
                
                if len(tokens) < 2:
                    choice_scores.append(float('-inf'))
                    continue
                
                input_ids = torch.tensor([tokens[:-1]], device=self.device)
                labels = torch.tensor([tokens[1:]], device=self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs['loss']
                    choice_scores.append(-loss.item())
            
            # Check if highest scoring choice is correct
            predicted_idx = choice_scores.index(max(choice_scores))
            if predicted_idx == correct_idx:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {"hellaswag_accuracy": accuracy}
    
    def run_lambada_eval(self, dataset_path: str = None) -> Dict[str, float]:
        """Evaluate on LAMBADA dataset (reading comprehension)"""
        # Simplified LAMBADA evaluation
        examples = [
            {
                "text": "George Washington was the first President of the United States. He served from 1789 to 1797. Washington was born in",
                "target": "Virginia"
            },
            # Add more examples...
        ]
        
        correct = 0
        total = 0
        
        for example in examples:
            text = example["text"]
            target = example["target"]
            
            # Generate continuation
            config = EvaluationConfig(max_new_tokens=10, temperature=0.0, do_sample=False)
            generation = self.generation_evaluator.generate_text(text, config)
            
            # Check if target word appears in generation
            if target.lower() in generation.lower():
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        return {"lambada_accuracy": accuracy}
    
    def run_comprehensive_eval(self) -> Dict[str, Any]:
        """Run comprehensive evaluation suite"""
        results = {}
        
        # Perplexity on sample texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world in unprecedented ways.",
            "Climate change represents one of the most significant challenges of our time."
        ]
        
        perplexity_results = self.perplexity_evaluator.evaluate_perplexity(sample_texts)
        results.update(perplexity_results)
        
        # Common sense reasoning
        hellaswag_results = self.run_hellaswag_eval()
        results.update(hellaswag_results)
        
        # Reading comprehension
        lambada_results = self.run_lambada_eval()
        results.update(lambada_results)
        
        # Generation quality
        prompts = [
            "Explain the concept of machine learning in simple terms:",
            "Write a short story about a robot discovering emotions:",
            "Describe the benefits of renewable energy:"
        ]
        
        generation_results = self.generation_evaluator.evaluate_generation_quality(prompts)
        results['generation_examples'] = generation_results['generations']
        results.update(generation_results['metrics'])
        
        return results


def run_evaluation(model, tokenizer, device, output_dir: str = "eval_results"):
    """Run complete evaluation suite"""
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = BenchmarkEvaluator(model, tokenizer, device)
    
    print("Running comprehensive evaluation...")
    start_time = time.time()
    
    results = evaluator.run_comprehensive_eval()
    
    end_time = time.time()
    results['evaluation_time'] = end_time - start_time
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed in {results['evaluation_time']:.2f} seconds")
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    if 'perplexity' in results:
        print(f"Perplexity: {results['perplexity']:.2f}")
    if 'hellaswag_accuracy' in results:
        print(f"HellaSwag Accuracy: {results['hellaswag_accuracy']:.3f}")
    if 'lambada_accuracy' in results:
        print(f"LAMBADA Accuracy: {results['lambada_accuracy']:.3f}")
    if 'bleu' in results:
        print(f"BLEU Score: {results['bleu']:.2f}")
    
    return results
