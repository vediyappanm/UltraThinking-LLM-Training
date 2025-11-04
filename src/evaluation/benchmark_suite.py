"""
Comprehensive Benchmark Evaluation Suite

Integrates with lm-evaluation-harness for standardized benchmarking
across multiple tasks: MMLU, HellaSwag, TruthfulQA, GSM8K, HumanEval, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    logger.warning("lm-eval not available. Install: pip install lm-eval")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""
    num_fewshot: int = 5  # Number of few-shot examples
    batch_size: int = 8  # Batch size for evaluation
    device: str = "cuda"  # Device to use
    limit: Optional[int] = None  # Limit number of examples (for testing)


class BenchmarkSuite:
    """
    Comprehensive evaluation suite for language models
    
    Supported Benchmarks:
    - MMLU: Massive Multitask Language Understanding (57 tasks)
    - HellaSwag: Commonsense reasoning
    - ARC: AI2 Reasoning Challenge
    - TruthfulQA: Truthfulness evaluation
    - GSM8K: Grade school math
    - HumanEval: Code generation (Python)
    - MBPP: Basic Python programming
    - ToxiGen: Toxicity detection
    
    Example:
        >>> suite = BenchmarkSuite("path/to/model")
        >>> results = suite.evaluate(["mmlu", "hellaswag", "gsm8k"])
        >>> suite.print_results(results)
    """
    
    # Available benchmarks with descriptions
    BENCHMARKS = {
        # Language Understanding
        "mmlu": {
            "name": "MMLU",
            "description": "Massive Multitask Language Understanding (57 subjects)",
            "metric": "accuracy",
            "baseline_gpt35": 70.0,
        },
        "hellaswag": {
            "name": "HellaSwag",
            "description": "Commonsense NLI and physical reasoning",
            "metric": "accuracy_norm",
            "baseline_gpt35": 85.5,
        },
        "arc_challenge": {
            "name": "ARC-Challenge",
            "description": "AI2 Reasoning Challenge (hard questions)",
            "metric": "accuracy_norm",
            "baseline_gpt35": 85.2,
        },
        "arc_easy": {
            "name": "ARC-Easy",
            "description": "AI2 Reasoning Challenge (easy questions)",
            "metric": "accuracy",
            "baseline_gpt35": 96.4,
        },
        
        # Truthfulness & Safety
        "truthfulqa_mc": {
            "name": "TruthfulQA",
            "description": "Truthfulness in question answering",
            "metric": "mc2",
            "baseline_gpt35": 47.0,
        },
        "toxigen": {
            "name": "ToxiGen",
            "description": "Toxicity detection",
            "metric": "accuracy",
            "baseline_gpt35": 85.0,
        },
        
        # Math & Reasoning
        "gsm8k": {
            "name": "GSM8K",
            "description": "Grade school math word problems",
            "metric": "accuracy",
            "baseline_gpt35": 57.1,
        },
        "mathqa": {
            "name": "MathQA",
            "description": "Math word problems",
            "metric": "accuracy",
            "baseline_gpt35": 45.0,
        },
        
        # Code Generation
        "humaneval": {
            "name": "HumanEval",
            "description": "Python code generation",
            "metric": "pass@1",
            "baseline_gpt35": 48.1,
        },
        "mbpp": {
            "name": "MBPP",
            "description": "Mostly Basic Python Problems",
            "metric": "accuracy",
            "baseline_gpt35": 52.2,
        },
    }
    
    def __init__(
        self,
        model_path: str,
        config: Optional[BenchmarkConfig] = None,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize benchmark suite
        
        Args:
            model_path: Path to model or HuggingFace model name
            config: Benchmark configuration
            tokenizer_path: Path to tokenizer (optional)
        """
        if not LM_EVAL_AVAILABLE:
            raise ImportError("lm-eval required. Install: pip install lm-eval")
        
        self.model_path = model_path
        self.config = config or BenchmarkConfig()
        self.tokenizer_path = tokenizer_path or model_path
        
        logger.info(f"Initializing benchmark suite for model: {model_path}")
        
        # Load model
        self.model = HFLM(
            pretrained=model_path,
            tokenizer=self.tokenizer_path,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )
        
        logger.info("Model loaded successfully")
    
    def evaluate(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on specified tasks
        
        Args:
            tasks: List of task names (default: all)
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples per task
            
        Returns:
            Dictionary with results
            
        Example:
            >>> results = suite.evaluate(["mmlu", "hellaswag", "gsm8k"])
        """
        if tasks is None:
            tasks = list(self.BENCHMARKS.keys())
        
        # Validate tasks
        invalid_tasks = [t for t in tasks if t not in self.BENCHMARKS]
        if invalid_tasks:
            logger.warning(f"Unknown tasks will be skipped: {invalid_tasks}")
            tasks = [t for t in tasks if t in self.BENCHMARKS]
        
        if not tasks:
            raise ValueError("No valid tasks specified")
        
        logger.info(f"Evaluating on tasks: {tasks}")
        
        # Run evaluation
        num_fewshot = num_fewshot or self.config.num_fewshot
        limit = limit or self.config.limit
        
        results = evaluator.simple_evaluate(
            model=self.model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=self.config.batch_size,
        )
        
        # Format results
        formatted_results = self._format_results(results)
        
        logger.info("Evaluation complete")
        return formatted_results
    
    def _format_results(self, raw_results: Dict) -> Dict[str, Any]:
        """Format raw evaluation results"""
        formatted = {
            "model": self.model_path,
            "config": {
                "num_fewshot": self.config.num_fewshot,
                "batch_size": self.config.batch_size,
            },
            "results": {}
        }
        
        for task, metrics in raw_results.get("results", {}).items():
            if task not in self.BENCHMARKS:
                continue
            
            benchmark_info = self.BENCHMARKS[task]
            metric_key = benchmark_info["metric"]
            
            # Extract primary metric
            score = metrics.get(metric_key, metrics.get("acc", 0.0))
            if isinstance(score, (int, float)):
                score = float(score) * 100  # Convert to percentage
            
            formatted["results"][task] = {
                "name": benchmark_info["name"],
                "description": benchmark_info["description"],
                "score": score,
                "metric": metric_key,
                "baseline_gpt35": benchmark_info.get("baseline_gpt35", 0),
                "vs_baseline": score - benchmark_info.get("baseline_gpt35", 0),
            }
        
        return formatted
    
    def print_results(self, results: Dict[str, Any]):
        """Print results in a formatted table"""
        print("\n" + "="*80)
        print(f"Benchmark Results for: {results['model']}")
        print("="*80)
        
        # Print config
        config = results["config"]
        print(f"Config: {config['num_fewshot']}-shot, batch_size={config['batch_size']}")
        print("-"*80)
        
        # Print results table
        print(f"{'Benchmark':<20} {'Score':<10} {'GPT-3.5':<10} {'Difference':<10}")
        print("-"*80)
        
        for task, data in results["results"].items():
            score = data["score"]
            baseline = data["baseline_gpt35"]
            diff = data["vs_baseline"]
            
            diff_str = f"{diff:+.1f}"
            if diff > 0:
                diff_str = f"✓ {diff_str}"
            elif diff < 0:
                diff_str = f"✗ {diff_str}"
            
            print(f"{data['name']:<20} {score:>6.1f}%   {baseline:>6.1f}%   {diff_str:<10}")
        
        print("="*80 + "\n")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def compare_checkpoints(
        self,
        checkpoint_paths: List[str],
        tasks: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple model checkpoints
        
        Args:
            checkpoint_paths: List of paths to checkpoints
            tasks: Tasks to evaluate on
            output_path: Path to save comparison results
            
        Returns:
            Comparison results
            
        Example:
            >>> suite.compare_checkpoints([
            ...     "checkpoint-1000",
            ...     "checkpoint-2000",
            ...     "checkpoint-3000"
            ... ])
        """
        logger.info(f"Comparing {len(checkpoint_paths)} checkpoints")
        
        all_results = {}
        
        for path in checkpoint_paths:
            logger.info(f"Evaluating checkpoint: {path}")
            
            # Create new suite for this checkpoint
            checkpoint_suite = BenchmarkSuite(path, self.config)
            results = checkpoint_suite.evaluate(tasks)
            
            all_results[path] = results
        
        # Create comparison table
        comparison = self._create_comparison(all_results, tasks or list(self.BENCHMARKS.keys()))
        
        if output_path:
            self.save_results(comparison, output_path)
        
        return comparison
    
    def _create_comparison(
        self,
        all_results: Dict[str, Dict],
        tasks: List[str]
    ) -> Dict[str, Any]:
        """Create comparison table from multiple results"""
        comparison = {
            "checkpoints": list(all_results.keys()),
            "tasks": tasks,
            "comparison": {}
        }
        
        for task in tasks:
            if task not in self.BENCHMARKS:
                continue
            
            task_comparison = {
                "name": self.BENCHMARKS[task]["name"],
                "scores": {}
            }
            
            for checkpoint, results in all_results.items():
                if task in results.get("results", {}):
                    task_comparison["scores"][checkpoint] = results["results"][task]["score"]
            
            comparison["comparison"][task] = task_comparison
        
        return comparison


def quick_eval(
    model_path: str,
    tasks: Optional[List[str]] = None,
    num_fewshot: int = 5
) -> Dict[str, Any]:
    """
    Quick evaluation function
    
    Args:
        model_path: Path to model
        tasks: List of tasks (default: mmlu, hellaswag, gsm8k)
        num_fewshot: Number of few-shot examples
        
    Returns:
        Evaluation results
        
    Example:
        >>> results = quick_eval("gpt2", ["mmlu", "hellaswag"])
    """
    if tasks is None:
        tasks = ["mmlu", "hellaswag", "gsm8k"]
    
    config = BenchmarkConfig(num_fewshot=num_fewshot)
    suite = BenchmarkSuite(model_path, config)
    results = suite.evaluate(tasks)
    suite.print_results(results)
    
    return results


# Example usage
if __name__ == "__main__":
    print("Benchmark Suite - Example usage:")
    print("""
    from src.evaluation.benchmark_suite import BenchmarkSuite, quick_eval
    
    # Quick evaluation
    results = quick_eval(
        "path/to/model",
        tasks=["mmlu", "hellaswag", "gsm8k", "humaneval"]
    )
    
    # Advanced usage
    from src.evaluation.benchmark_suite import BenchmarkConfig
    
    config = BenchmarkConfig(
        num_fewshot=5,
        batch_size=8,
        device="cuda"
    )
    
    suite = BenchmarkSuite("path/to/model", config)
    
    # Evaluate on specific tasks
    results = suite.evaluate([
        "mmlu",           # Language understanding
        "hellaswag",      # Commonsense reasoning
        "truthfulqa_mc",  # Truthfulness
        "gsm8k",          # Math
        "humaneval"       # Code
    ])
    
    # Print formatted results
    suite.print_results(results)
    
    # Save to file
    suite.save_results(results, "eval_results.json")
    
    # Compare checkpoints
    comparison = suite.compare_checkpoints([
        "checkpoint-1000",
        "checkpoint-2000",
        "checkpoint-3000"
    ])
    
    # Output:
    # ================================================================================
    # Benchmark Results for: path/to/model
    # ================================================================================
    # Benchmark            Score      GPT-3.5    Difference
    # --------------------------------------------------------------------------------
    # MMLU                  45.2%      70.0%     ✗ -24.8   
    # HellaSwag             68.3%      85.5%     ✗ -17.2   
    # GSM8K                 23.1%      57.1%     ✗ -34.0   
    # HumanEval             15.5%      48.1%     ✗ -32.6   
    # ================================================================================
    """)
