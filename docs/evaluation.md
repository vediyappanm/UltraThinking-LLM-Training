# Evaluation

## Built-in benchmark runner
```python
from src.evaluation.benchmarks import ComprehensiveBenchmarkSuite, BenchmarkConfig

suite = ComprehensiveBenchmarkSuite(BenchmarkConfig())
results = suite.run_all_benchmarks(model, datasets)
print(results["summary"])
```

## Practical tips
- Some public benchmarks require specific data fields; dummy/val sets may be incompatible.
- Run evaluations periodically: `--eval_frequency 1` during early runs to verify trends.
- Log results to W&B when available.
