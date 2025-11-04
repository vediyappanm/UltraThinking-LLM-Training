from __future__ import annotations

from typing import Dict, Any


def run_lmeval(model_name_or_path: str, tasks: str = "mmlu,hellaswag,truthfulqa,gsm8k") -> Dict[str, Any]:
    try:
        from lm_eval import evaluator, tasks as lm_tasks  # type: ignore
    except Exception:
        raise ImportError("lm-eval not installed")
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=f"pretrained={model_name_or_path}",
        tasks=task_list,
        batch_size=1,
    )
    return results
